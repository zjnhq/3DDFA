#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img_with_coord, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, crop_back_img
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
from vdc_loss import VDCLoss
from wpdc_loss import WPDCLoss
import argparse
import torch.backends.cudnn as cudnn
from train import GBDT_Predictor
from pdb import *
from sklearn.ensemble import *
import pickle as pkl
STD_SIZE = 120


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    # checkpoint_fp = 'models/phase1_wpdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    if args.gbdt==1:
        feat_index_filename = 'important_feature.pkl'
        gbdt_param_filename = './gbdt_param/gbdt_param'
        gbdt_pred = GBDT_Predictor(feat_index_filename, gbdt_param_filename)
    if args.loss.lower() == 'wpdc':
        print(args.opt_style)
        criterion = WPDCLoss(opt_style=args.opt_style).cuda()
        # logging.info('Use WPDC Loss')
    elif args.loss.lower() == 'vdc':
        criterion = VDCLoss(opt_style=args.opt_style).cuda()
        # logging.info('Use VDC Loss')
    elif args.loss.lower() == 'pdc':
        criterion = nn.MSELoss(size_average=args.size_average).cuda()
        # logging.info('Use PDC loss')
    else:
        raise Exception(f'Unknown Loss {args.loss}')

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    normalizer = NormalizeGjz(mean=127.5, std=128)
    transform = transforms.Compose([ToTensorGjz(), normalizer])
    for img_fp in args.files:
        for gbdt in [0,1]:
            for attack in [0,1]:
                prefix = '_orig'
                if attack == 0:
                    prefix = '_attack'
                if gbdt == 0:
                    prefix = prefix+ '_cnn'
                else:
                    prefix = prefix+ '_gbdt'
                img_ori = cv2.imread(img_fp)
                if args.dlib_bbox:
                    rects = face_detector(img_ori, 1)
                else:
                    rects = []

                if len(rects) == 0:
                    rects = dlib.rectangles()
                    rect_fp = img_fp + '.bbox'
                    lines = open(rect_fp).read().strip().split('\n')[1:]
                    for l in lines:
                        l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                        rect = dlib.rectangle(l, r, t, b)
                        rects.append(rect)

                pts_res = []
                Ps = []  # Camera matrix collection
                poses = []  # pose collection, [todo: validate it]
                vertices_lst = []  # store multiple face vertices
                ind = 0
                suffix = get_suffix(img_fp)
                print("save suffix:"+suffix)
                for rect in rects:
                    # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
                    if args.dlib_landmark:
                        # - use landmark for cropping
                        pts = face_regressor(img_ori, rect).parts()
                        pts = np.array([[pt.x, pt.y] for pt in pts]).T
                        roi_box = parse_roi_box_from_landmark(pts)
                    else:
                        # - use detected face bbox
                        bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                        roi_box = parse_roi_box_from_bbox(bbox)

                    crop_coord = [0]
                    img, crop_coord = crop_img_with_coord(img_ori, roi_box)
                    original_size = img.shape

                    # forward: one step
                    img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                    input = transform(img).unsqueeze(0)
                    
                    if args.mode == 'gpu':
                        input = input.cuda()
                    input.requires_grad = True
                    input.retain_grad()
                    param = model(input)
                    # set_trace()
                    target = param.clone().detach()
                    # set_trace()
                    target = target + torch.tensor(np.random.randn(target.shape[0],target.shape[1])).type(torch.FloatTensor).cuda()
                    loss = criterion(param, target)
                    # losses_cnn_original.append(loss.item())
                    if attack==1:
                        input_original = input.clone().detach()
                        # set_trace()
                        attack_maxstepsize = 0.03 #np.abs(input_original.cpu().numpy()).mean()*0.02
                        input_upper_limit= input_original + attack_maxstepsize
                        input_lower_limit = input_original - attack_maxstepsize
                        steps = 10
                        attack_stepsize = attack_maxstepsize/steps
                        # attack_stepsize = 100000.0
                        input.retain_grad()
                        for attack_iter in range(steps):
                            loss = criterion(param, target)
                            loss.backward()
                            input.detach()
                            input.data = input.data - input.grad.sign() * attack_stepsize
                            input.data[input.data>input_upper_limit] = input_upper_limit.data[input.data>input_upper_limit]
                            input.data[input.data<input_lower_limit] = input_lower_limit.data[input.data<input_lower_limit]
                            input.retain_grad()
                            param = model(input)    
                            # set_trace()
                    if gbdt==1:
                        param = gbdt_pred.predict(input, model.mid_features)
                    param = param.detach().squeeze().cpu().numpy().flatten().astype(np.float32)

                    if attack == 1:
                        reverse_input = normalizer.reverse(input.detach())
                        img_attacked = reverse_input.cpu().numpy()
                        img_attacked = img_attacked.reshape(3, STD_SIZE, STD_SIZE).transpose(1,2,0)
                        # set_trace()
                        diff = img_attacked - img
                        set_trace()
                        img_attacked = cv2.resize(img_attacked, dsize=(original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR)
                        crop_back_img(img_ori, roi_box, img_attacked, crop_coord)

                    # 68 pts
                    pts68 = predict_68pts(param, roi_box)

                    # # two-step for more accurate bbox to crop face
                    # if args.bbox_init == 'two':
                    #     roi_box = parse_roi_box_from_landmark(pts68)
                    #     img_step2 = crop_img(img_ori, roi_box)
                    #     img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                    #     input = transform(img_step2).unsqueeze(0)
                    #     with torch.no_grad():
                    #         if args.mode == 'gpu':
                    #             input = input.cuda()
                    #         param = model(input)
                    #         set_trace()
                    #         if args.gbdt==1:
                    #             # set_trace()
                    #             rawfeat = input.cpu().detach().numpy().reshape(-1)[important_rawfeature]
                    #             batched_gbdt_features[0,:num_feat] = rawfeat
                    #             midfeat = model.mid_features.cpu().detach().numpy().reshape(-1)[important_midfeature]
                    #             batched_gbdt_features[0,num_feat:] = midfeat
                    #             for d in range(target_dim):
                    #                 batched_gbdt_predict[0,d] = lightgbms[d].predict(batched_gbdt_features)
                    #                 gbdt_predict =torch.tensor(batched_gbdt_predict)
                    #                 if args.mode == 'gpu': gbdt_predict = gbdt_predict.cuda()
                    #                 param = param * (1- args.alpha) + gbdt_predict* args.alpha
                    #                 # output = output * (1- alpha) + torch.tensor(batched_gbdt_predict).cuda()* alpha
                    #             # set_trace()

                    #         param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                    #     pts68 = predict_68pts(param, roi_box)

                    pts_res.append(pts68)
                    P, pose = parse_pose(param)
                    Ps.append(P)
                    poses.append(pose)

                    # dense face 3d vertices
                    if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                        vertices = predict_dense(param, roi_box)
                        vertices_lst.append(vertices)
                    if args.dump_ply:
                        fname = '{}_{}_{}.ply'.format(img_fp.replace(suffix, ''), ind, prefix)
                        dump_to_ply(vertices, tri, fname)
                    if args.dump_vertex:
                        dump_vertex(vertices, '{}_{}_{}.mat'.format(img_fp.replace(suffix, ''), ind, prefix))
                    if args.dump_pts:
                        wfp = '{}_{}_{}.txt'.format(img_fp.replace(suffix, ''), ind, prefix)
                        np.savetxt(wfp, pts68, fmt='%.3f')
                        print('Save 68 3d landmarks to {}'.format(wfp))
                    if args.dump_roi_box:
                        wfp = '{}_{}_{}.roibox'.format(img_fp.replace(suffix, ''), ind, prefix)
                        np.savetxt(wfp, roi_box, fmt='%.3f')
                        print('Save roi box to {}'.format(wfp))
                    if args.dump_paf:
                        wfp_paf = '{}_{}_{}_paf.jpg'.format(img_fp.replace(suffix, ''), ind, prefix)
                        wfp_crop = '{}_{}_{}_crop.jpg'.format(img_fp.replace(suffix, ''), ind, prefix)
                        paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

                        cv2.imwrite(wfp_paf, paf_feature)
                        cv2.imwrite(wfp_crop, img)
                        print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
                    if args.dump_obj:
                        wfp = '{}_{}_{}.obj'.format(img_fp.replace(suffix, ''), ind, prefix)
                        colors = get_colors(img_ori, vertices)
                        write_obj_with_colors(wfp, vertices, tri, colors)
                        print('Dump obj with sampled texture to {}'.format(wfp))
                    ind += 1

                if args.dump_pose:
                    # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
                    img_pose = plot_pose_box(img_ori, Ps, pts_res)
                    wfp = img_fp.replace(suffix, '_pose') + prefix+ '.jpg'
                    cv2.imwrite(wfp, img_pose)
                    print('Dump to {}'.format(wfp))
                if args.dump_depth:
                    wfp = img_fp.replace(suffix, '_depth') + prefix+ '.png'
                    # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
                    depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
                    cv2.imwrite(wfp, depths_img)
                    print('Dump to {}'.format(wfp))
                if args.dump_pncc:
                    wfp =  img_fp.replace(suffix, '_pncc')+ prefix+ '.png'
                    pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
                    cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
                    print('Dump to {}'.format(wfp))
                if args.dump_res:
                    draw_landmarks(img_ori, pts_res, wfp=img_fp.replace(suffix, prefix+'.jpg'), show_flg=args.show_flg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('--gbdt', default=0, type=int)
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='true', type=str2bool)
    parser.add_argument('--dump_depth', default='true', type=str2bool)
    parser.add_argument('--dump_pncc', default='true', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')
    parser.add_argument('--opt-style', default='resample', type=str)  # resample
    parser.add_argument('--loss', default='vdc', type=str)

    args = parser.parse_args()
    main(args)
