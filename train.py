#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mobilenet_v1
import torch.backends.cudnn as cudnn
import scipy.io as sio
from utils.ddfa import DDFADataset, DDFAPlotDataset, ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from vdc_loss import VDCLoss
from wpdc_loss import WPDCLoss
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf

import pickle as pkl
import sys
# print(sys.path.append('D:/code/forest/scikit-learn'))
from gradient_boosting import HistGradientBoostingRegressor
from pdb import *
# global args (configuration)
args = None
lr = None
arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']

feat_index_filename = 'important_feature.pkl'
gbdt_param_filename = './gbdt_param/gbdt_param'

def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('--gbdt', default=0, type=int)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=32, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='0,1', type=str)
    parser.add_argument('--filelists-train',
                        default='', type=str)
    parser.add_argument('--filelists-val',
                        default='', type=str)
    parser.add_argument('--root', default='')
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--size-average', default='true', type=str2bool)
    parser.add_argument('--num-classes', default=62, type=int)
    parser.add_argument('--arch', default='mobilenet_1', type=str,
                        choices=arch_choices)
    parser.add_argument('--frozen', default='false', type=str2bool)
    parser.add_argument('--milestones', default='15,25,30', type=str)
    parser.add_argument('--task', default='all', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=-1, type=int)
    parser.add_argument('--param-fp-train',
                        default='',
                        type=str)
    parser.add_argument('--param-fp-val',
                        default='')
    parser.add_argument('--opt-style', default='resample', type=str)  # resample
    parser.add_argument('--resample-num', default=132, type=int)
    parser.add_argument('--loss', default='vdc', type=str)
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
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')

    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    # loader is batch style
    # for i, (input, target) in enumerate(train_loader):
    for i, (input, target) in enumerate(train_loader):
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        output = model(input)

        data_time.update(time.time() - end)

        if args.loss.lower() == 'vdc':
            loss = criterion(output, target)
        elif args.loss.lower() == 'wpdc':
            loss = criterion(output, target)
        elif args.loss.lower() == 'pdc':
            loss = criterion(output, target)
        else:
            raise Exception(f'Unknown loss {args.loss}')

        losses.update(loss.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log
        if i % args.print_freq == 0:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                         f'LR: {lr:8f}\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         f'Loss {losses.val:.4f} ({losses.avg:.4f})')

def prepare_gbdt(train_loader, model, criterion, optimizer, args):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # model.train()

    end = time.time()
    # loader is batch style
    # for i, (input, target) in enumerate(train_loader):

    # build a batch of raw/mid features for GBDT training
    batch_size = args.batch_size
    
    rawfeature_dim = 0
    midfeature_dim = 0
    mid_features = 0 
    target_dim = 0
    model.eval()
    for i, (input, target) in enumerate(train_loader):
        rawfeature_dim =  input.shape[1] * input.shape[2] *input.shape[3] 
        target_dim = target.shape[1]
        output = model(input)
        mid_features = model.module.mid_features
        midfeature_dim = mid_features.shape[1]* mid_features.shape[2] * mid_features.shape[3] 
        # set_trace()
        del model.module.mid_features
        break
    torch.cuda.empty_cache()
    feature_dim = rawfeature_dim + midfeature_dim
    feature_idx = 0
    batched_mid_features= np.zeros([batch_size, feature_dim])
    
    # try to accumulate the gradient abs, to select stable features of small gradients
    model = model.cuda()
    model.module.SetMidfeatureNeedGrad(True)
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        input.requires_grad = True
        input.retain_grad()
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        grad = input.grad.cpu().detach().numpy().reshape(batch_size,-1)
        batched_mid_features[:,:rawfeature_dim] += np.abs(grad).reshape(batch_size,-1)
        grad2 = model.module.mid_features.grad.cpu().detach().numpy()
        batched_mid_features[:,rawfeature_dim:] += np.abs(grad2).reshape(batch_size,-1)
        del input
        if i>1000:break
    del model.module.mid_features
    torch.cuda.empty_cache()
    
    # set_trace()
    feature_importance = batched_mid_features.mean(axis=0)
    num_feat = int(min(rawfeature_dim/8, midfeature_dim/4))
    important_rawfeature = np.argpartition(feature_importance[:rawfeature_dim], num_feat)[: num_feat]
    important_midfeature = np.argpartition(feature_importance[rawfeature_dim:], num_feat)[: num_feat]
    # important_midfeature = important_midfeature + rawfeature_dim
    # important_rawfeature = np.argpartition(feature_importance, - num_feat)[- num_feat:]
    # important_feature= [important_rawfeature, important_midfeature]
    print("select raw and mid feature:"+str(num_feat))
    pkl.dump([important_rawfeature, important_midfeature, num_feat, target_dim], open(feat_index_filename,'wb'))

def generate_gbdt_dataset(train_loader, model, criterion, optimizer, args):

    # if args.loss.lower() == 'vdc':
    #     loss = criterion(output, target)
    # elif args.loss.lower() == 'wpdc':
    #     loss = criterion(output, target)
    # elif args.loss.lower() == 'pdc':
    #     loss = criterion(output, target)
    # else:
    #     raise Exception(f'Unknown loss {args.loss}')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # model.train()

    end = time.time()
    # loader is batch style
    # for i, (input, target) in enumerate(train_loader):

    # build a batch of raw/mid features for GBDT training
    batch_size = args.batch_size

    [important_rawfeature, important_midfeature, num_feat, target_dim]=pkl.load(open(feat_index_filename,'rb'))

    feature_dim = important_rawfeature.shape[0] + important_midfeature.shape[0]

    num_batch_feat_gbdt = 1000
    batched_feature_sz = int(batch_size * num_batch_feat_gbdt)
    batched_mid_features = np.zeros([batched_feature_sz, feature_dim])
    # target_dim = 62
    batched_target = np.zeros([batched_feature_sz, target_dim])
    feature_idx = 0

    model.eval()
    # model = model.cpu()
    torch.set_num_interop_threads(8)
    # model.module = model.module.cpu()
    model = model.cuda()
    model.module.SetMidfeatureNeedGrad(False)
    fileid = 0

    for i, (input, target) in enumerate(train_loader):
        if(feature_idx + batch_size > batched_feature_sz):
            feature_idx = 0
            filename = './gbdt_feature/gbdt_feature' +str(fileid)+'.pkl'
            fileid += 1
            pkl.dump([batched_mid_features, batched_target], open(filename,'wb'))
            if fileid>10:
                break

        target = target.cuda()
        input = input.cuda()#.cuda(non_blocking=True)
        target.requires_grad = False
        output = model(input)
        rawfeat = input.cpu().detach().numpy().reshape(batch_size,-1)[:,important_rawfeature]
        batched_mid_features[feature_idx:feature_idx + batch_size, :num_feat] = rawfeat
        midfeat = model.module.mid_features.cpu().detach().numpy().reshape(batch_size,-1)[:,important_midfeature]
        batched_mid_features[feature_idx:feature_idx + batch_size, num_feat:] = midfeat
        batched_target[feature_idx:feature_idx + batch_size] = target.cpu().detach().numpy()
        feature_idx += batch_size
        batch_time.update(time.time() - end)
        end = time.time()
        # log
        if i % args.print_freq == 0:
            logging.info(f'file id: [{fileid}][{i}/{len(train_loader)}]\t'
                         # f'LR: {lr:8f}\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         # f'Loss {losses.val:.4f} ({losses.avg:.4f})'
                         )
        del input
        del model.module.mid_features

num_init_trees = 30
num_pkl_files = 2
num_tree_increase = num_init_trees#3 
# target_dim = 1
def train_gbdt(train_loader, model, criterion, optimizer, args):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # build a batch of raw/mid features for GBDT training
    fileid = 0
    filename = './gbdt_feature/gbdt_feature' +str(fileid)+'.pkl'
    
    [batched_mid_features, batched_target] = pkl.load(open(filename,'rb'))
    # print(batched_mid_features[:3,:3])
    
    lightgbms = []
    target_dim = batched_target.shape[1]
    max_leaf_nodes = 63
    max_bins = 31
    end =  time.time()
    for i in range(target_dim):
        # print("training gbdt for target dim:"+str(i))
        verbose = 0
        if i%15==0:
            verbose=1
        lightgbm = HistGradientBoostingRegressor(max_iter=num_init_trees, learning_rate=0.2, max_leaf_nodes=max_leaf_nodes, max_bins=max_bins, warm_start = False, verbose=verbose)
        # lightgbm = GradientBoostingRegressor(n_estimators=20, max_depth=6)

        lightgbm.fit(batched_mid_features, batched_target[:,i])
        pkl.dump(lightgbm, open(gbdt_param_filename+str(i)+'.pkl','wb'))
        lightgbms.append(lightgbm)
        elapse = time.time() - end
        logging.info(f'trained gbdt for target dim: [{i}][ time:{elapse/60:.2f}]\t')

def refine_gbdt(train_loader, model, criterion, optimizer, args):

    
    lightgbms = []
    target_dim = 62
    for i in range(target_dim):
        lightgbm = pkl.load(open(gbdt_param_filename+str(i)+'.pkl','rb'))
        lightgbms.append(lightgbm)

    print("original LightGBM paramers:")
    print(lightgbms[0].get_params())
    # target_dim = 0
    end =  time.time()
    for fileid in range(1, num_pkl_files):
        filename = './gbdt_feature/gbdt_feature' +str(fileid)+'.pkl'
        [batched_mid_features, batched_target] = pkl.load(open(filename,'rb'))
        logging.info("refine gbdt on feature file:"+filename)
        n_samples = batched_mid_features.shape[0]
        logging.info(f'with [{n_samples}] new training samples')
        # set_trace()
        for i in range(target_dim):
            params = lightgbms[i].get_params()
            params['max_iter'] = params['max_iter'] + num_tree_increase # for every batch data, increase 2 trees
            params['max_bins'] = max(int((params['max_bins'] + 1) /2  - 1), 7)

            # params['max_bins'] = params['max_bins'] + params['max_bins'] + 1 # increase like 15,31,63
            params['max_leaf_nodes'] = params['max_leaf_nodes'] + params['max_leaf_nodes'] +1
            params['warm_start'] = True 
            lightgbms[i].set_params(**params)
            lightgbms[i].fit(batched_mid_features, batched_target[:,i])
            elapse = time.time() - end
            end = time.time()
            logging.info(f'refined gbdt for target dim: [{i}][ time:{elapse/60:.2f}]')
            pkl.dump(lightgbms[i], open(gbdt_param_filename + '_ref'+str(i)+'.pkl','wb'))
    
        # for i in range(target_dim):
        #     pkl.dump(lightgbms[i], open(gbdt_param_filename + '_ref'+str(i)+'.pkl','wb'))
        print("refined LightGBM paramers:")
        print(lightgbms[0].get_params())
        print("saved gbdt after refinement.")

max_batches_for_eval = 450

def validate(val_loader, model, criterion, epoch, args):
    model.eval()

    end = time.time()
    with torch.no_grad():
        losses = []
        for i, (input, target) in enumerate(val_loader):
            # compute output
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            output = model(input)

            loss = criterion(output, target)
            losses.append(loss.item())
            # if i> max_batches_for_eval: break

        elapse = time.time() - end
        loss = np.mean(losses)
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'Loss {loss:.4f}\t'
                     f'Time {elapse:.3f}')

class GBDT_Predictor:
    def __init__(self, feat_index_filename, gbdt_param_filename):
        [self.rawfeature_index, self.midfeature_index, self.num_feat, self.target_dim] = pkl.load(open(feat_index_filename,'rb'))
        batch_size = 32
        self.feature_dim = self.rawfeature_index.shape[0] + self.midfeature_index.shape[0]
        self.batched_gbdt_features = np.zeros([batch_size, self.feature_dim])
        self.batched_gbdt_predict = np.zeros([batch_size, self.target_dim])
        if self.num_feat != self.rawfeature_index.shape[0]:
            print('num_feat != self.rawfeature_index.shape[0] ')
        self.lightgbms = []
        for i in range(self.target_dim):
            self.lightgbm = pkl.load(open(gbdt_param_filename+ '_ref' + str(i) + '.pkl','rb'))
            # self.lightgbm = pkl.load(open(gbdt_param_filename+ str(i) + '.pkl','rb'))
            self.lightgbms.append(self.lightgbm)
            # set_trace()
    def predict(self,input, mid_features):
        batch_size = input.shape[0]
        if batch_size != self.batched_gbdt_features.shape[0]:
            self.batched_gbdt_features = np.zeros([batch_size, self.feature_dim])
            self.batched_gbdt_predict = np.zeros([batch_size, self.target_dim])
        rawfeat = input.cpu().detach().numpy().reshape(batch_size,-1)[:,self.rawfeature_index]
        
        self.batched_gbdt_features[:,:self.num_feat] = rawfeat
        midfeat = mid_features.cpu().detach().numpy().reshape(batch_size,-1)[:,self.midfeature_index]
        self.batched_gbdt_features[:, self.num_feat:] = midfeat
        for d in range(self.target_dim):
            self.batched_gbdt_predict[:,d] = self.lightgbms[d].predict(self.batched_gbdt_features)
        # output = output * (1- alpha) + torch.tensor(batched_gbdt_predict).cuda()* alpha
        return torch.tensor(self.batched_gbdt_predict).cuda()

def validate_gbdt(val_loader, model, criterion, args):
    
    alpha = args.alpha
    gbdt = GBDT_Predictor(feat_index_filename, gbdt_param_filename)
    # [important_rawfeature, important_midfeature, num_feat, target_dim] = pkl.load(open(feat_index_filename,'rb'))
    # lightgbms = []
    # for i in range(target_dim):
    #     lightgbm = pkl.load(open(gbdt_param_filename + str(i) + '.pkl','rb'))
    #     lightgbms.append(lightgbm)
    batch_size = args.val_batch_size
    
    num_batch_for_eval = 100
    
    model.module.SetMidfeatureNeedGrad(False)
    model.eval()

    end = time.time()
    # with torch.no_grad():
    losses_cnn_original = []
    losses_cnn_attacked = []
    losses_gbdt_original = []
    losses_gbdt_attacked = []
    use_attack = 1
    if use_attack ==0:
        losses_cnn_attacked.append(0.0)
        losses_gbdt_attacked.append(0.0)
    if use_attack==1:
        model.train()
        model.module.SetMidfeatureNeedGrad(True)
    for i, (input, target) in enumerate(val_loader):
        # compute output
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        if use_attack==1:input.requires_grad = True
        output = model(input)

        gbdt_output = gbdt.predict(input, model.module.mid_features)
        gbdt_output = output * (1- alpha) + gbdt_output* alpha
        
        loss = criterion(gbdt_output, target)
        losses_gbdt_original.append(loss.item())

        loss = criterion(output, target)
        losses_cnn_original.append(loss.item())

        if use_attack==1:
            input_original = input.clone().detach()
            attack_maxstepsize = 0.01#np.abs(input_original.cpu().numpy()).mean()*0.02
            input_upper_limit= input_original + attack_maxstepsize
            input_lower_limit = input_original - attack_maxstepsize
            steps = 10
            attack_stepsize = attack_maxstepsize/steps
            # attack_stepsize = 100000.0
            input.retain_grad()
            for attack_iter in range(steps):
                loss = criterion(output, target)
                loss.backward()
                # set_trace()
                # input.abs().mean()/(input.grad.abs().mean())
                input.detach()
                input.data = input.data + input.grad.sign() * attack_stepsize
                input.data[input.data>input_upper_limit] = input_upper_limit.data[input.data>input_upper_limit]
                input.data[input.data<input_lower_limit] = input_lower_limit.data[input.data<input_lower_limit]
                input.requires_grad= True
                input.retain_grad()
                output = model(input)    
            gbdt_output = gbdt.predict(input, model.module.mid_features)
            # gbdt_output = output * (1- alpha) + gbdt_output * alpha    
            loss = criterion(output, target)
            losses_cnn_attacked.append(loss.item())
            loss = criterion(gbdt_output, target)
            losses_gbdt_attacked.append(loss.item())

        # if i> max_batches_for_eval: break
        if i % args.print_freq==0:
            print("validate gbdt for:"+str(i))
            elapse = time.time() - end
            # loss = np.mean(losses)
            logging.info(f'alpha[{alpha}] Time {elapse/60:.3f} Val: [{i} /{len(val_loader)}]\t'
                         f'CNN Loss original {np.mean(losses_cnn_original)*100:.4f}\t'
                         f'CNN Loss attacked {np.mean(losses_cnn_attacked)*100:.4f}\t')
            logging.info(f'GBDT Loss original {np.mean(losses_gbdt_original)*100:.4f}\t'
                         f'GBDT Loss attacked {np.mean(losses_gbdt_attacked)*100:.4f}\t')
    elapse = time.time() - end
    # loss = np.mean(losses)
    logging.info(f'alpha[{alpha}] Time {elapse/60:.3f} Val: [{i} /{len(val_loader)}]\t'
                 f'CNN Loss original {np.mean(losses_cnn_original)*100:.4f}\t'
                 f'CNN Loss attacked {np.mean(losses_cnn_attacked)*100:.4f}\t')
    logging.info(f'GBDT Loss original {np.mean(losses_gbdt_original)*100:.4f}\t'
                 f'GBDT Loss attacked {np.mean(losses_gbdt_attacked)*100:.4f}\t')

def plot_gbdt(val_loader, model, criterion, args, use_gbdt, use_attack):
    
    alpha = args.alpha
    gbdt = GBDT_Predictor(feat_index_filename, gbdt_param_filename)

    model.module.SetMidfeatureNeedGrad(False)
    model.eval()

    end = time.time()
    # with torch.no_grad():
    losses_cnn_original = []
    losses_cnn_attacked = []
    losses_gbdt_original = []
    losses_gbdt_attacked = []
    if use_attack ==0:
        losses_cnn_attacked.append(0.0)
        losses_gbdt_attacked.append(0.0)
    if use_attack==1:
        model.train()
        model.module.SetMidfeatureNeedGrad(True)

    tri = sio.loadmat('visualize/tri.mat')['tri']
    plot_num = 0
    for i, (img, target, img_cv) in enumerate(val_loader):
        # set_trace()
        if i>100 or plot_num>10:break
        
        input = args.transform(img.clone())
        prefix = '_orig'
        if use_attack == 1:
            prefix = '_attack'
        if use_gbdt == 0:
            prefix = prefix + '_cnn'
        else:
            prefix = prefix + '_gbdt'
        # compute output
        outputdict = [0,0,0,0]
        lossdict = [0,0,0,0]
        prefixdict = [0,0,0,0]
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        img_fp = "plot/save"+str(i)+'.jpg'
        suffix = get_suffix(img_fp)
        if use_attack==1:input.requires_grad = True
        output = model(input)
        loss = criterion(output, target)
        testid = 0
        outputdict[testid] = output.clone().detach()
        lossdict[testid] = loss.item()
        prefixdict[testid] = 'orig_cnn'

        if use_gbdt==1: 
            testid+=1
            output_gbdt = gbdt.predict(input, model.module.mid_features)
            loss_gbdt = criterion(output_gbdt, target)
            outputdict[testid] = output_gbdt.clone().detach()
            lossdict[testid] += loss_gbdt.item()
            prefixdict[testid] = 'orig_gbdt'

        if use_attack==1:
            input_original = input.clone().detach()
            attack_maxstepsize = 0.01#np.abs(input_original.cpu().numpy()).mean()*0.02
            input_upper_limit= input_original + attack_maxstepsize
            input_lower_limit = input_original - attack_maxstepsize
            steps = 3
            attack_stepsize = attack_maxstepsize/steps
            # print("prediction before:"+str(output[0,-6:]))
            input.requires_grad = True
            input.retain_grad()
            for attack_iter in range(steps):
                loss = criterion(output, target)
                loss.backward()
                input.detach()
                input.data = input.data + input.grad.sign() * attack_stepsize
                input.data[input.data>input_upper_limit] = input_upper_limit.data[input.data>input_upper_limit]
                input.data[input.data<input_lower_limit] = input_lower_limit.data[input.data<input_lower_limit]
                input.requires_grad= True
                input.retain_grad()
                output = model(input)    
            # print("prediction after:"+str(output[0,-6:]))
        
            # gbdt_output = output * (1- alpha) + gbdt_output * alpha    
            loss = criterion(output, target)
            losses_cnn_attacked.append(loss.item())
            testid +=1
            outputdict[testid] = output.clone().detach()
            lossdict[testid] += loss.item()
            prefixdict[testid] = 'attack_cnn'

            if use_gbdt==1: 
                output_gbdt = gbdt.predict(input, model.module.mid_features)
                loss_gbdt = criterion(output_gbdt, target)
                testid +=1
                outputdict[testid] = output_gbdt.clone().detach()
                lossdict[testid] += loss_gbdt.item()
                prefixdict[testid] = 'attack_gbdt'
            # loss = criterion(gbdt_output, target)
            # losses_gbdt_attacked.append(loss.item())

        print(prefixdict)
        print(lossdict)
        plot_this = False
        if lossdict[1]< lossdict[0] and (lossdict[3]< lossdict[2]) and lossdict[2]<0.1 and lossdict[0]<0.1:
            plot_this = True
            plot_num +=1
        else:
            print("skip this pic")
        if plot_this:
            for j in range(testid+1):

                param = outputdict[j]
                prefix = prefixdict[j]
                param = param.squeeze().detach().cpu().numpy().flatten().astype(np.float32)
                # 68 pts
                h,w,nc = img.shape[2], img.shape[3], img.shape[1]
                # set_trace()
                if use_attack==0:
                    img_ori = img.reshape(nc,h,w).permute(1,2,0).cpu().numpy() # 3 x H x W
                else:
                    img_attack = args.transform.reverse(input.detach())
                    img_attack = img_attack.reshape(nc,h,w).permute(1,2,0).cpu().numpy()#.permute(2,0,1)
                    img_attack=img_attack.astype(int)
                    img_attack[img_attack>255] =255
                    img_attack[img_attack<0] = 0

                    img_ori= img_attack

                img_ori = img.reshape(nc,h,w).permute(1,2,0).cpu().numpy() # 3 x H x W

                img_ori = img_ori.astype(int) #.transpose(1,2,0)
                # img_ori = img_cv

                # bbox= [0, 0, img.shape[2], img.shape[3]]
                # bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                # set_trace()
                roi_box = [0, 0, img.shape[2], img.shape[3]] #parse_roi_box_from_bbox(bbox)
                pts68 = predict_68pts(param, roi_box)

                pts_res = []
                Ps = []  # Camera matrix collection
                poses = []  # pose collection, [todo: validate it]
                vertices_lst = []  # store multiple face vertices
                ind = 0
                suffix = get_suffix(img_fp)

                pts_res.append(pts68)
                P, pose = parse_pose(param)
                Ps.append(P)
                poses.append(pose)

                ind=i

                # dense face 3d vertices
                if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                    vertices = predict_dense(param, roi_box)
                    vertices_lst.append(vertices)
                # if args.dump_ply:
                #     fname = '{}_{}_{}.ply'.format(img_fp.replace(suffix, ''), ind, prefix)
                #     dump_to_ply(vertices, tri, fname)
                # if args.dump_vertex:
                #     dump_vertex(vertices, '{}_{}_{}.mat'.format(img_fp.replace(suffix, ''), ind, prefix))
                # if args.dump_pts:
                #     wfp = '{}_{}_{}.txt'.format(img_fp.replace(suffix, ''), ind, prefix)
                #     np.savetxt(wfp, pts68, fmt='%.3f')
                #     print('Save 68 3d landmarks to {}'.format(wfp))
                # if args.dump_roi_box:
                #     wfp = '{}_{}_{}.roibox'.format(img_fp.replace(suffix, ''), ind, prefix)
                #     np.savetxt(wfp, roi_box, fmt='%.3f')
                #     print('Save roi box to {}'.format(wfp))
                if args.dump_paf:
                    wfp_paf = '{}_{}_{}_paf.jpg'.format(img_fp.replace(suffix, ''), ind, prefix)
                    wfp_crop = '{}_{}_{}_crop.jpg'.format(img_fp.replace(suffix, ''), ind, prefix)
                    paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

                    cv2.imwrite(wfp_paf, paf_feature)
                    cv2.imwrite(wfp_crop, img)
                    print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
                # if args.dump_obj:
                #     wfp = '{}_{}_{}.obj'.format(img_fp.replace(suffix, ''), ind, prefix)
                #     colors = get_colors(img_ori, vertices)
                #     write_obj_with_colors(wfp, vertices, tri, colors)
                #     print('Dump obj with sampled texture to {}'.format(wfp))
                ind += 1

                # if args.dump_pose:
                #     # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
                #     img_pose = plot_pose_box(img_ori, Ps, pts_res)
                #     wfp = img_fp.replace(suffix, '_pose') + prefix+ '.jpg'
                #     cv2.imwrite(wfp, img_pose)
                #     print('Dump to {}'.format(wfp))
                if args.dump_depth:
                    wfp = img_fp.replace(suffix, '_depth') + prefix+ '.png'
                    # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
                    depths_img = cget_depths_image(img_ori.copy(), vertices_lst, tri - 1)  # cython version
                    cv2.imwrite(wfp, depths_img)
                    print('Dump to {}'.format(wfp))
                if args.dump_pncc:
                    wfp =  img_fp.replace(suffix, '_pncc')+ prefix+ '.png'
                    pncc_feature = cpncc(img_ori.copy(), vertices_lst, tri - 1)  # cython version
                    cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
                    print('Dump to {}'.format(wfp))
                # set_trace()
                if args.dump_res:
                    note=prefix+" loss:{:.2f}".format(lossdict[j])
                    # set_trace()
                    draw_landmarks(img_ori.copy(), pts_res, wfp=img_fp.replace(suffix, prefix+'.jpg'), show_flg=args.show_flg, note =note)
                    # draw_landmarks(img_cv, pts_res, wfp=img_fp.replace(suffix, prefix+'_cv.jpg'), show_flg=args.show_flg)


    

def main():
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    print_args(args)  # print args

    # step1: define the model structure
    model = getattr(mobilenet_v1, args.arch)(num_classes=args.num_classes)

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU

    # step2: optimization: loss and optimization method
    # criterion = nn.MSELoss(size_average=args.size_average).cuda()
    if args.loss.lower() == 'wpdc':
        print(args.opt_style)
        criterion = WPDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use WPDC Loss')
    elif args.loss.lower() == 'vdc':
        criterion = VDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use VDC Loss')
    elif args.loss.lower() == 'pdc':
        criterion = nn.MSELoss(size_average=args.size_average).cuda()
        logging.info('Use PDC loss')
    else:
        raise Exception(f'Unknown Loss {args.loss}')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            # checkpoint = torch.load(args.resume)['state_dict']
            model.load_state_dict(checkpoint)

        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    train_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )
    val_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_val,
        param_fp=args.param_fp_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True, drop_last=True)

    # step4: run
    cudnn.benchmark = True
    if args.test_initial:
        logging.info('Testing from initial')
        validate(val_loader, model, criterion, args.start_epoch)

    # set_trace()
    if args.gbdt==1:
        # prepare_gbdt(train_loader, model, criterion, optimizer, args)
        # generate_gbdt_dataset(train_loader, model, criterion, optimizer, args)
        train_gbdt(train_loader, model, criterion, optimizer, args)
        refine_gbdt(train_loader, model, criterion, optimizer, args)
        for epoch in range(args.start_epoch, args.epochs + 1):
            # adjust learning rate
            adjust_learning_rate(optimizer, epoch, args.milestones)

            # train for one epoch
        
        base_alpha = 0.1
        # validate_gbdt(val_loader, model, criterion, args)
        plot_result = 0
        if plot_result:
            plot_dataset = DDFAPlotDataset(
                root=args.root,
                filelists=args.filelists_val,
                param_fp=args.param_fp_val, 
                transform= transforms.Compose([ToTensorGjz()])
            )

            plot_loader = DataLoader(plot_dataset, batch_size=1, num_workers=args.workers,
                                shuffle=True, pin_memory=True, drop_last=True)
            args.transform = normalize
            # for use_gbdt in [1,0]:
            #     for use_attack in [1,0]:
            use_gbdt = 1
            use_attack = 1
            plot_gbdt(plot_loader, model, criterion, args, use_gbdt, use_attack)
            print("with original model:")
            # validate(val_loader, model, criterion, epoch, args)
            # for epoch in range(0,4):
            #     args.alpha += base_alpha
            #     print("with gbdt model of alpha:"+str(args.alpha))
            #     validate_gbdt(val_loader, model, criterion, args)


    else:
        for epoch in range(args.start_epoch, args.epochs + 1):
            # adjust learning rate
            adjust_learning_rate(optimizer, epoch, args.milestones)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)
            filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    # 'optimizer': optimizer.state_dict()
                },
                filename
            )

            validate(val_loader, model, criterion, epoch)


if __name__ == '__main__':
    main()
