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
import torch.nn.functional as F
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
# from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
from utils.params import make_abs_path
import pickle as pkl
import sys
# print(sys.path.append('D:/code/forest/scikit-learn'))
# import sys
# print(sys.path.append('D:/code/face/3DDFA/TDDFA/'))
from gradient_boosting import HistGradientBoostingRegressor
from pdb import *
# from TDDFA.TDDFA import TDDFA
# from utils.render import render
# from utils.render_ctypes import render  # faster
# global args (configuration)
from bfm import BFMModel
# load BFM
# bfm = BFMModel(
#     bfm_fp='./utils/configs/bfm_noneck_v3.pkl',
#     shape_dim=40,
#     exp_dim=10
# )
# tri = bfm.tri
args = None
lr = None
arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']

loss_prefix =''# '_vdc'
feat_index_filename = 'important_feature' + loss_prefix
gbdt_param_filename = './gbdt_param' + loss_prefix +'/gbdt_param'
gbdt_feature_prefix ='./gbdt_feature' + loss_prefix+'/gbdt_feature'
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



class ConvSplitTree2(nn.Module):
    def __init__(self, tree_depth, in_channels, out_channels= 2, kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels =1, n_split=2, resize_small=1, is_regression=1):
        super(ConvSplitTree2, self).__init__()
        if tree_depth>6:
            tree_depth = 6
        self.tree_depth = tree_depth 
        self.is_regression = is_regression
        if self.is_regression ==1:
            self.n_split = n_split
        else:
            self.n_split = 1
        self.tree_nodes = (self.tree_depth * self.n_split)
        self.sum_out_channels = int(self.tree_nodes * out_channels)
        if self.is_regression ==1:
            self.convSplit = nn.Conv2d(guide_in_channels, self.sum_out_channels, kernel_size=1, stride=stride, padding=0, bias=False).type(torch.FloatTensor)
            nn.init.uniform_(self.convSplit.weight)
            self.convSplit.weight.data *=0.01
        self.out_channels = out_channels
        self.convPred = nn.Conv2d(in_channels, self.sum_out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=True).type(torch.FloatTensor)
        nn.init.uniform_(self.convPred.weight)
        self.convPred.weight.data *=0.1 
        self.normalize_weight_iter = False
        self.kernel_size = kernel_size
        self.resize_small = resize_small
        self.softmax_weight = 0.2


    def set_eval(self, eval_= True):
        if eval_:
            self.softmax_weight = 5.8
        else:
            self.softmax_weight = 0.2
    def forward(self,x, data):
        if x.shape[2]< data.shape[2]:
            if self.resize_small ==1:
                data = F.interpolate(data, x.size()[2:], mode='bilinear', align_corners=True)
            else:
                x = F.interpolate(x, data.size()[2:], mode='bilinear', align_corners=True)

        if x.shape[2]> data.shape[2]:
            if self.resize_small ==1:
                x = F.interpolate(x, data.size()[2:], mode='bilinear', align_corners=True)
            else:
                data = F.interpolate(data, x.size()[2:], mode='bilinear', align_corners=True)
        if self.is_regression:
            return self.forward_regression(x,data)
        else:
            return self.forward_classification(x,data)
    def forward_regression(self,x, data): # x is features
        # set_trace()
        score =self.convSplit(x).view(x.shape[0],self.tree_depth,self.n_split, self.out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data)
        w,h = data.shape[2],data.shape[3]
        data=data.view(x.shape[0],self.tree_depth,self.n_split,self.out_channels,w,h)
        score = score * self.softmax_weight
        score = F.softmax(score, dim=2)
        # set_trace()
        data = torch.sum(torch.sum(score * data, dim=2),dim=1)
        final_score,_ = torch.max(score, dim=2)
        final_score = torch.prod(final_score, dim=1)
        y= final_score * data
        return y

    def forward_classification(self,x, data): # x is probability vector for each class, channels of x should equal to output channels
        score = x.view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data).view(x.shape[0],self.tree_depth,self.out_channels,x.shape[2],x.shape[3])
        score = F.softmax(score, dim=2)
        score = score * self.softmax_weight
        data = torch.sum(score * data, dim=1)
        y = data
        # final_score,_ = torch.sum(score, dim=1)
        # # final_score = torch.sum(final_score, dim=1) 
        # y= final_score * data
        return y

class ConvSplitBT(nn.Module):
    def __init__(self, tree_depth, in_channels, out_channels= 2, kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels =1, n_split=2, resize_small=1, is_regression=1):
        super(ConvSplitBT, self).__init__()
        mid_channels = 64
        #input should be of 4x4 in resolution
        self.dim_reduct = nn.Conv2d(guide_in_channels, mid_channels, kernel_size=1)
        self.tree1 = ConvSplitTree2(tree_depth, in_channels, out_channels= mid_channels, kernel_size=kernel_size, stride=1, pad=1, dilation=1, guide_in_channels =mid_channels, n_split=2, resize_small=1, is_regression=1)
        self.tree2 = ConvSplitTree2(tree_depth, mid_channels, out_channels= out_channels, kernel_size=1, stride=1, pad=0, dilation=1, guide_in_channels =mid_channels, n_split=2, resize_small=1, is_regression=1)
        self.relu0 = nn.PReLU()
        self.pool1 = nn.AvgPool2d(2,stride=2)
        self.pool2 = nn.AvgPool2d(2,stride=2)
    def forward(self,x,data):
        x = self.dim_reduct(x)
        x=self.relu0(x)
        data=self.tree1(x,data)
        data=self.pool1(data)
        data=self.tree2(x,data)
        data=self.pool2(data)
        data= torch.mean(torch.mean(data,dim=3),dim=2)
        return data

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

def train_cst(train_loader, model, criterion, optimizer, args):

    end = time.time()
    # build a batch of raw/mid features for GBDT training
    batch_size = args.batch_size
    
    rawfeature_dim = 0
    midfeature_dim = 0
    mid_features = 0 
    target_dim = 0
    model.eval()
    model.module.SetMidfeatureNeedGrad(True)
    model.module.SetFeatureLayers(args.feature_layers)
    for i, (input, target) in enumerate(train_loader):
        target_dim = target.shape[1]
        output = model(input)
        rawfeatures = model.module.low_features
        rawfeature_dim =  rawfeatures.shape[1] * rawfeatures.shape[2] *rawfeatures.shape[3] 

        mid_features = model.module.mid_features
        # midfeature_dim = mid_features.shape[1]* mid_features.shape[2] * mid_features.shape[3] 
        del model.module.mid_features
        break
    torch.cuda.empty_cache()
    tree_depth = 3
    in_channels = mid_features.shape[1]
    gc=rawfeatures.shape[1]
    cst = ConvSplitBT(tree_depth, in_channels, out_channels= 62, kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels = gc, n_split=2, resize_small=1, is_regression=1)
    cst = cst.cuda()
    cst_optimizer = torch.optim.Adadelta(cst.parameters(), lr=1.0)

    # batched_mid_features= np.zeros([batch_size, feature_dim])
    
    # try to accumulate the gradient abs, to select stable features of small gradients
    model = model.cuda()
    model.module.SetMidfeatureNeedGrad(False)
    model.module.SetFeatureLayers(args.feature_layers)
    end= time.time()
    filename = 'cst_checkpoint.t'
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        # input.requires_grad = True
        # input.retain_grad()
        target = target.cuda(non_blocking=True)
        output = model(input)
        # set_trace()
        cst_optimizer.zero_grad()
        mid_features=model.module.mid_features.clone()
        low_features=model.module.low_features.clone()
        low_features.requires_grad = True
        mid_features.requires_grad = True
        pred_cst = cst(low_features, mid_features)
        # set_trace()
        loss = criterion(pred_cst, target)
        loss.backward()
        cst_optimizer.step()
        # grad = input.grad.cpu().detach().numpy().reshape(batch_size,-1)
        # grad = model.module.low_features.grad.cpu().detach().numpy().reshape(batch_size,-1)
        # batched_mid_features[:,:rawfeature_dim] += np.abs(grad).reshape(batch_size,-1)
        # grad2 = model.module.mid_features.grad.cpu().detach().numpy()
        # batched_mid_features[:,rawfeature_dim:] += np.abs(grad2).reshape(batch_size,-1)
        del input
        # if i>500:break
        if i % args.print_freq==0:
            print("train cst for:"+str(i))
            elapse = time.time() - end
            logging.info(f' Time {elapse/60:.3f} Val: [{i} /{len(train_loader)}] Loss {loss.item():.3f}\t')
            end=time.time()
        if i % (args.print_freq*10)==0:
            pkl.dump(cst,open('cst_model.pkl','wb'))
            save_checkpoint(
                            {
                                # 'epoch': epoch,
                                'state_dict': cst.state_dict(),
                                # 'optimizer': optimizer.state_dict()
                            },
                            filename
                        )

    del model.module.mid_features
    torch.cuda.empty_cache()
    
    # set_trace()
    feature_importance = batched_mid_features.mean(axis=0)
    num_feat = int(min(rawfeature_dim/8, midfeature_dim/4))
    num_feat = int(min(min(rawfeature_dim, midfeature_dim), 4096))
    select_small = 1
    if select_small==1:
        important_rawfeature = np.argpartition(feature_importance[:rawfeature_dim], num_feat)[: num_feat]
        important_midfeature = np.argpartition(feature_importance[rawfeature_dim:], num_feat)[: num_feat]
    else:
        important_rawfeature = np.argpartition(feature_importance[:rawfeature_dim], -num_feat)[-num_feat:]
        important_midfeature = np.argpartition(feature_importance[rawfeature_dim:], -num_feat)[-num_feat:]
    # important_midfeature = important_midfeature + rawfeature_dim
    # important_rawfeature = np.argpartition(feature_importance, - num_feat)[- num_feat:]
    # important_feature= [important_rawfeature, important_midfeature]
    logging.info("select raw and mid feature:"+str(num_feat))
    logging.info("original feature gradient L1 norm:"+str(feature_importance.mean()))
    logging.info("select raw feature gradient L1 norm:" +str(feature_importance[:rawfeature_dim][important_rawfeature].mean()))
    logging.info("select mid feature gradient L1 norm:" +str(feature_importance[rawfeature_dim:][important_midfeature].mean()))
    feat_index_filename_spec = feat_index_filename+ args.layer_spec_suffix+'.pkl'
    pkl.dump([important_rawfeature, important_midfeature, num_feat, target_dim], open(feat_index_filename_spec,'wb'))



max_batches_for_eval = 400

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

def validate_cst(val_loader, model, criterion, args):
    
    alpha = args.alpha
    cst = pkl.load(open('cst_model.pkl','rb'))
    cst.tree1.set_eval()
    cst.tree2.set_eval()
    # [important_rawfeature, important_midfeature, num_feat, target_dim] = pkl.load(open(feat_index_filename,'rb'))
    # lightgbms = []
    # for i in range(target_dim):
    #     lightgbm = pkl.load(open(gbdt_param_filename + str(i) + '.pkl','rb'))
    #     lightgbms.append(lightgbm)
    batch_size = args.val_batch_size
    
    num_batch_for_eval = 100
    
    model.module.SetMidfeatureNeedGrad(False)
    model.module.SetFeatureLayers(args.feature_layers)
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
        if input.type()== 'torch.DoubleTensor':
            input = input.type('torch.FloatTensor')
        if input.type()== 'torch.cuda.DoubleTensor':
            input = input.type('torch.cuda.FloatTensor')
        target = target.type('torch.FloatTensor')
        # compute output
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        if use_attack==1:input.requires_grad = True
        output = model(input)

        gbdt_output = cst(model.module.low_features, model.module.mid_features)
        # gbdt_output = output * (1- alpha) + gbdt_output* alpha
        # gbdt_output[12:] = output[12:]#* (1- alpha) + gbdt_output[12:] * alpha
        #GBDT seems to have better result in predicting R,t, than predicting component coefficients
        
        loss = criterion(gbdt_output, target)
        losses_gbdt_original.append(loss.item())
        loss = criterion(output, target)
        losses_cnn_original.append(loss.item())

        if use_attack==1:
            input_original = input.clone().detach()
            attack_maxstepsize = args.attacksize #np.abs(input_original.cpu().numpy()).mean()*0.02
            input_upper_limit= input_original + attack_maxstepsize
            input_lower_limit = input_original - attack_maxstepsize
            steps = max(int(attack_maxstepsize/0.001),5)
            attack_stepsize = attack_maxstepsize/steps
            # attack_stepsize = 100000.0
            input.retain_grad()
            for attack_iter in range(steps):
                loss = criterion(output, target)
                loss.backward()
                # input.abs().mean()/(input.grad.abs().mean())
                input.detach()
                input.data = input.data + input.grad.sign() * attack_stepsize
                input.data[input.data>input_upper_limit] = input_upper_limit.data[input.data>input_upper_limit]
                input.data[input.data<input_lower_limit] = input_lower_limit.data[input.data<input_lower_limit]
                input.requires_grad= True
                input.retain_grad()
                output = model(input)    
            gbdt_output = cst(model.module.low_features, model.module.mid_features)
            #* (1- alpha) + gbdt_output * alpha    
            # gbdt_output[12:] = output[12:]#* (1- alpha) + gbdt_output[12:] * alpha
            loss = criterion(output, target)
            losses_cnn_attacked.append(loss.item())
            loss = criterion(gbdt_output, target)
            losses_gbdt_attacked.append(loss.item())

        if i> max_batches_for_eval: break
        if i % args.print_freq==0:
            print("validate gbdt for:"+str(i))
            elapse = time.time() - end
            # loss = np.mean(losses)
            logging.info(args.layer_spec_suffix)
            logging.info(f'alpha[{alpha}] Time {elapse/60:.3f} Val: [{i} /{len(val_loader)}]\t'
                         f'CNN Loss original {np.mean(losses_cnn_original)*100:.4f}\t'
                         f'CNN Loss attacked {np.mean(losses_cnn_attacked)*100:.4f}\t')
            logging.info(f'GBDT Loss original {np.mean(losses_gbdt_original)*100:.4f}\t'
                         f'GBDT Loss attacked {np.mean(losses_gbdt_attacked)*100:.4f}\t')
    elapse = time.time() - end
    # loss = np.mean(losses)
    logging.info(args.layer_spec_suffix)
    logging.info(f'alpha[{alpha}] Time {elapse/60:.3f} Val: [{i} /{len(val_loader)}]\t'
                 f'CNN Loss original {np.mean(losses_cnn_original)*100:.4f}\t'
                 f'CNN Loss attacked {np.mean(losses_cnn_attacked)*100:.4f}\t')
    logging.info(f'GBDT Loss original {np.mean(losses_gbdt_original)*100:.4f}\t'
                 f'GBDT Loss attacked {np.mean(losses_gbdt_attacked)*100:.4f}\t')
    return [np.mean(losses_cnn_original)*100, np.mean(losses_cnn_attacked)*100, np.mean(losses_gbdt_original)*100, np.mean(losses_gbdt_attacked)*100]


def validate_gbdt(val_loader, model, criterion, args):
    
    alpha = args.alpha
    gbdt = GBDT_Predictor(feat_index_filename+args.layer_spec_suffix+'.pkl', gbdt_param_filename)
    # [important_rawfeature, important_midfeature, num_feat, target_dim] = pkl.load(open(feat_index_filename,'rb'))
    # lightgbms = []
    # for i in range(target_dim):
    #     lightgbm = pkl.load(open(gbdt_param_filename + str(i) + '.pkl','rb'))
    #     lightgbms.append(lightgbm)
    batch_size = args.val_batch_size
    
    num_batch_for_eval = 100
    
    model.module.SetMidfeatureNeedGrad(False)
    model.module.SetFeatureLayers(args.feature_layers)
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
        if input.type()== 'torch.DoubleTensor':
            input = input.type('torch.FloatTensor')
        if input.type()== 'torch.cuda.DoubleTensor':
            input = input.type('torch.cuda.FloatTensor')
        target = target.type('torch.FloatTensor')
        # compute output
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        if use_attack==1:input.requires_grad = True
        output = model(input)

        gbdt_output = gbdt.predict(model.module.low_features, model.module.mid_features)
        # gbdt_output = output * (1- alpha) + gbdt_output* alpha
        # gbdt_output[12:] = output[12:]#* (1- alpha) + gbdt_output[12:] * alpha
        #GBDT seems to have better result in predicting R,t, than predicting component coefficients
        
        loss = criterion(gbdt_output, target)
        losses_gbdt_original.append(loss.item())
        loss = criterion(output, target)
        losses_cnn_original.append(loss.item())

        if use_attack==1:
            input_original = input.clone().detach()
            attack_maxstepsize = args.attacksize #np.abs(input_original.cpu().numpy()).mean()*0.02
            input_upper_limit= input_original + attack_maxstepsize
            input_lower_limit = input_original - attack_maxstepsize
            steps = max(int(attack_maxstepsize/0.001),5)
            attack_stepsize = attack_maxstepsize/steps
            # attack_stepsize = 100000.0
            input.retain_grad()
            for attack_iter in range(steps):
                loss = criterion(output, target)
                loss.backward()
                # input.abs().mean()/(input.grad.abs().mean())
                input.detach()
                input.data = input.data + input.grad.sign() * attack_stepsize
                input.data[input.data>input_upper_limit] = input_upper_limit.data[input.data>input_upper_limit]
                input.data[input.data<input_lower_limit] = input_lower_limit.data[input.data<input_lower_limit]
                input.requires_grad= True
                input.retain_grad()
                output = model(input)    
            gbdt_output = gbdt.predict(model.module.low_features, model.module.mid_features)
            #* (1- alpha) + gbdt_output * alpha    
            # gbdt_output[12:] = output[12:]#* (1- alpha) + gbdt_output[12:] * alpha
            loss = criterion(output, target)
            losses_cnn_attacked.append(loss.item())
            loss = criterion(gbdt_output, target)
            losses_gbdt_attacked.append(loss.item())

        if i> max_batches_for_eval: break
        if i % args.print_freq==0:
            print("validate gbdt for:"+str(i))
            elapse = time.time() - end
            # loss = np.mean(losses)
            logging.info(args.layer_spec_suffix)
            logging.info(f'alpha[{alpha}] Time {elapse/60:.3f} Val: [{i} /{len(val_loader)}]\t'
                         f'CNN Loss original {np.mean(losses_cnn_original)*100:.4f}\t'
                         f'CNN Loss attacked {np.mean(losses_cnn_attacked)*100:.4f}\t')
            logging.info(f'GBDT Loss original {np.mean(losses_gbdt_original)*100:.4f}\t'
                         f'GBDT Loss attacked {np.mean(losses_gbdt_attacked)*100:.4f}\t')
    elapse = time.time() - end
    # loss = np.mean(losses)
    logging.info(args.layer_spec_suffix)
    logging.info(f'alpha[{alpha}] Time {elapse/60:.3f} Val: [{i} /{len(val_loader)}]\t'
                 f'CNN Loss original {np.mean(losses_cnn_original)*100:.4f}\t'
                 f'CNN Loss attacked {np.mean(losses_cnn_attacked)*100:.4f}\t')
    logging.info(f'GBDT Loss original {np.mean(losses_gbdt_original)*100:.4f}\t'
                 f'GBDT Loss attacked {np.mean(losses_gbdt_attacked)*100:.4f}\t')
    return [np.mean(losses_cnn_original)*100, np.mean(losses_cnn_attacked)*100, np.mean(losses_gbdt_original)*100, np.mean(losses_gbdt_attacked)*100]

from copy import deepcopy

def plot_gbdt(val_loader, model, criterion, args, use_gbdt, use_attack):
    
    alpha = args.alpha
    gbdt = GBDT_Predictor(feat_index_filename + args.layer_spec_suffix + '.pkl', gbdt_param_filename)

    model.module.SetMidfeatureNeedGrad(False)
    model.module.SetFeatureLayers(args.feature_layers)
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
    loss_for_dataset = dict()
    # tddfa = TDDFA()
    for i, (img, target, img_cv) in enumerate(val_loader):
        # set_trace()
        if i>1500 or plot_num>20:break
        
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
        pdc_lossdict = [0,0,0,0]
        prefixdict = [0,0,0,0]
        imgdict = [0,0,0,0]
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        
        if use_attack==1:input.requires_grad = True
        output = model(input)
        loss = criterion(output, target)
        # loss_pdc = criterion_pdc(output, target)

        testid = 0
        outputdict[testid] = target.clone().squeeze().detach().cpu().numpy() 
        #output.clone().squeeze().detach().cpu().numpy()
        lossdict[testid] = loss.item()
        # pdc_lossdict[testid] = loss_pdc.item()
        prefixdict[testid] = 'orig_gt'

        if use_gbdt==1: 
            testid+=1
            output_gbdt = gbdt.predict(model.module.low_features, model.module.mid_features).type(torch.cuda.FloatTensor)
            # set_trace()
            loss_gbdt = criterion(output_gbdt, target)
            outputdict[testid] = output_gbdt.clone().squeeze().detach().cpu().numpy()
            lossdict[testid] += loss_gbdt.item()
            prefixdict[testid] = 'orig_gbdt'

        if use_attack==1:
            input_original = input.clone().detach()
            attack_maxstepsize = 0.01  # 0.01#np.abs(input_original.cpu().numpy()).mean()*0.02
            input_upper_limit= input_original + attack_maxstepsize
            input_lower_limit = input_original - attack_maxstepsize
            steps = max(int(attack_maxstepsize/0.002),5)
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
            outputdict[testid] = output.clone().squeeze().detach().cpu().numpy()
            lossdict[testid] += loss.item()
            prefixdict[testid] = 'attack_cnn'

            if use_gbdt==1: 
                output_gbdt = gbdt.predict(model.module.low_features, model.module.mid_features).type(torch.cuda.FloatTensor)
                loss_gbdt = criterion(output_gbdt, target)
                testid +=1
                outputdict[testid] = output_gbdt.clone().squeeze().detach().cpu().numpy()
                lossdict[testid] += loss_gbdt.item()
                prefixdict[testid] = 'attack_gbdt'
            # loss = criterion(gbdt_output, target)
            # losses_gbdt_attacked.append(loss.item())

        print(prefixdict)
        print(lossdict)
        plot_this = False

        # if lossdict[1]< (lossdict[0]) and (lossdict[3]< (lossdict[2])):
        if lossdict[3]< (lossdict[2] * 0.5):
            if lossdict[1]>10: #vdc loss
                if lossdict[3]<30:
                    # if lossdict[2] - lossdict[3]>0.02:
                    plot_this = True
                    plot_num +=1
            else:
                if lossdict[3]<0.03: #wpdc loss
                    # if lossdict[2] - lossdict[3]>0.02:
                    plot_this = True
                    plot_num +=1

        # plot_this = True

        h,w,nc = img.shape[2], img.shape[3], img.shape[1]
        img_ori = img.reshape(nc,h,w).permute(1,2,0).cpu().numpy() # 3 x H x W
        img_ori = img_ori.astype(int) 
        if use_attack ==1 :
            img_attack = args.transform.reverse(input.detach())
            img_attack = img_attack.reshape(nc,h,w).permute(1,2,0).cpu().numpy()#.permute(2,0,1)
            img_attack=img_attack.astype(int)
            img_attack[img_attack>255] =255
            img_attack[img_attack<0] = 0
        imgdict[0] = img_ori
        imgdict[1] = img_ori
        imgdict[2] = img_attack
        imgdict[3] = img_attack
        if not plot_this:print("skip this pic")
        # set_trace()
        if plot_this:
            img_fp = "plot/save"+str(plot_num)+'.jpg'
            suffix = get_suffix(img_fp)
            difference = np.abs(img_attack - img_ori).mean()
            print("attack make difference of:"+str( difference)+" pixles")
            pkl.dump([lossdict, difference],open('plot/loss'+str(plot_num)+'.pkl','wb'))
            # loss_for_dataset[plot_num] = deepcopy(lossdict)
            for j in range(testid+1):
                param = outputdict[j]
                prefix = prefixdict[j]
                # param = param.squeeze().detach().cpu().numpy().flatten().astype(np.float32)
                param = param.flatten().astype(np.float32)
                img_ori = imgdict[j]
                # 68 pts
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

                ind= plot_num
                # dense face 3d vertices
                if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                    vertices = predict_dense(param, roi_box)
                    vertices_lst.append(vertices)
                if args.dump_ply:
                    fname = '{}_{}_{}.ply'.format(img_fp.replace(suffix, ''), ind, prefix)
                    dump_to_ply(vertices, tri, fname)
                # if args.dump_vertex:
                #     dump_vertex(vertices, '{}_{}_{}.mat'.format(img_fp.replace(suffix, ''), ind, prefix))

                # param_lst, roi_box_lst = tddfa(img, boxes)

                # ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
                # set_trace()
                wfp_3d = '{}_{}_{}_3d.jpg'.format(img_fp.replace(suffix, ''), ind, prefix)
                render(img_ori.astype(np.uint8), vertices_lst, bfm.tri, alpha=0.2, show_flag=False, wfp=wfp_3d)
                if args.dump_pts:
                    wfp = '{}_{}_{}.txt'.format(img_fp.replace(suffix, ''), ind, prefix)
                    np.savetxt(wfp, pts68, fmt='%.3f')
                    print('Save 68 3d landmarks to {}'.format(wfp))
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
                if args.dump_obj:
                    wfp = '{}_{}_{}.obj'.format(img_fp.replace(suffix, ''), ind, prefix)
                    colors = get_colors(img_ori, vertices)
                    write_obj_with_colors(wfp, vertices, tri, colors)
                    print('Dump obj with sampled texture to {}'.format(wfp))
                # ind += 1

                if args.dump_pose:
                    # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
                    img_pose = plot_pose_box(img_ori, Ps, pts_res)
                    wfp = img_fp.replace(suffix, '_pose') + prefix+ '.jpg'
                    cv2.imwrite(wfp, img_pose)
                    print('Dump to {}'.format(wfp))
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
                
                if args.dump_res:
                    note=prefix+" loss:{:.2f}".format(lossdict[j])
                    # set_trace()
                    draw_landmarks(img_ori.copy(), pts_res, wfp=img_fp.replace(suffix, prefix+'.jpg'), show_flg=args.show_flg)
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
            # model.load_state_dict(checkpoint)


            # checkpoint_fp = 'weights/mb_1.p'
            # arch = 'mobilenet_1'
            # checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
            # model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

            model_dict = model.state_dict()
            # because the model is trained by multiple gpus, prefix module should be removed
            for kc in checkpoint.keys():
                # kc = k.replace('module.', '')
                if kc in model_dict.keys():
                    model_dict[kc] = checkpoint[kc]
                if kc in ['fc_param.bias', 'fc_param.weight']:
                    model_dict[kc.replace('_param', '')] = checkpoint[kc]
            model.load_state_dict(model_dict)

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

    args.latex_table = 0
    if args.latex_table==1:
        result_list = []
        config_list = []
        # feature_layers = [ [0,12],[4,12], [8,15],[12,15],[0,15], [6,12]]
        # # feature_layers = [ [0,12]]
        # feature_layers = [ [8,15], [6,12], ]
        # attacksize = 0.01
        # loss_metric = VDCLoss(opt_style=args.opt_style).cuda()
        # for config_iter in range(len(feature_layers)):
        #     feature_layer = feature_layers[config_iter]
        #     for loss_metric in [ VDCLoss(opt_style=args.opt_style).cuda(), WPDCLoss(opt_style=args.opt_style).cuda(),  nn.MSELoss(size_average=args.size_average).cuda()]:
        #         for attacksize in [0.005, 0.01, 0.02]:
        #             args.attacksize = attacksize
        #             args.feature_layers = feature_layer
        #             args.layer_spec_suffix ='_layer'+ str(args.feature_layers[0])+'_'+str(args.feature_layers[1])           
        #             logging.info("start validating layer "+args.layer_spec_suffix)
        #             result = validate_gbdt(val_loader, model, loss_metric, args)
        #             logging.info("end validating layer "+args.layer_spec_suffix)
        #             result_list.append(result)
        #     # break
        #     pkl.dump([result_list,config_list],open(loss_prefix+'result_list.pkl','wb'))
        # [result_list,config_list] = pkl.load(open(loss_prefix+'result_list.pkl','rb'))
        result_latex = open(loss_prefix+'cst_result_latex.txt','w')
        lineid= 0
        for result in result_list:
            if lineid % 3==0:
                result_latex.write("\\\\ \\hline")
                result_latex.write("\n")
            result_latex.write("{:.2f}".format(float(result[0]))+"& ")
            result_latex.write("{:.2f}".format(float(result[1]))+"& ") 
            result_latex.write("{:.2f}".format(float(result[3]))+"& ") 
            lineid+=1
        result_latex.close()
        # return


    if args.gbdt==1:
        # for feature_layers in [ [8,15], [12,15], [0,12]]:
        args.attacksize = 0.01
        # try:
            
        feature_layers = [8,15]
        args.feature_layers = feature_layers

        args.layer_spec_suffix ='_layer'+ str(args.feature_layers[0])+'_'+str(args.feature_layers[1])
        logging.info("start generating feature from layer "+args.layer_spec_suffix)
        # train_cst(train_loader, model, criterion, optimizer, args)
        validate_cst(val_loader, model, criterion, args)
        # # generate_gbdt_dataset(train_loader, model, criterion, optimizer, args)
        # logging.info("start training GBDT on layer "+ args.layer_spec_suffix)
        # train_gbdt(train_loader, model, criterion, optimizer, args)
        # # refine_gbdt(train_loader, model, criterion, optimizer, args)
        # logging.info("start validating layer "+args.layer_spec_suffix)
        # validate_gbdt(val_loader, model, criterion, args)
        # logging.info("end validating layer "+args.layer_spec_suffix)
        # # except:
        # #     logging.info("some error happened")

        # feature_layers = [6,12]
        # args.feature_layers = feature_layers

        plot_result = 1
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
            feature_layers = [6,12]
            args.feature_layers = feature_layers
            args.layer_spec_suffix ='_layer'+ str(args.feature_layers[0])+'_'+str(args.feature_layers[1])
            # for use_gbdt in [1,0]:
            #     for use_attack in [1,0]:
            use_gbdt = 1
            use_attack = 1
            # criterions = []
            # criterions.append(WPDCLoss(opt_style=args.opt_style).cuda())
            
            # criterions.append(VDCLoss(opt_style=args.opt_style).cuda())
            # criterions.append(nn.MSELoss(size_average=args.size_average).cuda())
            # criterion = VDCLoss(opt_style=args.opt_style).cuda()
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
