#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import mobilenet_v1
import time
import numpy as np

from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw

from vdc_loss import VDCLoss
from wpdc_loss import WPDCLoss

from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex, reconstruct_vertex_tensor
import argparse
# from train import GBDT_Predictor
from gradient_boosting import HistGradientBoostingRegressor
import pickle as pkl

from pdb import set_trace
class GBDT_Predictor:
    def __init__(self, feat_index_filename_spec, gbdt_param_filename, layer_spec_suffix, fileid = 1):
        [self.rawfeature_index, self.midfeature_index, self.num_feat, self.target_dim] = pkl.load(open(feat_index_filename_spec,'rb'))
        batch_size = 32
        self.feature_dim = self.rawfeature_index.shape[0] + self.midfeature_index.shape[0]
        self.batched_gbdt_features = np.zeros([batch_size, self.feature_dim])
        self.batched_gbdt_predict = np.zeros([batch_size, self.target_dim])
        if self.num_feat != self.rawfeature_index.shape[0]:
            print('num_feat != self.rawfeature_index.shape[0] ')
        self.lightgbms = []
        
        for i in range(self.target_dim):
            # self.lightgbm = pkl.load(open(gbdt_param_filename+ '_ref' + str(i) + '.pkl','rb'))
            self.lightgbm = pkl.load(open(gbdt_param_filename+ str(i) + layer_spec_suffix +'_batch' +str(fileid) + '.pkl','rb'))
            self.lightgbms.append(self.lightgbm)

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
        return torch.tensor(self.batched_gbdt_predict).type('torch.FloatTensor').cuda()
        # return self.batched_gbdt_predict

datafolder = 'D:/data/facealignment/'
def extract_param(checkpoint_fp, root='', filelists=None, arch='mobilenet_1', num_classes=62, device_ids=[0],
                  batch_size=128, num_workers=4, args=[]):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(device_ids[0])
    model = getattr(mobilenet_v1, arch)(num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint)

    dataset = DDFATestDataset(filelists=filelists, root=root, transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
        labeled = True)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    alpha = 0.8
    def use_gbdt(args, model):
        feat_index_filename = 'important_feature'
        gbdt_param_filename = './gbdt_param/gbdt_param'
        # set_trace()
        feature_layers = [8,15] #[0,12]
        # args.feature_layers = feature_layers

        # args.
        layer_spec_suffix ='_layer'+ str(feature_layers[0])+'_'+str(feature_layers[1])
        gbdt = GBDT_Predictor(feat_index_filename + layer_spec_suffix + '.pkl', gbdt_param_filename, layer_spec_suffix)

        model.module.SetMidfeatureNeedGrad(False)
        model.module.SetFeatureLayers(feature_layers)
        return gbdt

    if args.gbdt==1:gbdt = use_gbdt(args, model)
    use_attack = 1
    # if args.loss.lower() == 'wpdc':

    criterion = WPDCLoss(opt_style='resample').cuda()
    # logging.info('Use WPDC Loss')
    # with torch.no_grad():

    std_size = 120
    for _, (input, target, roi_boxs) in enumerate(data_loader):
        inputs = input.cuda()
        if use_attack:
            input.requires_grad = True

        output = model(input)
        if use_attack==1:
            set_trace()
            pts68s = reconstruct_vertex_tensor(output)
            for i in range(roi_boxs.shape[0]):
                pts68 = pts68s[i]
                pts68_gt = target[i]

                sx, sy, ex, ey = roi_boxs[i]
                scale_x = (ex - sx) / std_size
                scale_y = (ey - sy) / std_size
                pts68[0, :] = pts68[0, :] * scale_x + sx
                pts68[1, :] = pts68[1, :] * scale_y + sy

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
                # loss = criterion(output, target)
                tmp = reconstruct_vertex(output[0].detach().cpu().numpy())
                pts68 = reconstruct_vertex_tensor(output)
                set_trace()
                diff = (pts68 - target) ** 2
                loss = torch.mean(diff)
                loss.backward()
                # input.abs().mean()/(input.grad.abs().mean())
                input.detach()
                input.data = input.data + input.grad.sign() * attack_stepsize
                input.data[input.data>input_upper_limit] = input_upper_limit.data[input.data>input_upper_limit]
                input.data[input.data<input_lower_limit] = input_lower_limit.data[input.data<input_lower_limit]
                input.requires_grad= True
                input.retain_grad()
                output = model(input)  
        output = output.detach()  
        if args.gbdt==1:
            gbdt_output = gbdt.predict(model.module.low_features, model.module.mid_features)

            output = alpha * output +(1- alpha)* gbdt_output
            # output[:,:12] = gbdt_output[:,:12]

        # set_trace()
        for i in range(output.shape[0]):
            param_prediction = output[i].cpu().numpy().flatten()

            outputs.append(param_prediction)
            # set_trace()
    outputs = np.array(outputs, dtype=np.float32)

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def _benchmark_aflw(outputs):
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


def benchmark_alfw_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_pipeline(arch, checkpoint_fp, args):
    device_ids = [0]

    def aflw():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root=datafolder+'test.data/AFLW_GT_crop',
            filelists=datafolder+'test.data/AFLW_GT_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=32, args=args)

        benchmark_alfw_params(params)

    def aflw2000():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root=datafolder+'test.data/AFLW2000-3D_crop',
            filelists=datafolder+'test.data/AFLW2000-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=32, args=args)

        benchmark_aflw2000_params(params)

    aflw2000()
    aflw()


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='mobilenet_1', type=str)
    parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str)
    parser.add_argument('--gbdt', default=1, type=int)
    args = parser.parse_args()
    args.attacksize = 0.01

    benchmark_pipeline(args.arch, args.checkpoint_fp, args)


if __name__ == '__main__':
    main()
