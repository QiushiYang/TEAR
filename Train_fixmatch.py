from __future__ import print_function
import random
import time
import argparse
import os
import sys
import ast
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import BatchNorm2d

from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader

from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler, visualizaion

import torchvision.models as models
from torchvision.datasets import ImageFolder
from models.densenet import densenet121

from models.alexnet import AlexNet
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import accuracy_score
import torch_optimizer

import tensorboard_logger
import cv2
from scipy.stats import wasserstein_distance
from torch.nn.parallel import DistributedDataParallel as DDP

log_dir = './logs'

def set_model(args):
    if args.backbone == 'WideResnet':
        model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=False)
    elif args.backbone == 'alexnet':
        model = AlexNet(batch_size=args.batchsize, n_classes=args.n_classes, std=0.15, noise=False, data=args.dataset)
    elif args.backbone == 'densenet121':
        model = densenet121(progress=False, pretrained=args.pre_trained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, args.n_classes)
    
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        save_epoch = checkpoint['epoch']
        print('loaded from checkpoint: %s'%args.checkpoint)
    else:
        save_epoch = 0
        save_optim = None

    if len(args.gpu)>1:
        torch.distributed.init_process_group(backend="nccl",init_method='tcp://localhost:1001', rank=0, world_size=1)
        model.train()
        model.cuda()
        model = DDP(model, device_ids=[0,1])
    else:
        model.train()
        model.cuda()

    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()
    
    if args.eval_ema:
        if args.backbone == 'WideResnet':
            ema_model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=False)
        elif args.backbone == 'alexnet':
            ema_model = AlexNet(batch_size=args.batchsize, n_classes=args.n_classes, std=0.15, noise=False, data=args.dataset)
        elif args.backbone == 'densenet121':
            ema_model = densenet121(progress=False, pretrained=args.pre_trained)
            num_ftrs = ema_model.classifier.in_features
            ema_model.classifier = nn.Linear(num_ftrs, args.n_classes) 
      
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()  
        ema_model.eval() 
    else:
        ema_model = None   
    
    return model, criteria_x, criteria_u, ema_model, save_epoch
    

def sigmoid(x, gamma=1, k=1):
    return 1 / (k + torch.exp(-x*gamma))

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):  # Copy BN params.
        buffer_eval.copy_(buffer_train)  

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def train_one_epoch(epoch,
                    model,
                    ema_model,
                    criteria_x,
                    criteria_u,
                    optim,
                    lr_schdlr,
                    dltrain_x,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    prob_list,
                    l_probs_list,
                    prob_list_ema,
                    example_stats,
                    ):
    
    model.train()
    
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    n_correct_u_lbs_meter = AverageMeter()
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()

    adapl_loss_meter = AverageMeter()
    n_correct_u_lbs_all_meter = AverageMeter()
    guess_recall = AverageMeter()

    ul_true_loss_meter = AverageMeter()
    ul_true_select_meter = AverageMeter()
    ul_true_unselect_meter = AverageMeter()
    
    adapl_value_select_meter = AverageMeter()
    adapl_value_unselect_meter = AverageMeter()
    adapl_value_withEntropy_meter = AverageMeter()
    adapl_value_ori_withEnt_meter = AverageMeter()
    weight_term_meter = AverageMeter()
    
    entropy_select_meter = AverageMeter()
    entropy_unselect_meter = AverageMeter()
    adapl_value_withEntropy_select_meter = AverageMeter()
    adapl_value_withEntropy_unselect_meter = AverageMeter()

    param_grad_sum = 0.0

    epoch_start = time.time()  

    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters): # Training

        ims_x_weak, lbs_x = next(dl_x)  
        
        if args.diff_aug:    
            (ims_u_weak, ims_u_strong, ims_u_strong_mlp), lbs_u_real = next(dl_u)
        else:
            (ims_u_weak, ims_u_strong), lbs_u_real = next(dl_u)
        
        lbs_x = lbs_x.cuda()
        lbs_u_real = lbs_u_real.cuda()

        bt = ims_x_weak.size(0)
        mu = int(ims_u_weak.size(0) // bt)
        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong], dim=0).cuda()

        if args.diff_aug:
            imgs_mlp = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong_mlp], dim=0).cuda()
        
        logits, feat = model(imgs,True)
        feat_x = feat[:bt]
        feat_u_w, feat_u_s = torch.split(feat[bt:], bt * mu)

        logits_x = logits[:bt]
        logits_u_w, logits_u_s = torch.split(logits[bt:], bt * mu) # only calculate the int-times loss (unlabeled:labeled)
        loss_x = criteria_x(logits_x, lbs_x.long()) # CE loss

        if args.mean_teacher:
            logits, feat = ema_model(imgs,True)
            logits_x = logits[:bt]
            logits_u_w, _ = torch.split(logits[bt:], bt * mu)
            loss_x += criteria_x(logits_x, lbs_x.long())
            loss_x /= 2.0

        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)
            
            if args.DA:
                prob_list.append(probs.mean(0))
                if len(prob_list)>32:
                    prob_list.pop(0)  
                prob_avg = torch.stack(prob_list,dim=0).mean(0)
                probs = probs / prob_avg
                if args.standard_DA:
                    l_probs = torch.softmax(logits_x, dim=1)
                    l_probs_list.append(l_probs.mean(0))
                    if len(l_probs_list)>32:
                        l_probs_list.pop(0)  
                    probs = probs * torch.stack(l_probs_list,dim=0).mean(0)
                probs = probs / probs.sum(dim=1, keepdim=True)  
                if args.standard_DA:
                    # 2. Apply sharpening.
                    probs = probs ** (1. / args.DA_T) # T=0.5
                    probs = probs / probs.sum(dim=1, keepdim=True)   

            scores, lbs_u_guess = torch.max(probs, dim=1)

            mask = scores.ge(args.thr).float()  ## scores.ge: 1 if scores>=thr; else 0;
            

        ul_true_loss = criteria_x(logits_u_w, lbs_u_real.long())
        ul_true_loss_meter.update(ul_true_loss.item())
        ul_true_select_loss = (criteria_u(logits_u_s, lbs_u_real)*mask).sum()/(mask.sum()+1e-5)
        ul_true_select_meter.update(ul_true_select_loss.item())
        ul_true_unselect_loss = (criteria_u(logits_u_s, lbs_u_real)*(1-mask)).sum()/((1-mask).sum()+1e-5)
        ul_true_unselect_meter.update(ul_true_unselect_loss.item())

        if args.dynamic_ema:
            loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()
        else:
            with torch.no_grad():
                probs = probs.detach()
            loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()  # CE loss of Unlabelled data

        if args.adapl == True:
            _ema_scores, _ema_feat = ema_model(imgs, True)
            ema_x_scores = _ema_scores[:bt]
            ema_u_scores_weak, ema_u_scores_strong = torch.split(_ema_scores[bt:], bt * mu)
            ema_x_feat = _ema_feat[:bt]
            ema_u_feat_weak, ema_u_feat_strong = torch.split(_ema_feat[bt:], bt * mu)

            if args.adapl_use_weak:                    
                ema_u_scores = ema_u_scores_weak   ###
                ema_u_feat = ema_u_feat_weak   ###
            if args.adapl_use_strong:
                ema_u_scores = ema_u_scores_strong   ###
                ema_u_feat = ema_u_feat_strong   ###
            
            if args.distribution_adapl:
                _dist_mask = scores.ge(args.thr_dist).float()

                lbs_x_guess = torch.softmax(logits_x, dim=1)
                _scores_x, lbs_x_guess = torch.max(lbs_x_guess, dim=1)
                mask_x = _scores_x.ge(args.thr_dist).float()

                label_joint = torch.cat([lbs_x_guess, lbs_u_guess],0) # L+UL label/pred
                feat_joint = torch.cat([feat_x, feat_u_w],0)
                ema_feat_joint = torch.cat([ema_x_feat, ema_u_feat_weak],0)

                clf = KMeans(n_clusters=args.n_classes)
                clf.fit(feat_u_w.detach().cpu().numpy())
                centers = clf.cluster_centers_
                kmeans_labels = clf.labels_
                
                probs_x_ema = torch.softmax(ema_x_scores, dim=1)
                _scores_x_ema, lbs_x_guess_ema = torch.max(probs_x_ema, dim=1)
                mask_eam_x = _scores_x_ema.ge(args.thr_dist).float()

                probs_u_ema = torch.softmax(ema_u_scores, dim=1)
                if args.DA:
                    prob_list_ema.append(probs_u_ema.mean(0))
                    if len(prob_list_ema)>32:
                        prob_list_ema.pop(0)   ## maintain a 32-length squeue prob_list to calculate distributions
                    prob_avg = torch.stack(prob_list_ema,dim=0).mean(0)
                    probs_u_ema = probs_u_ema / prob_avg
                    probs_u_ema = probs_u_ema / probs_u_ema.sum(dim=1, keepdim=True)   ## Normalize the prob_list_ema                
                _scores_ema, lbs_u_guess_ema = torch.max(probs_u_ema, dim=1)
                mask_ema = _scores_ema.ge(args.thr_dist).float()  ## scores.ge: 1 if scores>=thr; else 0;
                
                mask_dist = torch.cat([mask_x, _dist_mask],0)  # mask
                mask_ema_dist = torch.cat([mask_eam_x, mask_ema],0)

                lbs_joint_guess_ema = torch.cat([lbs_x_guess_ema, lbs_u_guess_ema],0)

                prototype_u_w = torch.zeros([args.n_classes, feat_x.shape[-1]])
                std_u_k = torch.zeros([args.n_classes, feat_x.shape[-1]])
                prototype_u_w_ema = torch.zeros([args.n_classes, feat_x.shape[-1]])
                std_u_k_ema = torch.zeros([args.n_classes, feat_x.shape[-1]])

                for k in range(args.n_classes):
                    index = torch.where([torch.tensor(kmeans_labels)==k][0]==True)[0].tolist()
                    
                    if index != []:
                        proto_k = torch.mean(feat_joint[index,...],0)
                        prototype_u_w[k] = proto_k

                        std_k = torch.mul(torch.transpose((feat_joint[index,...]-proto_k)**2,0,1), mask_dist[index]).sum(1) / (mask_dist[index].sum()+1e-4)
                        std_k = torch.sqrt(std_k+1e-4)
                        std_u_k[k] = std_k
                    
                    index_ema = torch.where([lbs_joint_guess_ema==k][0]==True)[0].tolist()
                    if index_ema != []:
                        proto_k_ema = torch.mean(ema_feat_joint[index_ema,...],0)
                        prototype_u_w_ema[k] = proto_k_ema
                        
                        std_k_ema = torch.mul(torch.transpose((ema_feat_joint[index_ema,...]-proto_k_ema)**2,0,1), mask_ema_dist[index_ema]).sum(1) / (mask_ema_dist[index_ema].sum()+1e-4)
                        std_k_ema = torch.sqrt(std_k_ema+1e-4)
                        std_u_k_ema[k] = std_k_ema
                                    
                cov_u = 1 / (prototype_u_w.shape[0]-1) * (prototype_u_w-prototype_u_w.mean(1).unsqueeze(-1)) @ (prototype_u_w-prototype_u_w.mean(1).unsqueeze(-1)).transpose(-1,-2)
                cov_u_ema = 1 / (prototype_u_w_ema.shape[0]-1) * (prototype_u_w_ema-prototype_u_w_ema.mean(1).unsqueeze(-1)) @ (prototype_u_w_ema-prototype_u_w_ema.mean(1).unsqueeze(-1)).transpose(-1,-2)

                loss_dist_proto = torch.tensor([0.0]).cuda()
                loss_dist_std = torch.tensor([0.0]).cuda()
                num_c = 0
                _num_c = 0
                for k in range(args.n_classes):
                    if prototype_u_w[k,...].mean() != 0 and prototype_u_w_ema.mean() != 0:
                        loss_dist_proto += 1-(prototype_u_w[k,...]*prototype_u_w_ema[k,...]).sum()/(torch.norm(prototype_u_w[k,...])*torch.norm(prototype_u_w_ema[k,...])+1e-4) # Cosine Similarity
                        diff_std = ((std_u_k[k,...] - std_u_k_ema[k,...])**2).mean()
                        if std_u_k[k,...].mean()>2*std_u_k_ema[k,...].mean() or std_u_k_ema[k,...].mean()>2*std_u_k[k,...].mean():
                            loss_dist_std += 0.0
                        else:
                            loss_dist_std += diff_std
                            _num_c += 1
                        cov_u[k,k] = 1.0
                        cov_u_ema[k,k] = 1.0
                    else:
                        cov_u[k,:] = 0.0
                        cov_u[:,k] = 0.0
                        cov_u[k,k] = 1.0
                        cov_u_ema[k,:] = 0.0
                        cov_u_ema[:,k] = 0.0
                        cov_u_ema[k,k] = 1.0
                    num_c += 1
                
                if _num_c != 0:
                    loss_dist_proto /= _num_c #num_c #args.n_classes
                    loss_dist_std /= _num_c #num_c #args.n_classes
                else:
                    loss_dist_proto = torch.tensor([0.0]).cuda()

                loss_dist_cov = ((cov_u - torch.eye(args.n_classes))**2).sum() / args.n_classes
                loss_dist_cov_ema = ((cov_u_ema - torch.eye(args.n_classes))**2).sum() / args.n_classes
                if it % 50 == 0: # 50
                    logger.info("loss_dist_proto: {:.6f}. ".format(loss_dist_proto.item()))
                loss_distribution = loss_dist_proto
        
        if it == 1 and args.analysis == True:
            logger.info("GT loss: ")
            logger.info(list(((criteria_u(logits_u_s, lbs_u_real)).cpu().detach().numpy())))
            logger.info("Top-1 Prob: ")
            logger.info(list(scores.cpu().detach().numpy()))
            logger.info("Prediction class: ")
            logger.info(list(lbs_u_guess.cpu().detach().numpy()))
            logger.info("Prediction prob: ")
            logger.info(list([probs[i,j].cpu().detach().item() for i,j in enumerate(lbs_u_guess)])[:10])
            logger.info("Mask: ")
            logger.info(list(mask.cpu().detach().numpy()))
            logger.info("Label: ")
            logger.info(list(lbs_u_real.cpu().detach().numpy()))
        
        if args.adapl == True:
            adapl_value_select = (torch.tensor(adapl_ul)*mask).sum()/(mask.sum()+1e-5)
            adapl_value_select_meter.update(adapl_value_select.item())
            adapl_value_unselect = (torch.tensor(adapl_ul)*(1-mask)).sum()/((1-mask).sum()+1e-5)
            adapl_value_unselect_meter.update(adapl_value_unselect.item())
        

        if args.distribution_adapl:
            loss = loss_x + args.lam_u * loss_u + loss_distribution*args.lam_dist 
        optim.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) 

        optim.step()
        lr_schdlr.step()

        if it % args.ema_step == 0:
            if args.eval_ema:
                    with torch.no_grad():
                        ema_model_update(model, ema_model, args.ema_m) 

        
        adapl_loss_meter.update(0.0)

        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        mask_meter.update(mask.mean().item())

        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        corr_u_lb_all = (lbs_u_guess == lbs_u_real).float()
        n_correct_u_lbs_all_meter.update(corr_u_lb_all.sum().item())
        guess_recall.update((corr_u_lb.sum()/(corr_u_lb_all.sum()+1e-5)).item())

        if (it + 1) % 64 == 0:
            t = time.time() - epoch_start
            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_u: {:.3f}. loss_x: {:.3f}. "
                    "n_correct_u: {:.2f}/{:.2f}. n_correct_u_all: {:.2f}. guess_recall: {:.2f}. Mask:{:.3f}. LR: {:.6f}. Time: {:.2f}".format(
            args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_u_meter.avg, loss_x_meter.avg,
            n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, n_correct_u_lbs_all_meter.avg, guess_recall.avg,
            mask_meter.avg, lr_log, t))

            epoch_start = time.time()

    return loss_x_meter.avg, loss_u_meter.avg, mask_meter.avg, \
            n_correct_u_lbs_meter.avg/n_strong_aug_meter.avg, \
            n_correct_u_lbs_all_meter.avg/lbs_u_real.shape[0], guess_recall.avg, \
            prob_list, l_probs_list


def evaluate(model, ema_model, dataloader, criterion, args=None, each_class_acc=False):
    model.eval()
    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()
    
    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            
            logits = model(ims)
            scores = torch.softmax(logits, dim=1)
            if each_class_acc == True:
                pred = np.argmax(scores.cpu().numpy(),1)
                class_acc_list = []
                for ii in range(10):
                    tmp_label = lbs.cpu().numpy()   # lbs.size=[64], range:(0-9)
                    for iter, kk in enumerate(tmp_label):  # np.argwhere(lbs.cpu().numpy()==ii)
                        if kk != ii:
                            tmp_label[iter] = -1
                    class_acc_list.append((pred == tmp_label).sum().item() / ((lbs==ii).sum().item()+1e-5))
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            
            if ema_model is not None:
                logits = ema_model(ims)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))                
                ema_top1_meter.update(top1.item())
    
    if each_class_acc == True:
        return top1_meter.avg, ema_top1_meter.avg, class_acc_list
    else:
        return top1_meter.avg, ema_top1_meter.avg


def main():
    parser = argparse.ArgumentParser(description='Semi-supervised Training')
    parser.add_argument('--root', default='/home/qsyang2/codes/ssl/datasets', type=str, help='dataset directory')
    
    parser.add_argument('--backbone', type=str, default='alexnet ',   # 'WideResnet', 'resnet18', 'resnet50'
                        help='name of used backbone')
    parser.add_argument('--wresnet-k', default=2, type=int,  
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int, 
                        help='depth of wide resnet')    
    
    parser.add_argument('--pre-trained', default=True, type=ast.literal_eval, help='use ImageNet pre-trained parameters')
    
    parser.add_argument('--dataset', type=str, default='isic2018 ', 
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=7,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=350,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=256,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,  #### ratio of unlabelled/labelled data
                        help='factor of train batch size of unlabeled samples')
    
    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)    

    parser.add_argument('--n-imgs-per-epoch', type=int, default=3560,  #64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=1e-3,  # 0.03
                        help='learning rate for training') 
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument('--DA', default=True, help='use distribution alignment')

    parser.add_argument('--thr', type=float, default=0.95,   # threshold for hard-pseudo label
                        help='pseudo label threshold')   
    
    parser.add_argument('--exp-dir', default='FixMatch', type=str, help='experiment directory')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')

    parser.add_argument('--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')

    parser.add_argument('--aug-type', type=str, default='RA',
                    help='type of used augmentation techniques')  #['RA', 'CTA']

    # =========================================================================================================
    parser.add_argument('--adapl', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--adapl-WEIGHT', default=0.1, type=float, help='weight of adapl loss') 
    parser.add_argument('--adapl-dist', choices=['mse','kl','emd'], default='mse', type=str, help='distance metric for two distributions') 
    parser.add_argument('--Ent', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--gamma', default=1, type=int, help='temperature factor for sigmoid re-weight term')  #[1,4]
    parser.add_argument('--adapl-use-weak', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--adapl-use-strong', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--grad-adapl', default=False, type=ast.literal_eval) #bool

    parser.add_argument('--min-weight', type=float, default=0.5, help='min weight value')
    parser.add_argument('--label-smooth', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--Corr-upperbound', default=False, type=ast.literal_eval) #bool

    parser.add_argument('--distribution-adapl', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--lam-dist', type=float, default=0.1, help='coefficient of distribution-adapl loss')
    parser.add_argument('--thr-dist', type=float, default=0.8, help='distribution examples threshold')   # threshold for hard-pseudo label  

    parser.add_argument('--standard-DA', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--DA-T', default=0.5, type=float, help='temperature factor for sharpening of standard_DA')

    parser.add_argument('--analysis', default=True, type=ast.literal_eval) #bool
    
    parser.add_argument('--setting', default='fixmatch_base', type=str, help='setting message for saving logs')

    parser.add_argument('--mean-teacher', default=False, type=ast.literal_eval) #bool
    # ==========================================================================================================

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    
    logger, output_dir = setup_default_logging(args)

    logger.info(dict(args._get_kwargs()))
    
    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)

    if args.seed > 0:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  
    n_iters_all = n_iters_per_epoch * args.n_epoches 

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")
    
    model, criteria_x, criteria_u, ema_model, save_epoch = set_model(args)
    
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_x, dltrain_u = get_train_loader(
        args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root, method='fixmatch', aug_type=args.aug_type, seed=args.seed, noisy_rate=args.noisy_rate)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root, aug_type=args.aug_type, seed=args.seed)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)  
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        optim = torch.optim.SGD(param_list, lr=checkpoint['optim_dict']['param_groups'][0]['lr'])
        optim.load_state_dict(checkpoint['optim_dict']) 
        lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0, last_epoch=n_iters_per_epoch*checkpoint['epoch'])
    else:
        optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                        momentum=args.momentum, nesterov=True)
        lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    prob_list = []
    l_probs_list = []
    prob_list_ema = []
    train_args = dict(
        model=model,
        ema_model=ema_model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger,
        prob_list=prob_list,
        l_probs_list=l_probs_list,
        prob_list_ema=prob_list_ema,
        example_stats=example_stats
    )
    best_acc = -1
    best_epoch = 0

    global iters
    iters = 0

    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        if args.checkpoint != '':
            epoch = save_epoch

        loss_x, loss_u, mask_mean, guess_label_acc, guess_label_all_acc, guess_label_recall, prob_list, l_probs_list = train_one_epoch(epoch, **train_args)
        
        if (epoch < 200 and epoch % 20 == 0 and epoch != 0) or (epoch > 200 and epoch % 50 == 0):
            visualizaion(model, ema_model, dltrain_x, dltrain_u, epoch, output_dir)
      
        top1, ema_top1, class_acc = evaluate(model, ema_model, dlval, criteria_x, args, each_class_acc=True)
    
        tb_logger.log_value('loss_x', loss_x, epoch)
        tb_logger.log_value('loss_u', loss_u, epoch)
        tb_logger.log_value('guess_label_acc', guess_label_acc, epoch)
        tb_logger.log_value('guess_label_all_acc', guess_label_all_acc, epoch)
        tb_logger.log_value('guess_label_recall', guess_label_recall, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('test_ema_acc', ema_top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)
        
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        if (best_acc < top1 and (best_epoch+1) >= 100) or (epoch % 50 == 0 and 'debug' not in args.setting):
            file_name = os.path.join(output_dir, args.dataset+'_'+str(args.n_labeled)+'_'+str(best_acc)+'_epoch_{}.pth'.format(epoch+1))
            torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optim.state_dict(),
            'lr_schdlr': lr_schdlr.state_dict(),
            'prob_list': prob_list,
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            },
            file_name)

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, best_acc, best_epoch))

if __name__ == '__main__':
    main()
