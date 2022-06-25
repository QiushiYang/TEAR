import time
import logging
import os
import sys
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pylab import *


def setup_default_logging(args, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    
    output_dir = os.path.join('./logs/'+args.dataset, args.backbone, f'x{args.n_labeled}_seed{args.seed}', args.setting)
        
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('train')

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, output_dir


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # return value, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-20)

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return time.strftime(fmt)


class WarmupCosineLrScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))            
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


def plot_embedding(data, label, title, dist_wise=False, _color=0, _marker='o', _s=8):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    color_bank = ['royalblue', 'violet', 'r', 'darkseagreen', 'darkorange']
    label_bank = ['unlabeled data', 'labeled data', 'class prototype']

    if dist_wise:  
        for i in range(data.shape[0]):
            if i < data.shape[0]-1:
                plt.scatter(data[i, 0], data[i, 1], s=_s, marker=_marker,
                        c=color_bank[_color],)  
            else:
                plt.scatter(data[i, 0], data[i, 1], label=label_bank[_color], s=_s, marker=_marker,
                    c=color_bank[_color],)
    else:
        for i in range(data.shape[0]):
            plt.scatter(data[i, 0], data[i, 1], label=str(label[i]), marker=_marker,
                    s=_s, c=plt.cm.Set1(label[i] / 10.), 
            )

    plt.title(title)

def visualizaion(model, ema_model, dltrain_x, dltrain_u, epoch, output_dir, iter_num=5):
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    ims_x_weak, _lbs_x = next(dl_x)
    feat_dim = model(ims_x_weak.cuda(), out_fea=True)[1].shape[1]

    l_fea_all = torch.ones((0,feat_dim))
    l_label_all = torch.ones((0))
    ii = 0
    iter_num_l = 7
    iter_num_ul = 1  
    for ims, lbs in dltrain_x:
        ims = ims.cuda()
        lbs = lbs.cuda()
        logits, fea = model(ims, out_fea=True)
        if ii < iter_num_l:
            l_fea_all = torch.cat((l_fea_all, fea.cpu().detach()), 0)
            l_label_all = torch.cat((l_label_all, lbs.cpu()), 0)
            ii += 1
        else:
            break
    
    ul_fea_all = torch.ones((0,feat_dim)) 
    ul_label_all = torch.ones((0))
    ii = 0
    for ims, lbs in dltrain_u:
        ims = ims[0].cuda()
        lbs = lbs.cuda()
        logits, fea = model(ims, out_fea=True)
        if ii < iter_num_ul:
            ul_fea_all = torch.cat((ul_fea_all, fea.cpu().detach()), 0)
            ul_label_all = torch.cat((ul_label_all, lbs.cpu()), 0)
            ii += 1
        else:
            break

    vis_all = torch.cat((l_fea_all, ul_fea_all), 0)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result_all = tsne.fit_transform(vis_all.numpy())
    result_l = result_all[:l_fea_all.shape[0]]
    result_ul = result_all[l_fea_all.shape[0]:]

    fig = plt.figure()
    plot_embedding(result_l, l_label_all.numpy(),
                        't-SNE embedding of the labeled & unlabeled features', dist_wise=True, _color=1)
                    
    plot_embedding(result_ul, ul_label_all.numpy(),
                        't-SNE embedding of the labeled & unlabeled features', dist_wise=True, _color=0)
    
    tick_params(top='on',bottom='on',left='on',right='on')
    tick_params(which='both',direction='in')
    plt.legend()
    vis_save_dir = os.path.join(output_dir, 'vis')
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)
    plt.savefig(vis_save_dir + '/all_dist_epoch_'+ str(epoch) +'.png')

