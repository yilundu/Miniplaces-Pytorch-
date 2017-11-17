# example use: python predict_probs.py --path ./exp/proto/latest.pth --gpu 0 --output submit-1.csv

from __future__ import print_function

import argparse
import os
from multiprocessing import Pool

import utils
import numpy as np
import pandas as pd
import torch
# from hparams import hp
from PIL import Image
from resnet import resnet50
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.models as models
from tqdm import tqdm
from ayang_net import AyangNet
from torchvision import transforms

from data import MiniPlace
import scipy.misc


def reconstruct_image(im):
    im = im.numpy()
    im = np.transpose(im, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im = 256 * (im * std + mean)
    return im


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


NAME_TO_MODEL = {
    'resnet50': resnet50(num_classes=100),
    'ayangnet': AyangNet(),
    'densenet': models.densenet169(num_classes=100)
}


if __name__ == '__main__':
    default_path = './preprocess'
    loss_fn = CrossEntropyLoss()

    # set up argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--path', default = None, type = str)
    parser.add_argument('--clean', action = 'store_true')

    # dataset
    parser.add_argument('--data_path', default = default_path)
    parser.add_argument('--synset', default = '')
    parser.add_argument('--categories', default = '../data/categories.txt')

    # training
    parser.add_argument('--batch', default = 1, type = int)
    parser.add_argument('--workers', default = 1, type = int)
    parser.add_argument('--gpu', default = '7')
    parser.add_argument('--name', default = 'resnet50')

    # submitting
    parser.add_argument('--output', default = 'submit.csv')

    # parse arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # set up gpus for training
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set up datasets and loaders
    data, loaders = {}, {}
    for split in ['test']:
        data[split] = MiniPlace(data_path = os.path.join(args.data_path, args.synset), split = split)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = False, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0}'.format(len(data['test'])))

    # set up map for different categories
    categories = np.genfromtxt(args.categories, dtype='str')[:, 0]

    # set up model and convert into cuda
    model = NAME_TO_MODEL[args.name].cuda()
    print('==> model loaded')

    # load snapshot of model and optimizer
    if args.path is not None:
        if os.path.isfile(args.path):
            snapshot = torch.load(args.path)
            epoch = snapshot['epoch']
            model.load_state_dict(snapshot['model'])
            # If this doesn't work, can use optimizer.load_state_dict
            # optimizer.load_state_dict(snapshot['optimizer'])
            print('==> snapshot "{0}" loaded (epoch {1})'.format(args.path, epoch))
        else:
            raise FileNotFoundError('no snapshot found at "{0}"'.format(args.path))
    else:
        epoch = 0

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize])

    # testing the model
    model.eval()
    answers = []
    inds = []
    ls = []
    # outputs = []
    for i in range(0, 10000, 16):
        outputs = torch.zeros(16, 100)
        for k in range(5):
            list_im = []
            for j in range(i, i + 16):
                path = 'test/%08d.jpg' % (j + 1)
                image = (scipy.misc.imread(os.path.join('../data/images/', path)))
                image = Image.fromarray(scipy.misc.imresize(image, (256,256)))
                image = preprocess(image)
                list_im.append(image)
            list_im = Variable(torch.stack(list_im).cuda())
            outputs += model.forward(list_im).cpu().data
        for output in outputs:
            tmp = output.numpy().flatten()
            # tmp2 = tmp.argsort()[-5:][::-1]
            answers.append(tmp)
            # print(tmp2)
        if i % 400 == 400 - 16:
            print(i + 16, " / 10000")

    out = pd.DataFrame(data=answers)
    out.to_csv('predictions/' + args.output)
    # f = open('predictions/' + args.output, 'w+')
    # for i in range(10000):
    #     s = 'test/%08d.jpg' % (i + 1)
    #     for ans in answers[i]:
    #         s += ' ' + str(ans)
    #     f.write(s + '\n')
    #     if i % 100 == 99:
    #         print(i + 1, " / 10000")
    # f.close()
