from __future__ import print_function

import argparse
import os
from multiprocessing import Pool

import utils
import numpy as np
import torch
# from hparams import hp
from resnet import resnet50
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm
import torchvision.models as models
from tqdm import tqdm
from ayang_net import AyangNet

from data import MiniPlace
from inception import inception_v3
from inception2 import inception_v4
from wideresnet import WideResNet
from densenet import DenseNet
from PIL import Image
import time
import transforms
import affine_transforms
import h5py
from scipy.misc import logsumexp


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.3 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def compute_softmax(i):
    i = i - i.min(axis=1, keepdims=True)
    log_sum = logsumexp(i, axis=1, keepdims=True)
    return i - log_sum


NAME_TO_MODEL = {
    'resnet50': resnet50(num_classes=100),
    'ayangnet': AyangNet(),
    'densenet': models.densenet161(num_classes=100),
    'inceptionv3': inception_v3(num_classes=100),
    'inceptionv4': inception_v4(num_classes=100),
    'wideresnet': WideResNet(28, 100, widen_factor=10),
    'widedensenet': DenseNet(60, (6, 6, 6, 6), 64, num_classes=100)
}


if __name__ == '__main__':
    default_path = './preprocess'
    noise_decay = 0.55
    loss_fn = CrossEntropyLoss()

    # set up argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'first')
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--clean', action = 'store_true')

    # dataset
    parser.add_argument('--data_path', default = default_path)
    parser.add_argument('--synset', default = '')
    parser.add_argument('--categories', default = '../data/categories.txt')

    # training
    parser.add_argument('--epochs', default = 500, type = int)
    parser.add_argument('--batch', default = 16, type = int)
    parser.add_argument('--snapshot', default = 2, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--gpu', default = '7')
    parser.add_argument('--name', default = 'resnet50')

    # Training Parameters
    parser.add_argument('--lr', default = 0.1, type = float)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--weight_decay', default = 1e-5, type = float)
    parser.add_argument('--noise', default = 0, type = float)

    # parse arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # set up gpus for training
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set up datasets and loaders
    file_path = './preprocess/miniplaces_256_val.h5'
    dataset = h5py.File(file_path)


    # set up map for different categories
    categories = np.genfromtxt(args.categories, dtype='str')[:, 0]

    # set up model and convert into cuda
    model = NAME_TO_MODEL[args.name].cuda()
    print('==> model loaded')
    best_top_5 = 0

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transform = [
        transforms.Scale(256),
        transforms.RandomCrop(224),
        # transforms.RandomResizedCrop(224)
    ]
    # transform.extend([transforms.RandomResizedCrop(224)])
    transform.extend([
        # for inception 1
        # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.0),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.35, hue=0.1),
        # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(),
    ])

    transform += [transforms.ToTensor()]

    # transform.append(
    # affine_transforms.Affine(rotation_range=20.0, translation_range=0.02, fill_mode='constant')
    # )

    transform += [normalize]

    preprocess = transforms.Compose(transform)


    # load snapshot of model and optimizer
    if args.resume is not None:
        if os.path.isfile(args.resume):
            snapshot = torch.load(args.resume)
            epoch = snapshot['epoch']
            model.load_state_dict(snapshot['model'])
            # If this doesn't work, can use optimizer.load_state_dict
            # optimizer.load_state_dict(snapshot['optimizer'])
            print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
        else:
            raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))
    epoch = 0

    # testing the model
    model.eval()
    count = 0
    offset = 2000
    for i in range(offset, 10000):
        image = dataset['images'][i]
        ans = np.zeros((1, 100))
        for j in range(11):
            img_tensor = preprocess(Image.fromarray(image))
            img_tensor = Variable(torch.stack([img_tensor]).cuda())

            temp_output = model.forward(img_tensor).cpu().data.numpy()
            temp_softmax = compute_softmax(temp_output)
            ans += temp_softmax
        tmp = ans.flatten()
        # tmp /= num_rep
        tmp2 = tmp.argsort()[-5:][::-1]
        if dataset['labels'][i] in tmp2:
            count += 1
        if i % 50 == 49:
            print(count, "/", i - offset + 1, "=", count * 1.0 / (i - offset + 1))
    print("Validation Accuracy: ", count, " / 10000 = ", count / 100.0)



        # top1 = AverageMeter()
        # top5 = AverageMeter()
        # for split in ['train', 'val']:
        #     # sample one batch from dataset
        #     images, labels = iter(loaders[split]).next()
        #     images = Variable(images.cuda()).float()

        #     # forward pass
        #     outputs = model.forward(images).cpu().data
        #     images = images.cpu().data

        #     # add summary to logger
        #     for image, output in zip(images, outputs):
        #         category = categories[output.numpy().flatten().argmax()]
        #         category = category.replace('/', '_')
        #         logger.image_summary('{}-outputs, category: {} '.format(split, category), [reconstruct_image(image)], step)

        # for images, labels in tqdm(loaders['val'], desc = 'epoch %d' % (epoch + 1)):
        #     outputs = model.forward(Variable(images.cuda())).cpu()

            # prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            # top1.update(prec1[0], images.size(0))
            # top5.update(prec5[0], images.size(0))

        # if top5.avg > best_top_5:
        #     best_top_5 = top5.avg

        #     # snapshot model and optimizer
        #     snapshot = {
        #         'epoch': epoch + 1,
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict()
        #     }
        #     torch.save(snapshot, os.path.join(exp_path, 'best.pth'))
        #     print('==> saved snapshot to "{0}"'.format(os.path.join(exp_path, 'best.pth')))

        # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} in validation'.format(top1=top1, top5=top5))
        # logger.scalar_summary('Top 1', top1.avg, epoch)
        # logger.scalar_summary('Top 5', top5.avg, epoch)
