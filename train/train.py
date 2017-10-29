from __future__ import print_function

import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch
from hparams import hp
from networks import PrimitiveRNN
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import ShapeNet
from modules.binvox import Voxel
from modules.render import render_primitives, render_voxel


if __name__ == '__main__':
    default_path = "/data/vision/billf/jwu-phys/scene3d/dataset/" \
                   "SceneRGBD/pySceneNetRGBD/data"

    # set up argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'actual')
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--clean', action = 'store_true')

    # dataset
    parser.add_argument('--data_path', default = default_path)
    parser.add_argument('--synset', default = '')

    # training
    parser.add_argument('--epochs', default = 500, type = int)
    parser.add_argument('--batch', default = 16, type = int)
    parser.add_argument('--snapshot', default = 10, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--gpu', default = '7')

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
    for split in ['train', 'test']:
        data[split] = ShapeNet(data_path = os.path.join(args.data_path, args.synset), split = split)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0} + {1}'.format(len(data['train']), len(data['test'])))

    # set up model and convert into cuda
    model = PrimitiveRNN().cuda()
    print('==> model loaded')

    # set up optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    print('==> optimizer loaded')

    # set up experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = args.clean)
    logger = utils.Logger(exp_path)
    print('==> save logs to {0}'.format(exp_path))

    # load snapshot of model and optimizer
    if args.resume is not None:
        if os.path.isfile(args.resume):
            snapshot = torch.load(args.resume)
            epoch = snapshot['epoch']
            model.load_state_dict(snapshot['model'])
            optimizer.load_state_dict(snapshot['optimizer'])
            print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
        else:
            raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))
    else:
        epoch = 0

    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])

        # training the model
        model.train()
        for voxels, ((c, s, t, q), lengths) in tqdm(loaders['train'], desc = 'epoch %d' % (epoch + 1)):
            # convert voxels into cuda tensor
            voxels = Variable(voxels.cuda()).float()

            # convert c, s, t and q into cuda tensors
            c = Variable(c.cuda()).float()
            s = Variable(s.cuda()).float()
            t = Variable(t.cuda()).float()
            q = Variable(q.cuda()).float()

            # set up cstq for inference
            cstq = torch.cat([c, s, t, q], dim = 2)

            # set up start tokens
            batch_size = voxels.size(0)
            stop_token = Variable(torch.from_numpy(hp.stop_token).float()).cuda()
            stop_tokens = torch.stack([stop_token.unsqueeze(0)] * batch_size)

            # set up labels
            labels = torch.cat([cstq, stop_tokens], 1)

            # initilize optimizer
            optimizer.zero_grad()

            # forward pass
            outputs = model.forward(voxels, cstq)
            loss, (loss_c, loss_s, loss_t, loss_q) = chamfer_distance(outputs, labels, lengths)

            # add summary to logger
            logger.scalar_summary('loss', loss.data[0], step)
            logger.scalar_summary('loss_c', loss_c.data[0], step)
            logger.scalar_summary('loss_s', loss_s.data[0], step)
            logger.scalar_summary('loss_t', loss_t.data[0], step)
            logger.scalar_summary('loss_q', loss_q.data[0], step)
            step += batch_size

            # backward pass
            loss.backward()
            optimizer.step()

        if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
            # snapshot model and optimizer
            snapshot = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(snapshot, os.path.join(exp_path, 'epoch-%d.pth' % (epoch + 1)))
            torch.save(snapshot, os.path.join(exp_path, 'latest.pth'))
            print('==> saved snapshot to "{0}"'.format(os.path.join(exp_path, 'epoch-%d.pth' % (epoch + 1))))

            # testing the model
            model.train(False)
            for split in ['train', 'test']:
                # sample one batch from dataset
                voxels, ((c, s, t, q), lengths) = iter(loaders[split]).next()

                # convert voxels into cuda tensor
                voxels = Variable(voxels.cuda()).float()

                # convert c, s, t and q into cuda tensors
                c = Variable(c.cuda()).float()
                s = Variable(s.cuda()).float()
                t = Variable(t.cuda()).float()
                q = Variable(q.cuda()).float()

                # set up cstq for inference
                cstq = torch.cat([c, s, t, q], dim = 2)

                # set up start tokens
                batch_size = voxels.size(0)
                stop_token = Variable(torch.from_numpy(hp.stop_token).float()).cuda()
                stop_tokens = torch.stack([stop_token.unsqueeze(0)] * batch_size)

                # set up labels
                labels = torch.cat([cstq, stop_tokens], 1)

                # forward pass
                outputs = model.forward(voxels)

                # visualize inputs and outputs
                voxels, outputs, labels = visualize(voxels, outputs, labels)

                # add summary to logger
                logger.image_summary('%s-inputs' % split, voxels, step)
                logger.image_summary('%s-outputs' % split, outputs, step)
                logger.image_summary('%s-labels' % split, labels, step)
