import pdb
import numpy as np
import copy
import argparse
import os
import time
import logging
from random import uniform
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from ast import literal_eval
from torch.nn.utils import clip_grad_norm
from math import ceil

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='data augment')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='data augment')
parser.set_defaults(augment=False)
parser.add_argument('--results_dir', metavar='RESULTS_DIR',
                    default='./TrainingResults', help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N', help='mini-batch size (default: 2048)')
parser.add_argument('-mb', '--mini-batch-size', default=128, type=int,
                    help='mini-mini-batch size (default: 128)')
parser.add_argument('--lr_bb_fix', dest='lr_bb_fix', action='store_true',
                    help='learning rate fix for big batch lr =  lr0*(batch_size/128)**0.5')
parser.add_argument('--no-lr_bb_fix', dest='lr_bb_fix', action='store_false',
                    help='learning rate fix for big batch lr =  lr0*(batch_size/128)**0.5')
parser.set_defaults(lr_bb_fix=True)
parser.add_argument('--regime_bb_fix', dest='regime_bb_fix', action='store_true',
                    help='regime fix for big batch e = e0*(batch_size/128)')
parser.add_argument('--no-regime_bb_fix', dest='regime_bb_fix', action='store_false',
                    help='regime fix for big batch e = e0*(batch_size/128)')
parser.set_defaults(regime_bb_fix=False)
parser.add_argument('--visualize_train', dest='visualize_train', action='store_true',
                    help='visualize train sharpness')
parser.add_argument('--no-visualize_train', dest='visualize_train', action='store_false',
                    help='visualize train sharpness')
parser.set_defaults(visualize_train=True)
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--sharpness-smoothing', '--ss', default=1e-4, type=float,
                    metavar='SS', help='sharpness smoothing (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='master evaluate model FILE on validation set')
#parser.add_argument('-se', '--slave-evaluate', type=str, metavar='FILE',
#                    help='slave evaluate model FILE on validation set')
parser.add_argument('--alpha', type=str, default='-1:0.1:1.01', metavar='FILE',
                    help='Range to explore near minimum')

def main():
    #torch.manual_seed(123)
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    if args.regime_bb_fix:
            args.epochs *= (int)(ceil(args.batch_size / args.mini_batch_size))

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise OSError('Directory {%s} exists. Use a new one.' % save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        #torch.cuda.manual_seed_all(123)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)


    # optionally preload from a slave and master models
    slave_checkpoint = None
    checkpoint = None
    alpha = [0.]
    alpha_str = args.alpha.split(':')
    alpha = np.arange(float(alpha_str[0]), float(alpha_str[2]), float(alpha_str[1]))

    # resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    else:
        raise ImportError("Please specify your master model path.")

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=args.augment),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        # get directions
        directions={}
        for key, p in model.named_parameters():
            _direction = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.)
            #_direction = _direction/_direction.norm()*p.norm().data
            #_direction = torch.cuda.FloatTensor(p.size()).normal_()
            for idx in range(0, _direction.shape[0]):
                if 1==len(_direction.shape):
                  _direction[idx] = _direction[idx] /np.linalg.norm(_direction[idx]) * np.linalg.norm(p.data[idx])
                else:
                  _direction[idx] = _direction[idx]/_direction[idx].norm()*p.data[idx].norm()

            #_direction = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.) * torch.abs(p.data)

            #_direction = torch.abs(p.data)
            directions[key] = _direction

        data_res = np.zeros((len(alpha), 4))
        data_idx = 0
        # for each alpha, modify weights and evaluate
        for _alpha in alpha:

          for key, p in model.named_parameters():
              p.data.add_(directions[key]*_alpha)

          val_result = validate(val_loader, model, criterion, 0)

          for key, p in model.named_parameters():
              p.data.sub_(directions[key]*_alpha)

          val_loss, val_prec1, val_prec5 = [val_result[r]
                                            for r in ['loss', 'prec1', 'prec5']]
          logging.info('\nalpha {_alpha} \t'
                       'Validation Loss {val_loss:.4f} \t'
                       'Validation Prec@1 {val_prec1:.3f} \t'
                       'Validation Prec@5 {val_prec5:.3f} \n'
                       .format(_alpha=_alpha,
                               val_loss=val_loss,
                               val_prec1=val_prec1,
                               val_prec5=val_prec5))
          if args.visualize_train:

            for key, p in model.named_parameters():
                p.data.add_(directions[key] * _alpha)

            train_result = validate(train_loader, model, criterion, 0)

            for key, p in model.named_parameters():
                p.data.sub_(directions[key] * _alpha)

            train_loss, train_prec1, train_prec5 = [train_result[r]
                                              for r in ['loss', 'prec1', 'prec5']]
            logging.info('\nalpha {_alpha} \t'
                         'Train Loss {train_loss:.4f} \t'
                         'Train Prec@1 {train_prec1:.3f} \t'
                         'Train Prec@5 {train_prec5:.3f} \n'
                         .format(_alpha=_alpha,
                                 train_loss= train_loss,
                                 train_prec1=train_prec1,
                                 train_prec5=train_prec5))
            data_res[data_idx, 2] = train_loss
            data_res[data_idx, 3] = train_prec1
          data_res[data_idx, 0] = val_loss
          data_res[data_idx, 1] = val_prec1
          data_idx += 1

        # plotting
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.semilogy(alpha, data_res[:, 0], 'b--')
        if args.visualize_train:
          ax1.semilogy(alpha, data_res[:, 2], 'b-')
        #ax1.plot(alpha, data_res[:, 0], 'b-')
        #ax1.plot(alpha, data_res[:, 2], 'b--')

        ax2.plot(alpha, data_res[:, 1], 'r--')
        if args.visualize_train:
          ax2.plot(alpha, data_res[:, 3], 'r-')

        ax1.set_xlabel('alpha')
        ax1.set_ylabel('Cross Entropy', color='b')
        ax2.set_ylabel('Accuracy', color='r')
        if args.visualize_train:
          ax1.legend(('Test', 'Train'), loc=0)
        else:
          ax1.legend(('Test'), loc=0)

#        ax1.grid(b=True, which='both')
        plt.savefig('res.pdf')
        plt.show()
        print 'Done'
        return

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()


    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # compute output
        if not training:
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

        else:
            mini_inputs = input_var.chunk(args.batch_size // args.mini_batch_size)
            mini_targets = target_var.chunk(args.batch_size // args.mini_batch_size)


            optimizer.zero_grad()

            for k, mini_input_var in enumerate(mini_inputs):

                noises = {}
                noise_idx = 0
                # randomly change current model @ each mini-mini-batch
                for p in model.parameters():
                  noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.) * args.sharpness_smoothing * args.lr
                  #noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.) * args.sharpness_smoothing * optimizer.param_groups[0]['lr']
                  noises[noise_idx] = noise
                  noise_idx += 1
                  p.data.add_(noise)

                mini_target_var = mini_targets[k]
                output = model(mini_input_var)
                loss = criterion(output, mini_target_var)

                prec1, prec5 = accuracy(output.data, mini_target_var.data, topk=(1, 5))
                losses.update(loss.data[0], mini_input_var.size(0))
                top1.update(prec1[0], mini_input_var.size(0))
                top5.update(prec5[0], mini_input_var.size(0))

                # compute gradient and do SGD step
                loss.backward()

                # denoise @ each mini-mini-batch. Do we need denoising???
                noise_idx = 0
                for p in model.parameters():
                  p.data.sub_(noises[noise_idx])
                  noise_idx += 1

            for p in model.parameters():
                p.grad.data.div_(len(mini_inputs))
            clip_grad_norm(model.parameters(), 5.)
            optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return {'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg}


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
