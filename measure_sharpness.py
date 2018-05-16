import pdb
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
import numpy as np
import scipy.optimize as sciopt
import warnings
from sklearn import random_projection as rp
import re
#import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

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
parser.add_argument('-b', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch size (default: 2048)')
parser.add_argument('-mb', '--mini-batch-size', default=100, type=int,
                    help='mini-mini-batch size (default: 128)')
parser.add_argument('--save_all', dest='save_all', action='store_true',
                    help='save all better checkpoints')
parser.add_argument('--no-save_all', dest='save_all', action='store_false',
                    help='save all better checkpoints')
parser.set_defaults(save_all=False)
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='data augment')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='data augment')
parser.set_defaults(augment=True)
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--sharpness-smoothing', '--ss', default=0.0, type=float,
                    metavar='SS', help='sharpness smoothing (default: 0)')
parser.add_argument('--anneal-index', '--ai', default=0.55, type=float,
                    metavar='AI', help='Annealing index of noise (default: 0.55)')
parser.add_argument('--tanh-scale', '--ts', default=10., type=float,
                    metavar='TS', help='scale of transition in tanh')
parser.add_argument('--smoothing-type', default='tanh', type=str, metavar='ST',
                    help='The type of chaning smoothing noise: constant, anneal, or tanh')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--epsilon', default=0.0005, type=float,
                    help='epsilon to contrain the box size for sharpness measure')
parser.add_argument('-m', '--manifolds', default=0, type=int, metavar='M',
                    help='The dimensionality of manifolds to measure sharpness. (0: full-space)')
parser.add_argument('-t', '--times', default=1, type=int, metavar='T',
                    help='Times to average over for sharpness')
def main():
    #torch.manual_seed(123)
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

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
        if len(args.gpus) != 1:
          raise NotImplementedError('Please use one gpu.')
    else:
        args.gpus = None

    if args.batch_size != args.mini_batch_size:
      args.mini_batch_size = args.batch_size
      warnings.warn('--mini-batch-size is enforced to be set as --batch-size {}'.format(args.mini_batch_size),
                    RuntimeWarning)

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    else:
      raise ValueError("Please specify the path of evaluated model")

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

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

#    for k in model.state_dict():
#        if re.match('.*weight.*', k):
#            plt.figure()
#            plt.hist(model.state_dict()[k].cpu().numpy().reshape(-1), bins='auto')
#            plt.title(k)
#    plt.show()

    val_result = validate(val_loader, model, criterion, 0)
    val_loss, val_prec1, val_prec5 = [val_result[r]
                                      for r in ['loss', 'prec1', 'prec5']]
    logging.info('\nValidation Loss {val_loss:.4f} \t'
                 'Validation Prec@1 {val_prec1:.3f} \t'
                 'Validation Prec@5 {val_prec5:.3f} \n'
                 .format(val_loss=val_loss,
                         val_prec1=val_prec1,
                         val_prec5=val_prec5))
    sharpnesses= []
    for time in range(args.times):
        sharpness = get_sharpness(val_loader, model, criterion, manifolds=args.manifolds)
        sharpnesses.append(sharpness)
        logging.info('sharpness {} = {}'.format(time,sharpness))
    logging.info('sharpnesses = {}'.format(str(sharpnesses)))
    _std = np.std(sharpnesses)*np.sqrt(args.times)/np.sqrt(args.times-1)
    _mean = np.mean(sharpnesses)
    logging.info(u'mean sharpness = {sharpness:.4f}\u00b1{err:.4f}'.format(sharpness=_mean,err=_std))

    return


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)

    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    grad_vec = None
    if training:
      optimizer = torch.optim.SGD(model.parameters(), 1.0)
      optimizer.zero_grad()  # only zerout at the beginning


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

            for k, mini_input_var in enumerate(mini_inputs):

                mini_target_var = mini_targets[k]
                output = model(mini_input_var)
                loss = criterion(output, mini_target_var)

                prec1, prec5 = accuracy(output.data, mini_target_var.data, topk=(1, 5))
                losses.update(loss.data[0], mini_input_var.size(0))
                top1.update(prec1[0], mini_input_var.size(0))
                top5.update(prec5[0], mini_input_var.size(0))

                # compute gradient and do SGD step
                loss.backward()

            #optimizer.step() # no step in this case

    # reshape and averaging gradients
    if training:
      for p in model.parameters():
        p.grad.data.div_(len(data_loader))
        if grad_vec is None:
          grad_vec = p.grad.data.view(-1)
        else:
          grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))

    #logging.info('{phase} - \t'
    #             'Loss {loss.avg:.4f}\t'
    #             'Prec@1 {top1.avg:.3f}\t'
    #             'Prec@5 {top5.avg:.3f}'.format(
    #              phase='TRAINING' if training else 'EVALUATING',
    #              loss=losses, top1=top1, top5=top5))

    return {'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg}, grad_vec


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    raise NotImplementedError('train functionality is changed. Do not use it!')


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    res, _ = forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)
    return res


def get_minus_cross_entropy(x, data_loader, model, criterion, training=False):
  if type(x).__module__ == np.__name__:
    x = torch.from_numpy(x).float()
    x = x.cuda()
  # switch to evaluate mode
  model.eval()

  # fill vector x of parameters to model
  x_start = 0
  for p in model.parameters():
    psize = p.data.size()
    peltnum = 1
    for s in psize:
      peltnum *= s
    x_part = x[x_start:x_start+peltnum]
    p.data = x_part.view(psize)
    x_start += peltnum

  result, grads = forward(data_loader, model, criterion, 0,
                 training=training, optimizer=None)
  #print ('get_minus_cross_entropy {}!'.format(-result['loss']))
  return (-result['loss'], None if grads is None else grads.cpu().numpy().astype(np.float64))

def get_sharpness(data_loader, model, criterion, manifolds=0):

  # extract current x0
  x0 = None
  for p in model.parameters():
    if x0 is None:
      x0 = p.data.view(-1)
    else:
      x0 = torch.cat((x0, p.data.view(-1)))
  x0 = x0.cpu().numpy()

  # get current f_x
  f_x0, _ = get_minus_cross_entropy(x0, data_loader, model, criterion)
  f_x0 = -f_x0
  logging.info('min loss f_x0 = {loss:.4f}'.format(loss=f_x0))

  # get the bounds
  epsilon = args.epsilon
  # find the minimum
  if 0==manifolds:
    x_min = np.reshape(x0 - epsilon * (np.abs(x0) + 1), (x0.shape[0], 1))
    x_max = np.reshape(x0 + epsilon * (np.abs(x0) + 1), (x0.shape[0], 1))
    bounds = np.concatenate([x_min, x_max], 1)
    func = lambda x: get_minus_cross_entropy(x, data_loader, model, criterion, training=True)
    init_guess = x0
  else:
    warnings.warn("Small manifolds may not be able to explore the space.")
    assert(manifolds<=x0.shape[0])
    #transformer = rp.GaussianRandomProjection(n_components=manifolds)
    #transformer.fit(np.random.rand(manifolds, x0.shape[0]))
    #A_plus = transformer.components_
    #A = np.linalg.pinv(A_plus)
    A_plus = np.random.rand(manifolds, x0.shape[0])*2.-1.
    # normalize each column to unit length
    A_plus_norm = np.linalg.norm(A_plus, axis=1)
    A_plus = A_plus / np.reshape(A_plus_norm, (manifolds,1))
    A = np.linalg.pinv(A_plus)
    abs_bound = epsilon * (np.abs(np.dot(A_plus, x0))+1)
    abs_bound = np.reshape(abs_bound, (abs_bound.shape[0], 1))
    bounds = np.concatenate([-abs_bound, abs_bound], 1)
    def func(y):
      floss, fg = get_minus_cross_entropy(x0 + np.dot(A, y), data_loader, model, criterion, training=True)
      return floss, np.dot(np.transpose(A), fg)
    #func = lambda y: get_minus_cross_entropy(x0+np.dot(A, y), data_loader, model, criterion, training=True)
    init_guess = np.zeros(manifolds)

  #rand_selections = (np.random.rand(bounds.shape[0])+1e-6)*0.99
  #init_guess = np.multiply(1.-rand_selections, bounds[:,0])+np.multiply(rand_selections, bounds[:,1])

  minimum_x, f_x, d = sciopt.fmin_l_bfgs_b(
    func,
    init_guess,
    maxiter=10,
    bounds=bounds,
    #factr=10.,
    #pgtol=1.e-12,
    disp=1)
  f_x = -f_x
  logging.info('max loss f_x = {loss:.4f}'.format(loss=f_x))
  sharpness = (f_x - f_x0)/(1+f_x0)*100

  # recover the model
  x0 = torch.from_numpy(x0).float()
  x0 = x0.cuda()
  x_start = 0
  for p in model.parameters():
      psize = p.data.size()
      peltnum = 1
      for s in psize:
          peltnum *= s
      x_part = x0[x_start:x_start + peltnum]
      p.data = x_part.view(psize)
      x_start += peltnum

  return sharpness


if __name__ == '__main__':
    main()
