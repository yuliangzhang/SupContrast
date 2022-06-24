from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.audio_net import SupCENet
from audio_utils.data_esc50_generator import (Esc50Dataset, Esc50TrainSampler, Esc50BalancedTrainSampler,
    Esc50AlternateTrainSampler, Esc50EvaluateSampler, esc50_collate_fn)
from audio_utils.data_fsd50k_generator import (Fsd50kDataset, Fsd50kTrainSampler, Fsd50kBalancedTrainSampler,
    Fsd50kAlternateTrainSampler, Fsd50kEvaluateSampler, fsd50k_collate_fn)
from audio_utils.pytorch_utils import move_data_to_device

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # audio processing parameters
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)

    # data organized method and augmentation method
    parser.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser.add_argument('--augmentation', type=str, default='none', choices=['none', 'mixup'])
    parser.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='esc50',
                        choices=['esc50', 'fsd50k'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    # set dataset of audio file in hdf5 format
    opt.data_folder = '/media/storage/home/22828187/audioset_tagging_cnn/hdf5s/'

    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'esc50':
        opt.n_cls = 50
    elif opt.dataset == 'fsd50k':
        opt.n_cls = 200
    elif opt.dataset == 'esc50_major':
        opt.ncls = 5
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def get_esc50_data_loader(opt):

    balanced = opt.balanced
    batch_size = opt.batch_size
    augmentation = opt.augmentation
    # Paths
    black_list_csv = None

    num_workers = opt.num_workers

    dataset = Esc50Dataset(sample_rate=opt.sample_rate)
    train_indexes_hdf5_path = os.path.join(opt.data_folder, 'indexes', 'esc_dev.h5')
    eval_bal_indexes_hdf5_path = os.path.join(opt.data_folder, 'indexes', 'esc_eval.h5')
    # Train sampler
    if balanced == 'none':
        Sampler = Esc50TrainSampler
    elif balanced == 'balanced':
        Sampler = Esc50BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = Esc50AlternateTrainSampler

    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path,
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)
    validate_sampler = Esc50EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path,
        batch_size=batch_size)
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_sampler=train_sampler, collate_fn=esc50_collate_fn,
                                               num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_sampler=validate_sampler, collate_fn=esc50_collate_fn,
                                                  num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_fsd50k_data_loader(opt):
    balanced = opt.balanced
    batch_size = opt.batch_size
    augmentation = opt.augmentation
    # Paths
    black_list_csv = None

    num_workers = opt.num_workers

    dataset = Fsd50kDataset(sample_rate=opt.sample_rate)
    train_indexes_hdf5_path = os.path.join(opt.data_folder, 'indexes', 'fsd50k_dev.h5')
    eval_bal_indexes_hdf5_path = os.path.join(opt.data_folder, 'indexes', 'fsd50k_eval.h5')
    # Train sampler
    if balanced == 'none':
        Sampler = Fsd50kTrainSampler
    elif balanced == 'balanced':
        Sampler = Fsd50kBalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = Fsd50kAlternateTrainSampler

    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path,
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)
    validate_sampler = Fsd50kEvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path,
        batch_size=batch_size)
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_sampler=train_sampler, collate_fn=fsd50k_collate_fn,
                                               num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=validate_sampler, collate_fn=fsd50k_collate_fn,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def set_audio_loader(opt):
    # construct data loader
    if 'esc50' in opt.dataset:
        return get_esc50_data_loader(opt)
    elif opt.dataset == 'fsd50k':
        return get_fsd50k_data_loader(opt)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))



def set_model(opt):

    model = SupCENet(sample_rate=opt.sample_rate,
                     window_size=opt.window_size,
                     hop_size=opt.hop_size,
                     mel_bins=opt.mel_bins,
                     fmin=opt.fmin,
                     fmax=opt.fmax,
                     name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    augmentation = opt.augmentation

    batch_num = len(train_loader)
    idx = 0


    end = time.time()
    for batch_data_dict in train_loader:
        idx += 1
        data_time.update(time.time() - end)

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        audio_waveforms = batch_data_dict['waveform']
        labels = batch_data_dict['target']
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(audio_waveforms)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
        if idx > batch_num:
            break
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, batch_data_dict in enumerate(val_loader):

            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            audio_waveforms = batch_data_dict['waveform']
            labels = batch_data_dict['target']
            bsz = labels.shape[0]

            # forward
            output = model(audio_waveforms)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_audio_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
