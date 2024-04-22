import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data import ModelNet40, ScanObjectNN
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'quant'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_quant_cls', help='model name [default: pointnet_quant_cls]/pointnet_cls')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40', 'scanobjectnn'])
    parser.add_argument('--num_category', default=40, type=int, choices=[40, 15], help='training on ModelNet40/ScanObjectNN')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    # new argument
    parser.add_argument('--max_probability', type=float, default=0.1, help='max probability to protect point')
    parser.add_argument('--clip', type=float, default=0.001, help='percentile to drop value in quantization')
    parser.add_argument('--quant_bit', type=int, default=8, help='quant bit')
    parser.add_argument('--loss2_weight', type=float, default=0.1, help='loss2 weight')
    parser.add_argument('--knn_k', type=int, default=20, help='semantic module knn k points')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda().squeeze()

        points = points.transpose(2, 1)
        if args.model == 'pointnet_cls':
            pred, _ = classifier(points)
        else:
            pred, _, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    if args.dataset == 'modelnet40':
        args.num_category = 40
        trainDataLoader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=8, batch_size=args.batch_size, shuffle=True,
                                     drop_last=True)
        testDataLoader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=8, batch_size=args.batch_size, shuffle=True,
                                    drop_last=False)
    elif args.dataset == 'scanobjectnn':
        args.num_category = 15
        trainDataLoader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=8, batch_size=args.batch_size,
                                     shuffle=True, drop_last=True)
        testDataLoader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=8, batch_size=args.batch_size,
                                    shuffle=True, drop_last=False)
    else:
        raise Exception("Not implemented")
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./train_classification.py', str(exp_dir))

    from common import CommonIntWeightPerChannelQuant
    from common import CommonIntActQuant
    from common import CommonUintActQuant

    class IW(CommonIntWeightPerChannelQuant):
        bit_width = args.quant_bit

    class IA(CommonIntActQuant):
        bit_width = args.quant_bit

    class UIA(CommonUintActQuant):
        bit_width = args.quant_bit

    def make_quantizer():
        return IW, IA, UIA

    if args.model == 'pointnet_cls':
        classifier = model.get_model(num_class)
    else:
        classifier = model.get_model(num_class, model_quantizer=make_quantizer, max_probability=args.max_probability, knn_k=args.knn_k)
    criterion = model.get_loss()
    if args.model != 'pointnet_cls':
        criterion2 = model.get_loss2(args.loss2_weight)
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        if args.model != 'pointnet_cls':
            criterion2 = criterion2.cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda().squeeze()

            if args.model == 'pointnet_cls':
                pred, trans_feat = classifier(points)
                loss = criterion(pred, target.long(), trans_feat)
            else:
                pred, trans_feat, pred_importance, lable_importance = classifier(points)
                loss1 = criterion(pred, target.long(), trans_feat)
                loss2 = criterion2(pred_importance, lable_importance.float())
                loss = loss1 + loss2
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
        scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if instance_acc >= best_instance_acc:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
