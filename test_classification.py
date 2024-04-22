import os
import sys
import torch
import numpy as np

import logging
import importlib
import argparse

from tqdm import tqdm
from data import ModelNet40, ScanObjectNN
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'quant'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_quant_cls', help='model name [default: pointnet_quant_cls]/pointnet_cls')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40', 'scanobjectnn'])
    parser.add_argument('--num_category', default=40, type=int, choices=[40, 15], help='training on ModelNet40/ScanObjectNN')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--quant_bit', type=int, default=8, help='quant bit')
    return parser.parse_args()


def test(model, loader, num_class=40):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

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
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    if args.dataset == 'modelnet40':
        args.num_category = 40
        testDataLoader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=8, batch_size=args.batch_size, shuffle=True,
                                    drop_last=False)
    elif args.dataset == 'scanobjectnn':
        args.num_category = 15
        testDataLoader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=8, batch_size=args.batch_size,
                                    shuffle=True, drop_last=False)
    else:
        raise Exception("Not implemented")
    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

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
        classifier = model.get_model(num_class, model_quantizer=make_quantizer)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(torch.device("cuda"))
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
