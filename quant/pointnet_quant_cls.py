import brevitas.nn as qnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.pointnet_quant_utils import PointNetQuantEncoder, feature_transform_reguliarzer
from quant.point_semantic_module import SemanticModule


def evaluate_prob_mask(prob_mask):
    return torch.bernoulli(prob_mask).to(torch.bool)


class get_model(nn.Module):
    def __init__(self, k=40, model_quantizer=None, max_probability=0.1, knn_k=10):
        super(get_model, self).__init__()
        channel = 3

        self.wq, self.aq, self.uaq = model_quantizer()

        self.knn_k = knn_k
        self.point_score = SemanticModule(self.knn_k)
        self.max_probability = max_probability

        self.qfeat = PointNetQuantEncoder(global_feat=True, feature_transform=True, channel=channel, model_quantizer=model_quantizer)
        self.qfc1 = qnn.QuantLinear(1024, 512, bias=True, input_quant=self.uaq, weight_quant=self.wq, return_quant_tensor=False)
        self.qfc2 = qnn.QuantLinear(512, 256, bias=True, input_quant=self.uaq, weight_quant=self.wq, return_quant_tensor=False)
        self.qfc3 = qnn.QuantLinear(256, k, bias=True, input_quant=self.uaq, weight_quant=self.wq, return_quant_tensor=False)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.training:
            pred_importance = self.point_score(x)
            prob_mask = pred_importance * self.max_probability
        else:
            prob_mask = torch.ones_like(x)
            pred_importance = None
        mask = evaluate_prob_mask(prob_mask)

        x, trans, trans_feat, lable_importance = self.qfeat(x, mask)
        x = F.relu(self.bn1(self.qfc1(x)))
        x = F.relu(self.bn2(self.dropout(self.qfc2(x))))
        x = self.qfc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat, pred_importance, lable_importance


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class get_loss2(torch.nn.Module):
    def __init__(self, loss_weight=0.1):
        super(get_loss2, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, score, degree):
        loss = F.binary_cross_entropy(score, degree)

        total_loss2 = loss * self.loss_weight
        return total_loss2
