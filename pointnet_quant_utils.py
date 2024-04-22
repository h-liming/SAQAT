import brevitas.nn as qnn
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from quant.common import CommonIntWeightPerChannelQuant


class W8(CommonIntWeightPerChannelQuant):
    bit_width = 8


class QuantSTN3d(nn.Module):
    def __init__(self, channel, model_quantizer=None):
        super(QuantSTN3d, self).__init__()

        self.wq, self.aq, self.uaq = model_quantizer()

        self.output_quant_1 = qnn.QuantIdentity(act_quant=self.uaq)
        self.output_no_quant_1 = qnn.QuantIdentity(act_quant=None)
        self.output_quant_2 = qnn.QuantIdentity(act_quant=self.uaq)
        self.output_no_quant_2 = qnn.QuantIdentity(act_quant=None)
        self.output_quant_3 = qnn.QuantIdentity(act_quant=self.uaq)
        self.output_no_quant_3 = qnn.QuantIdentity(act_quant=None)

        self.qconv1 = qnn.QuantConv1d(channel, 64, 1, weight_quant=W8)
        self.qconv2 = qnn.QuantConv1d(64, 128, 1, weight_quant=self.wq)
        self.qconv3 = qnn.QuantConv1d(128, 1024, 1, weight_quant=self.wq)
        self.qfc1 = qnn.QuantLinear(1024, 512, bias=True, weight_quant=self.wq)
        self.qfc2 = qnn.QuantLinear(512, 256, bias=True, input_quant=self.uaq, weight_quant=self.wq)
        self.qfc3 = qnn.QuantLinear(256, 9, bias=True, input_quant=self.uaq, weight_quant=self.wq)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, mask):
        batch_size, channel, num_points = x.size()

        x = self.qconv1(x)
        x = F.relu(self.bn1(x))

        if self.training:
            feature_mask = mask.view(batch_size, 1, num_points).repeat(1, 64, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_1(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_1(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_1(x)  # low precision

        x = self.qconv2(x_q)
        x = F.relu(self.bn2(x))

        if self.training:
            feature_mask = mask.view(batch_size, 1, num_points).repeat(1, 128, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_2(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_2(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_2(x)  # low precision

        x = self.qconv3(x_q)
        x = F.relu(self.bn3(x))

        if self.training:
            feature_mask = mask.view(batch_size, 1, num_points).repeat(1, 1024, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_3(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_3(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_3(x)  # low precision

        x = torch.max(x_q, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.qfc1(x)))
        x = F.relu(self.bn5(self.qfc2(x)))
        x = self.qfc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class QuantSTNkd(nn.Module):
    def __init__(self, k=64, model_quantizer=None):
        super(QuantSTNkd, self).__init__()

        self.wq, self.aq, self.uaq = model_quantizer()

        self.output_quant_1 = qnn.QuantIdentity(act_quant=self.uaq)
        self.output_no_quant_1 = qnn.QuantIdentity(act_quant=None)
        self.output_quant_2 = qnn.QuantIdentity(act_quant=self.uaq)
        self.output_no_quant_2 = qnn.QuantIdentity(act_quant=None)
        self.output_quant_3 = qnn.QuantIdentity(act_quant=self.uaq)
        self.output_no_quant_3 = qnn.QuantIdentity(act_quant=None)

        self.qconv1 = qnn.QuantConv1d(k, 64, 1, weight_quant=self.wq)
        self.qconv2 = qnn.QuantConv1d(64, 128, 1, weight_quant=self.wq)
        self.qconv3 = qnn.QuantConv1d(128, 1024, 1, weight_quant=self.wq)
        self.qfc1 = qnn.QuantLinear(1024, 512, bias=True, weight_quant=self.wq)
        self.qfc2 = qnn.QuantLinear(512, 256, bias=True, input_quant=self.uaq, weight_quant=self.wq)
        self.qfc3 = qnn.QuantLinear(256, k * k, bias=True, input_quant=self.uaq, weight_quant=self.wq)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x, mask):
        batch_size, channel, num_points = x.size()

        x = self.qconv1(x)
        x = F.relu(self.bn1(x))

        if self.training:
            feature_mask = mask.view(batch_size, 1, num_points).repeat(1, 64, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_1(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_1(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_1(x)  # low precision

        x = self.qconv2(x_q)
        x = F.relu(self.bn2(x))

        if self.training:
            feature_mask = mask.view(batch_size, 1, num_points).repeat(1, 128, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_2(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_2(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_2(x)  # low precision

        x = self.qconv3(x_q)
        x = F.relu(self.bn3(x))

        if self.training:
            feature_mask = mask.view(batch_size, 1, num_points).repeat(1, 1024, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_3(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_3(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_3(x)  # low precision

        x = torch.max(x_q, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.qfc1(x)))
        x = F.relu(self.bn5(self.qfc2(x)))
        x = self.qfc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetQuantEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3, model_quantizer=None):
        super(PointNetQuantEncoder, self).__init__()

        self.wq, self.aq, self.uaq = model_quantizer()

        self.input_quant = qnn.QuantIdentity()
        self.input_no_quant = qnn.QuantIdentity(act_quant=None)

        self.output_quant_1 = qnn.QuantIdentity(act_quant=self.uaq, )
        self.output_no_quant_1 = qnn.QuantIdentity(act_quant=None)
        self.output_quant_2 = qnn.QuantIdentity(act_quant=self.uaq)
        self.output_no_quant_2 = qnn.QuantIdentity(act_quant=None)

        self.qstn = QuantSTN3d(channel, model_quantizer)
        self.qconv1 = qnn.QuantConv1d(channel, 64, 1, weight_quant=W8)
        self.qconv2 = qnn.QuantConv1d(64, 128, 1, weight_quant=self.wq)
        self.qconv3 = qnn.QuantConv1d(128, 1024, 1, weight_quant=self.wq)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.qfstn = QuantSTNkd(k=64, model_quantizer=model_quantizer)

    def forward(self, x, mask):
        B, D, N = x.size()
        if self.training:
            feature_mask = mask.view(B, 1, N).repeat(1, D, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.input_no_quant(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.input_quant(x[~feature_mask])  # low precision
        else:
            x_q = self.input_quant(x)  # low precision
        trans = self.qstn(x_q, mask)

        x = x_q.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.qconv1(x)))

        if self.training:
            feature_mask = mask.view(B, 1, N).repeat(1, 64, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_1(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_1(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_1(x)  # low precision

        if self.feature_transform:
            trans_feat = self.qfstn(x_q, mask)
            x = x_q.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.qconv2(x)))

        if self.training:
            feature_mask = mask.view(B, 1, N).repeat(1, 128, 1)
            x_q = torch.empty_like(x)
            x_q[feature_mask] = self.output_no_quant_2(x[feature_mask])  # high precision
            x_q[~feature_mask] = self.output_quant_2(x[~feature_mask])  # low precision
        else:
            x_q = self.output_quant_2(x)  # low precision

        x = self.bn3(self.qconv3(x_q))

        '''↓↓↓ Point Importance Evaluation Algorithm (PIEA) ↓↓↓'''
        idx = torch.argmax(x, 2, keepdim=False)

        degree = torch.zeros(B, N, dtype=idx.dtype, device=x.device)
        values = torch.ones_like(idx)
        degree.scatter_add_(1, idx, values)

        counts = torch.zeros(B, x.shape[1] + 1, dtype=idx.dtype, device=x.device)
        values = torch.ones_like(degree)
        counts.scatter_add_(1, degree, values)

        no_zero = N - counts[:, 0]
        step_size = 1.0 / no_zero.view(B, 1)

        ps = counts * step_size
        ps[:, 0] = 0
        ps = torch.cumsum(ps, dim=1)
        torch.cuda.synchronize()
        batch_index = torch.arange(B, dtype=torch.long).to(x.device).view(B, 1).repeat(1, N)
        lable_importance = ps[batch_index, degree]
        '''↑↑↑ Point Importance Evaluation Algorithm (PIEA) ↑↑↑'''

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat, lable_importance
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
