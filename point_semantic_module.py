import torch
import torch.nn as nn
from pytorch3d.ops import knn_points


def get_graph_feature(x, k=64, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    xyz = x.permute(0, 2, 1)
    if idx is None:
        _, idx, _ = knn_points(xyz, xyz, K=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((torch.square(feature - x), feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 3*num_dims, num_points, k)


class SemanticModule(nn.Module):
    def __init__(self, k=20):
        super(SemanticModule, self).__init__()
        self.k = k
        self.in_channel = 9
        self.c1_channel = 64
        self.c2_channel = 256
        self.c3_channel = 128

        self.bn1 = nn.BatchNorm2d(self.c1_channel)
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.c1_channel, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.bn2 = nn.BatchNorm1d(self.c2_channel)
        self.conv2 = nn.Sequential(nn.Conv1d(self.c1_channel, self.c2_channel, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.bn3 = nn.BatchNorm1d(self.c3_channel)
        self.conv3 = nn.Sequential(nn.Conv1d(self.c2_channel, self.c3_channel, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(self.c3_channel, 1, kernel_size=1, bias=True)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*3, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*3, num_points, k) -> (batch_size, conv1_channel, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, conv1_channel, num_points, k) -> (batch_size, conv1_channel, num_points)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).squeeze()
        pred_importance = self.sigmoid(x)

        return pred_importance  # mask:[batch_size,num_points], pred_importance:[batch_size,num_points]
