import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
logger = logging.getLogger(__name__)
try:
  import apex.normalization.fused_layer_norm.FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
  BertLayerNorm = torch.nn.LayerNorm
from model.transformer.bert import BertEncoder
from model.modeling_utils import mask_logits
import pdb

class LinearLayer(nn.Module):
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True,tanh=False):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.tanh = tanh
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = BertLayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        if self.tanh:
            x = torch.tanh(x)
        return x


class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1 / math.sqrt(feature_size))* torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1 / math.sqrt(feature_size))* torch.randn(1, feature_size, cluster_size))

        self.add_norm = add_norm
        self.LayerNorm = BertLayerNorm(cluster_size)
        self.out_dim = cluster_size * feature_size

    def forward(self, x):
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters)

        if self.add_norm:
            assignment = self.LayerNorm(assignment)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad


class JointSpaceEmbedding(nn.Module):

    def __init__(self, config, input_dim):
        super(JointSpaceEmbedding, self).__init__()
        self.trans_linear = LinearLayer(in_hsz=input_dim, out_hsz=config.hidden_size)
        self.encoder = BertEncoder(config)

    def forward(self, features, feat_mask):
        transformed_features = self.trans_linear(features)

        encoder_outputs = self.encoder(transformed_features, feat_mask)

        sequence_output = encoder_outputs[0]

        return sequence_output



