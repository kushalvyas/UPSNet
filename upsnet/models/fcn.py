# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from upsnet.config.config import config
from upsnet.operators.modules.deform_conv import DeformConv, DeformConvWithOffset
from upsnet.operators.modules.roialign import RoIAlign

if config.train.use_horovod and config.network.use_syncbn:
    from upsnet.operators.modules.distbatchnorm import BatchNorm2d


class FCNSubNet(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, deformable_group=1, dilation=1, with_norm='none'):
        super(FCNSubNet, self).__init__()

        assert with_norm in ['none', 'batch_norm', 'group_norm']
        assert num_layers >= 2
        self.num_layers = num_layers
        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(in_channel):
                return nn.GroupNorm(32, in_channel)
            norm = group_norm
        else:
            norm = None
        self.conv = nn.ModuleList()
        for i in range(num_layers):
            conv = []
            if i == num_layers - 2:
                conv.append(DeformConvWithOffset(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation))
                in_channels = out_channels
            else:
                conv.append(DeformConvWithOffset(in_channels, in_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation))
            if with_norm != 'none':
                conv.append(norm(in_channels))
            conv.append(nn.ReLU(inplace=True))
            self.conv.append(nn.Sequential(*conv))

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)
            elif isinstance(m, DeformConv):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv[i](x)
        return x


class FCNHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_layers, with_norm='none', with_roi_loss=False, upsample_rate=4):
        super(FCNHead, self).__init__()
        self.fcn_subnet = FCNSubNet(in_channels, 128, num_layers, with_norm=with_norm)
        self.upsample_rate = upsample_rate

        self.score = nn.Conv2d(512, num_classes, 1)
        if with_roi_loss:
            self.roipool = RoIAlign(config.network.mask_size, config.network.mask_size, 1/4.0)
        self.initialize()

    def forward(self, fpn_p2, fpn_p3, fpn_p4, fpn_p5, roi=None):
        fpn_p2 = self.fcn_subnet(fpn_p2)
        fpn_p3 = self.fcn_subnet(fpn_p3)
        fpn_p4 = self.fcn_subnet(fpn_p4)
        fpn_p5 = self.fcn_subnet(fpn_p5)

        fpn_p3 = F.interpolate(fpn_p3, None, 2, mode='bilinear', align_corners=False)
        fpn_p4 = F.interpolate(fpn_p4, None, 4, mode='bilinear', align_corners=False)
        fpn_p5 = F.interpolate(fpn_p5, None, 8, mode='bilinear', align_corners=False)
        feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)
        score = self.score(feat)
        ret = {'fcn_score': score, 'fcn_feat': feat}
        if self.upsample_rate != 1:
            output = F.interpolate(score, None, self.upsample_rate, mode='bilinear', align_corners=False)
            ret.update({'fcn_output': output})
        if roi is not None:
            roi_feat = self.roipool(feat, roi)
            roi_score = self.score(roi_feat)
            ret.update({'fcn_roi_score': roi_score})

        return ret


    def initialize(self):
        nn.init.normal_(self.score.weight.data, 0, 0.01)
        self.score.bias.data.zero_()

import torch.nn.functional as F
class DistillationLoss(nn.Module):
    def __init__(self, size_average=True, ignore_label=255, lambda_val=10):
        super(DistillationLoss, self).__init__()
        self.lambda = lambda_val
        self.size_average = size_average
        self.ignore_label = ignore_label
        # self.kl = nn.KLDivLoss(size_average=True)
        self.cs = nn.CosineSimilarity(dim=1)
        # self.mse = nn.MSELoss(size_average=False, reduce=True)


    def forward(self, student_dict, teacher_dict, target):
        student_feat = student_dict['fcn_feat'].detach()
        student_score = student_dict['fcn_score'].detach()
        teacher_feat = teacher_dict['fcn_feat'].detach()
        teacher_score = teacher_dict['fcn_score'].detach()

        # scores KL divergence
        student_score_flat = student_score.view(student_score.shape[0], -1)
        teacher_score_flat = teacher_score.view(teacher_score.shape[0], -1)
        score_kl = F.kl_div(student_score_flat, teacher_score_flat, size_average=True)

        # pairwise distillation loss
        W = student_feat.shape[3]
        H = student_feat.shape[2]

        cs_feat_student = self.cs(student_feat, student_feat)
        cs_feat_teacher = self.cs(teacher_feat, teacher_feat)

        cs_feat_student_flat = cs_feat_student.view(cs_feat_student.shape[0], -1)
        cs_feat_teacher_flat = cs_feat_teacher.view(cs_feat_teacher.shape[0], -1)


        l2_loss_feat = (1/(W*H)**2) * F.mse_loss(cs_feat_teacher_flat, cs_feat_student_flat, size_average=False, reduce=True)

        cross_entropy_loss = F.cross_entropy(student_score, target, size_average=self.size_average, ignore_index=self.ignore_label)

        loss = cross_entropy_loss + self.lambda*(l2_loss_feat + score_kl)

        return loss








class CrossEntropyLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        n, c, h, w = predict.size()

        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average, ignore_index=self.ignore_label)
        return loss
