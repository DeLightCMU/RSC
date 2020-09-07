import os
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn as nn

from models.alexnet import Id
from models.model_utils import ReverseLayerF
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random


class AlexNetCaffe(nn.Module):
    def __init__(self, jigsaw_classes=1000, n_classes=100, domains=3, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        self.jigsaw_classifier = nn.Linear(4096, jigsaw_classes)
        self.class_classifier = nn.Linear(4096, n_classes)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(256 * 6 * 6, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, domains))

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.jigsaw_classifier.parameters()
                                 , self.class_classifier.parameters()#, self.domain_classifier.parameters()
                                 ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x = self.features(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        #d = ReverseLayerF.apply(x, lambda_val)
        x = self.classifier(x)
        return self.jigsaw_classifier(x), self.class_classifier(x)#, self.domain_classifier(d)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AlexNetCaffeAvgPool(AlexNetCaffe):
    def __init__(self, jigsaw_classes=1000, n_classes=100):
        super().__init__()
        print("Global Average Pool variant")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            #             ("relu5", nn.ReLU(inplace=True)),
            #             ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))

        self.jigsaw_classifier = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(128 * 6 * 6, jigsaw_classes)
        )
        self.class_classifier = nn.Sequential(
            nn.Conv2d(1024, n_classes, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(13),
            Flatten(),
            # nn.Linear(1024, n_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AlexNetCaffeFC7(AlexNetCaffe):
    def __init__(self, jigsaw_classes=1000, n_classes=100, dropout=True):
        super(AlexNetCaffeFC7, self).__init__()
        print("FC7 branching variant")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id())]))

        self.jigsaw_classifier = nn.Sequential(OrderedDict([
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout()),
            ("fc8", nn.Linear(4096, jigsaw_classes))]))
        self.class_classifier = nn.Sequential(OrderedDict([
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout()),
            ("fc8", nn.Linear(4096, n_classes))]))


def caffenet(jigsaw_classes, classes):
    model = AlexNetCaffe(jigsaw_classes, classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)

    return model


def caffenet_gap(jigsaw_classes, classes):
    model = AlexNetCaffe(jigsaw_classes, classes)
    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc6.weight"]
    del state_dict["classifier.fc6.bias"]
    del state_dict["classifier.fc7.weight"]
    del state_dict["classifier.fc7.bias"]
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)
    # weights are initialized in the constructor
    return model


def caffenet_fc7(jigsaw_classes, classes):
    model = AlexNetCaffeFC7(jigsaw_classes, classes)
    state_dict = torch.load("models/pretrained/alexnet_caffe.pth.tar")
    state_dict["jigsaw_classifier.fc7.weight"] = state_dict["classifier.fc7.weight"]
    state_dict["jigsaw_classifier.fc7.bias"] = state_dict["classifier.fc7.bias"]
    state_dict["class_classifier.fc7.weight"] = state_dict["classifier.fc7.weight"]
    state_dict["class_classifier.fc7.bias"] = state_dict["classifier.fc7.bias"]
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    del state_dict["classifier.fc7.weight"]
    del state_dict["classifier.fc7.bias"]
    model.load_state_dict(state_dict, strict=False)
    nn.init.xavier_uniform_(model.jigsaw_classifier.fc8.weight, .1)
    nn.init.constant_(model.jigsaw_classifier.fc8.bias, 0.)
    nn.init.xavier_uniform_(model.class_classifier.fc8.weight, .1)
    nn.init.constant_(model.class_classifier.fc8.bias, 0.)
    return model


class AlexNetCaffeRSC(nn.Module):
    def __init__(self, n_classes=100, percent=6, dropout=True):
        super(AlexNetCaffeRSC, self).__init__()
        print("Using Caffe AlexNet")
        self.percent = percent
        print("Using Total Percent Sample: 1 / {}".format(self.percent))
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        # self.jigsaw_classifier = nn.Linear(4096, jigsaw_classes)
        self.class_classifier = nn.Linear(4096, n_classes)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(256 * 6 * 6, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, domains))

    # def get_params(self, base_lr):
    #     return [{"params": self.features.parameters(), "lr": 0.},
    #             {"params": chain(self.classifier.parameters()
    #                              , self.class_classifier.parameters()#, self.domain_classifier.parameters()
    #                              ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, flag=None):
        # x = self.features(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        # x = x.view(x.size(0), -1)
        # #d = ReverseLayerF.apply(x, lambda_val)
        # x = self.classifier(x)
        # return self.class_classifier(x)#, self.domain_classifier(d)
        # -------------------------------------------------------------------
        x = self.features(x * 57.6)
        # x = self.features.conv1(x * 57.6)
        # x = self.features.relu1(x)
        # x = self.features.pool1(x)
        # x = self.features.norm1(x)
        # x = self.features.conv2(x)
        # x = self.features.relu2(x)
        # x = self.features.pool2(x)
        # x = self.features.norm2(x)
        # x = self.features.conv3(x)
        # x = self.features.relu3(x)
        # x = self.features.conv4(x)
        # x = self.features.relu4(x)
        # x = self.features.conv5(x)
        # x = self.features.relu5(x)
        # x = self.features.pool5(x)

        if flag:
            self.eval()
            x_new = x.clone().detach()

            # x_new = self.features.conv4(x_new)
            # x_new = self.features.relu4(x_new)
            # x_new = self.features.conv5(x_new)
            # x_new = self.features.relu5(x_new)
            # x_new = self.features.pool5(x_new)

            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = x_new.view(x_new.size(0), -1)
            x_new_view = self.classifier(x_new_view)
            output = self.class_classifier(x_new_view)
            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)  # [256, 21]
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            spatial_mean = torch.mean(grads_val, dim=1)
            spatial_mean = spatial_mean.view(num_rois, H, H).view(num_rois, HW)
            self.zero_grad()

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = int(HW * 1 / 3.0)
                th_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                th_mask_value = th_mask_value.view(num_rois, 1).expand(num_rois, HW)
                mask_all_cuda = torch.where(spatial_mean >= th_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all_cuda.detach().cpu().numpy()
                for q in range(num_rois):
                    mask_all_temp = np.ones((HW), dtype=np.float32)
                    zero_index = np.where(mask_all[q, :] == 0)[0]
                    num_zero_index = zero_index.size
                    if num_zero_index >= spatial_drop_num:
                        dumy_index = npr.choice(zero_index, size=spatial_drop_num, replace=False)
                    else:
                        zero_index = np.arange(HW)
                        dumy_index = npr.choice(zero_index, size=spatial_drop_num, replace=False)
                    mask_all_temp[dumy_index] = 0
                    mask_all[q, :] = mask_all_temp
                mask_all = torch.from_numpy(mask_all.reshape(num_rois, 7, 7)).cuda()
                mask_all = mask_all.view(num_rois, 1, 7, 7)
            else:
                # -------------------------- channel ----------------------------
                mask_all = torch.zeros((num_rois, num_channel, 1, 1)).cuda()
                vector_thresh_percent = int(num_channel * 1 / 3.0)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                vector_all = vector.detach().cpu().numpy()
                channel_drop_num = int(num_channel * 1 / 3.2)
                vector_all_new = np.ones((num_rois, num_channel), dtype=np.float32)
                for q in range(num_rois):
                    vector_all_temp = np.ones((num_channel), dtype=np.float32)
                    zero_index = np.where(vector_all[q, :] == 0)[0]
                    num_zero_index = zero_index.size
                    if num_zero_index >= channel_drop_num:
                        dumy_index = npr.choice(zero_index, size=channel_drop_num, replace=False)
                    else:
                        zero_index = np.arange(num_channel)
                        dumy_index = npr.choice(zero_index, size=channel_drop_num, replace=False)
                    vector_all_temp[dumy_index] = 0
                    vector_all_new[q, :] = vector_all_temp
                vector = torch.from_numpy(vector_all_new).cuda()
                for m in range(num_rois):
                    index_channel = vector[m, :].nonzero()[:, 0].long()
                    index_channel = index_channel.detach().cpu().numpy().tolist()
                    mask_all[m, index_channel, :, :] = 1

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.classifier(x_new_view_after)
            x_new_view_after = self.class_classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * 1 / 3.0))]
            drop_index_fg = change_vector.gt(th_fg_value)
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x)  # , self.domain_classifier(d)

def caffenetRSC(classes, percent):
    model = AlexNetCaffeRSC(classes, percent)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)

    return model
