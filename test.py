from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import time
import sys
import psutil

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


# anchors = [1, 2, 3, 4, 5, 6]
# anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
# print(anchors)
# prediction = torch.Tensor([1, 2, 3, 4, 5, 6])  # , 7, 8, 9
# prediction = prediction.view(1, 2, 3)
# prediction = prediction.transpose(1, 2).contiguous()
# print(prediction)
# res = np.arange(5)
# a, b = np.meshgrid(res, res)
# x_offset = torch.FloatTensor(a).view(-1, 1)
# x_offset = x_offset.cuda()
# print(x_offset)
# a = torch.Tensor([[1], [2]])
# b = torch.Tensor([[0], [0]])
# # a = torch.ones([2, 2, 2])
# # b = torch.ones([2, 2, 2])
# c = torch.cat([a, b], 1).repeat(1, 3).view(-1, 2).unsqueeze(0)
#
# print(c)
# aa = torch.Tensor([[[1, 2, 3, 4],
#       [1, 2, 3, 4],
#       [1, 2, 3, 4]],
#                    [[1, 2, 3, 4],
#                     [1, 2, 3, 4],
#                     [1, 2, 3, 4]]
#                    ])
# b = torch.Tensor([2, 2, 2, 3])
# print(aa*b)

# a = np.array([1, 2, 3, 4])
# a = torch.from_numpy(a).float()
# print(a)
# anchors = [(i, i) for i in a]
# print(np.shape(anchors))


# a = np.array([[[1, 2, 3, 4],
#                [1, 2, 3, 4],
#                [1, 2, 3, 4]],
#               [[10, 20, 3, 4],
#                [10, 20, 3, 4],
#                [10, 20, 30, 4]]])
#
# img = cv2.imread("dog-cycle-car.png")
# img = cv2.resize(img, (416, 416))          #Resize to the input dimension
# img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W X C -> C X H X W
# print(np.shape(img_))
# cv2.imshow('picture', img_)
# cv2.waitKey()
# img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
# img_ = torch.from_numpy(img_).float()     #Convert to float
# img_ = Variable(img_)

# a = np.array([[1, 1, 2, 3, 4, 5, 16, 7, 8],
#               [0, 0, 0, 0, 0, 5, 6, 7, 48],
#               [1, 1, 2, 3, 4, 5, 6, 7, 18],
#               [1, 1, 2, 3, 4, 5, 6, 7, 18]])
# a = torch.from_numpy(a)
# a = a.float()
#
# max_conf, max_conf_score = torch.max(a, 1)
# max_conf = max_conf.float().unsqueeze(1)
# max_conf_score = max_conf_score.float().unsqueeze(1)
# seq = (a[:, :5], max_conf, max_conf_score)
# image_pred = torch.cat(seq, 1)
# non_zero_ind = (torch.nonzero(image_pred[:, 4]))
# image_pred_ = image_pred[non_zero_ind, :].view(-1, 7)
# tensor = image_pred_[:, -1]
# tensor_np = tensor.cpu().numpy()
# unique_np = np.unique(tensor_np)
# unique_tensor = torch.from_numpy(unique_np)
# tensor_res = tensor.new(unique_tensor.shape)
# tensor_res.copy_(unique_tensor)  # tensor([6., 8.])
#
# print(tensor_res)
# print(image_pred)

# a = torch.Tensor([0.2, 0.2, 0.5, 0.5])
# non_zero_ind = torch.nonzero(a[:, 4]).squeeze()
# b2_x1, b2_y1, b2_x2, b2_y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
# b1_x1 = torch.Tensor([2, 5, 5])
# b2_x1 = torch.Tensor([3])
# v = b1_x1, b2_x1
# batch_ind = a.new(a.size(0), 1).fill_(10)
# seq = batch_ind, a
# output = torch.cat(seq, 1)
# output = torch.cat((output, output))
# print(output)
# fp = open('data/coco.names', "r")
# names = fp.read().split("\n")[:-1]
# print(names)
# images = 'E:\PycharmProject\YOLO_pytorch\\image'
# try:
#     imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
# except NotADirectoryError:
#     print('no file')
#     # exit()
# loaded_ims = [cv2.imread(x) for x in imlist]
# print(np.shape(loaded_ims[1]))

# canvas = np.full((9, 9), 128)
# print(canvas)
# aa = torch.max(2+a, 1)
# print(a[:, 0].long())
# print(a[0])
#
# im_dim = 1, 2
# im_dim = torch.Tensor(im_dim).repeat(1, 2)
# im_dim = im_dim.repeat(2, 1)
# scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)
# print(scaling_factor)

# videofile = 'E:\\PycharmProject\\YOLO_pytorch\\video\\MyVideo_1.mp4'  # or path to the video file.
# cap = cv2.VideoCapture(videofile)
# assert cap.isOpened(), 'Cannot capture source'
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imshow('bgr', frame[:, :, 1])
#     cv2.waitKey(1)

a = torch.Tensor([[0, 1, 2, 3, 4, 5, 16, 7, 8],
                  [0, 0, 0, 0, 0, 5, 6, 7, 48],
                  [0, 1, 2, 3, 4, 5, 6, 7, 18],
                  [0, 1, 2, 3, 4, 5, 6, 7, 18]])
a_list = torch.Tensor([[0.2, 0.2, 0.5, 0.5]])
im_batches = [torch.cat((a, a))]
print(im_batches)

