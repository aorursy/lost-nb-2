#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import random


def flip(img):
  return img[:, :, ::-1].copy()

# todo what the hell is this?
def get_border(border, size):
  i = 1
  while size - border // i <= border // i:
    i *= 2
  return border // i

def transform_preds(coords, center, scale, output_size):
  target_coords = np.zeros(coords.shape)
  trans = get_affine_transform(center, scale, 0, output_size, inv=1)
  for p in range(coords.shape[0]):
    target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
  return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
  if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
    scale = np.array([scale, scale], dtype=np.float32)

  scale_tmp = scale
  src_w = scale_tmp[0]
  dst_w = output_size[0]
  dst_h = output_size[1]

  rot_rad = np.pi * rot / 180
  src_dir = get_dir([0, src_w * -0.5], rot_rad)
  dst_dir = np.array([0, dst_w * -0.5], np.float32)

  src = np.zeros((3, 2), dtype=np.float32)
  dst = np.zeros((3, 2), dtype=np.float32)
  src[0, :] = center + scale_tmp * shift
  src[1, :] = center + src_dir + scale_tmp * shift
  dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
  dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

  src[2:, :] = get_3rd_point(src[0, :], src[1, :])
  dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

  if inv:
    trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
  else:
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

  return trans


def affine_transform(pt, t):
  new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
  new_pt = np.dot(t, new_pt)
  return new_pt[:2]


def get_3rd_point(a, b):
  direct = a - b
  return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
  _sin, _cos = np.sin(rot_rad), np.cos(rot_rad)

  src_result = [0, 0]
  src_result[0] = src_point[0] * _cos - src_point[1] * _sin
  src_result[1] = src_point[0] * _sin + src_point[1] * _cos

  return src_result


def crop(img, center, scale, output_size, rot=0):
  trans = get_affine_transform(center, scale, rot, output_size)

  dst_img = cv2.warpAffine(img,
                           trans,
                           (int(output_size[0]), int(output_size[1])),
                           flags=cv2.INTER_LINEAR)

  return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1 = 1
  b1 = (height + width)
  c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  # r1 = (b1 + sq1) / 2 #
  r1 = (b1 - sq1) / (2 * a1)

  a2 = 4
  b2 = 2 * (height + width)
  c2 = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  # r2 = (b2 + sq2) / 2
  r2 = (b2 - sq2) / (2 * a2)

  a3 = 4 * min_overlap
  b3 = -2 * min_overlap * (height + width)
  c3 = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3 = (b3 + sq3) / 2
  # r3 = (b3 + sq3) / (2 * a3)
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]

  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter * 2 + 1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                    radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
               radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap


def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
  alpha = data_rng.normal(scale=alphastd, size=(3,))
  image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
  image1 *= alpha
  image2 *= (1 - alpha)
  image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
  functions = [brightness_, contrast_, saturation_]
  random.shuffle(functions)

  gs = grayscale(image)
  gs_mean = gs.mean()
  for f in functions:
    f(data_rng, image, gs, gs_mean, 0.4)
  lighting_(data_rng, image, 0.1, eig_val, eig_vec)


# In[2]:


# import torch
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo

# BN_MOMENTUM = 0.1

# model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#               'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#               'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#               'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#               'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class PoseResNet(nn.Module):
#     def __init__(self, block, layers, head_conv, num_classes):
#         super(PoseResNet, self).__init__()
#         self.inplanes = 64
#         self.deconv_with_bias = False
#         self.num_classes = num_classes

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # used for deconv layers
#         self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
#         # self.final_layer = []

#         if head_conv > 0:
#             # heatmap layers
#             self.hmap = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, num_classes, kernel_size=1))
# #             self.hmap[-1].bias.data.fill_(-2.19)
#             # regression layers
#             self.regs = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, 2, kernel_size=1))
#             self.w_h_ = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, 2, kernel_size=1))
#         else:
#             # heatmap layers
#             self.hmap = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
#             # regression layers
#             self.regs = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
#             self.w_h_ = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

#         # self.final_layer = nn.ModuleList(self.final_layer)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
#                                                  kernel_size=1, stride=stride, bias=False),
#                                        nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)

#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0

#         return deconv_kernel, padding, output_padding

#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'

#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

#             planes = num_filters[i]
#             layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
#                                              out_channels=planes,
#                                              kernel_size=kernel,
#                                              stride=2,
#                                              padding=padding,
#                                              output_padding=output_padding,
#                                              bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.deconv_layers(x)
#         out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
#         return out

#     def init_weights(self, num_layers):
#         for m in self.deconv_layers.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.hmap.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.constant_(m.bias, -2.19)
#         for m in self.regs.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.w_h_.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)
#         url = model_urls['resnet{}'.format(num_layers)]
#         pretrained_state_dict = model_zoo.load_url(url)
#         print('=> loading pretrained model {}'.format(url))
#         self.load_state_dict(pretrained_state_dict, strict=False)


# resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
#                34: (BasicBlock, [3, 4, 6, 3]),
#                50: (Bottleneck, [3, 4, 6, 3]),
#                101: (Bottleneck, [3, 4, 23, 3]),
#                152: (Bottleneck, [3, 8, 36, 3])}

# def resnet_18():
#     model = PoseResNet(BasicBlock, [2, 2, 2, 2], head_conv=64, num_classes=80)
#     model.init_weights(18)
#     return model

# def get_pose_net(num_layers, head_conv, num_classes=80):
#     block_class, layers = resnet_spec[num_layers]

#     model = PoseResNet(block_class, layers, head_conv=head_conv, num_classes=num_classes)
# #     model.init_weights(num_layers)
#     return model


# In[3]:


# import torch
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo

# BN_MOMENTUM = 0.1

# model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#               'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#               'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#               'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#               'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class PoseResNet(nn.Module):
#     def __init__(self, block, layers, head_conv, num_classes, verbose = True):
#         super(PoseResNet, self).__init__()
#         self.inplanes = 64
#         self.deconv_with_bias = False
#         self.num_classes = num_classes
#         self.verbose = verbose

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # used for deconv layers
#         self.deconv_layer16 = self._make_deconv_layer(3, [128, 128, 128], [4, 4, 4],2048)
#         self.deconv_layer32 = self._make_deconv_layer(2, [64,64], [4, 4],1024)
#         self.deconv_layer64 = self._make_deconv_layer(1, [64], [4],512)
#         # self.final_layer = []

#         if head_conv > 0:
#             # heatmap layers
#             self.hmap = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, num_classes, kernel_size=1))
# #             self.hmap[-1].bias.data.fill_(-2.19)
#             # regression layers
#             self.regs = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, 2, kernel_size=1))
#             self.w_h_ = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, 2, kernel_size=1))
#         else:
#             # heatmap layers
#             self.hmap = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
#             # regression layers
#             self.regs = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
#             self.w_h_ = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

#         # self.final_layer = nn.ModuleList(self.final_layer)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
#                                                  kernel_size=1, stride=stride, bias=False),
#                                        nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)

#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0

#         return deconv_kernel, padding, output_padding

#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels, inplanes):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        
#         if self.verbose:
#             print("_make_deconv_layer inplanes: {}".format(inplanes))

#         layers = []
        
#         for i in range(num_layers):
#             kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

#             planes = num_filters[i]
#             layers.append(nn.ConvTranspose2d(in_channels=inplanes,
#                                              out_channels=planes,
#                                              kernel_size=kernel,
#                                              stride=2,
#                                              padding=padding,
#                                              output_padding=output_padding,
#                                              bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             inplanes = planes

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         if self.verbose:
#             print("x shape: {}".format(x.size()))

#         x = self.layer1(x)
#         if self.verbose:
#             print("x shape: {}".format(x.size()))
#         x64 = self.layer2(x)
#         if self.verbose:
#             print("x64 shape: {}".format(x64.size()))
#         x32 = self.layer3(x64)
#         if self.verbose:
#             print("x32 shape: {}".format(x32.size()))
#         x16 = self.layer4(x32)
#         if self.verbose:
#             print("x16 shape: {}".format(x16.size()))
#         # conbine deconv x from diff layer
#         x16 = self.deconv_layer16(x16)
#         x32 = self.deconv_layer32(x32)
#         x64 = self.deconv_layer64(x64)
        
#         x = torch.cat([x16,x32,x64],1)
        
#         if self.verbose:
#             print("x shape: {}".format(x.size()))
#         out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
#         return out

#     def init_weights(self, num_layers):
#         for m in self.deconv_layer16.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
#         for m in self.deconv_layer32.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.deconv_layer64.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
#         for m in self.hmap.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.constant_(m.bias, -2.19)
#         for m in self.regs.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.w_h_.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)
#         url = model_urls['resnet{}'.format(num_layers)]
#         pretrained_state_dict = model_zoo.load_url(url)
#         print('=> loading pretrained model {}'.format(url))
#         self.load_state_dict(pretrained_state_dict, strict=False)


# resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
#                34: (BasicBlock, [3, 4, 6, 3]),
#                50: (Bottleneck, [3, 4, 6, 3]),
#                101: (Bottleneck, [3, 4, 23, 3]),
#                152: (Bottleneck, [3, 8, 36, 3])}

# def resnet_18():
#     model = PoseResNet(BasicBlock, [2, 2, 2, 2], head_conv=64, num_classes=80,verbose=False)
#     model.init_weights(18)
#     return model

# def get_pose_net(num_layers, head_conv, num_classes=80):
#     block_class, layers = resnet_spec[num_layers]

#     model = PoseResNet(block_class, layers, head_conv=head_conv, num_classes=num_classes, verbose= False)
# #     model.init_weights(num_layers)
#     return model


# In[4]:


# import torch
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo

# BN_MOMENTUM = 0.1

# model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#               'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#               'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#               'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#               'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class PoseResNet(nn.Module):
#     def __init__(self, block, layers, head_conv, num_classes, verbose = True):
#         super(PoseResNet, self).__init__()
#         self.inplanes = 64
#         self.deconv_with_bias = False
#         self.num_classes = num_classes
#         self.verbose = verbose

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # used for deconv layers
#         self.deconv_layer16 = self._make_deconv_layer(1, [1024], [4],2048)
#         self.deconv_layer32 = self._make_deconv_layer(1, [512], [4],2048)
#         self.deconv_layer64 = self._make_deconv_layer(1, [256], [4],1024)
#         # self.final_layer = []

#         if head_conv > 0:
#             # heatmap layers
#             self.hmap = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, num_classes, kernel_size=1))
# #             self.hmap[-1].bias.data.fill_(-2.19)
#             # regression layers
#             self.regs = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, 2, kernel_size=1))
#             self.w_h_ = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(head_conv, 2, kernel_size=1))
#         else:
#             # heatmap layers
#             self.hmap = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
#             # regression layers
#             self.regs = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
#             self.w_h_ = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

#         # self.final_layer = nn.ModuleList(self.final_layer)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
#                                                  kernel_size=1, stride=stride, bias=False),
#                                        nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)

#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0

#         return deconv_kernel, padding, output_padding

#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels, inplanes):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        
#         if self.verbose:
#             print("_make_deconv_layer inplanes: {}".format(inplanes))

#         layers = []
        
#         for i in range(num_layers):
#             kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

#             planes = num_filters[i]
#             layers.append(nn.ConvTranspose2d(in_channels=inplanes,
#                                              out_channels=planes,
#                                              kernel_size=kernel,
#                                              stride=2,
#                                              padding=padding,
#                                              output_padding=output_padding,
#                                              bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             inplanes = planes

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         if self.verbose:
#             print("x shape: {}".format(x.size()))

#         x = self.layer1(x)
#         if self.verbose:
#             print("x shape: {}".format(x.size()))
#         x64 = self.layer2(x)
#         if self.verbose:
#             print("x64 shape: {}".format(x64.size()))
#         x32 = self.layer3(x64)
#         if self.verbose:
#             print("x32 shape: {}".format(x32.size()))
#         x16 = self.layer4(x32)
#         if self.verbose:
#             print("x16 shape: {}".format(x16.size()))
#         # conbine deconv x from diff layer
# #         x16 = self.deconv_layer16(x16)
# #         x32 = self.deconv_layer32(x32)
# #         x64 = self.deconv_layer64(x64)
#         up1 = self.deconv_layer16(x16)
#         if self.verbose:
#             print("up1 shape: {}".format(up1.size()))
#         up1 = torch.cat([up1,x32],1)
#         if self.verbose:
#             print("up1 shape: {}".format(up1.size()))
#         up2 = self.deconv_layer32(up1)
#         if self.verbose:
#             print("up2 shape: {}".format(up2.size()))
#         up2 = torch.cat([up2,x64],1)
#         if self.verbose:
#             print("up2 shape: {}".format(up2.size()))
#         x = self.deconv_layer64(up2)
        
# #         x = torch.cat([x16,x32,x64],1)
        
#         if self.verbose:
#             print("x shape: {}".format(x.size()))
#         out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
#         return out

#     def init_weights(self, num_layers):
#         for m in self.deconv_layer16.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
#         for m in self.deconv_layer32.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.deconv_layer64.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
#         for m in self.hmap.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.constant_(m.bias, -2.19)
#         for m in self.regs.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.w_h_.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)
#         url = model_urls['resnet{}'.format(num_layers)]
#         pretrained_state_dict = model_zoo.load_url(url)
#         print('=> loading pretrained model {}'.format(url))
#         self.load_state_dict(pretrained_state_dict, strict=False)


# resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
#                34: (BasicBlock, [3, 4, 6, 3]),
#                50: (Bottleneck, [3, 4, 6, 3]),
#                101: (Bottleneck, [3, 4, 23, 3]),
#                152: (Bottleneck, [3, 8, 36, 3])}

# def resnet_18():
#     model = PoseResNet(BasicBlock, [2, 2, 2, 2], head_conv=64, num_classes=80,verbose=False)
#     model.init_weights(18)
#     return model

# def get_pose_net(num_layers, head_conv, num_classes=80,verbose = False):
#     block_class, layers = resnet_spec[num_layers]

#     model = PoseResNet(block_class, layers, head_conv=head_conv, num_classes=num_classes, verbose= verbose)
# #     model.init_weights(num_layers)
#     return model


# In[5]:


import numpy as np
import torch
import torch.nn as nn


class convolution(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(convolution, self).__init__()
    pad = (k - 1) // 2
    self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
    self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv = self.conv(x)
    bn = self.bn(conv)
    relu = self.relu(bn)
    return relu


class residual(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(residual, self).__init__()

    self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)

    self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                              nn.BatchNorm2d(out_dim)) \
      if stride != 1 or inp_dim != out_dim else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv1 = self.conv1(x)
    bn1 = self.bn1(conv1)
    relu1 = self.relu1(bn1)

    conv2 = self.conv2(relu1)
    bn2 = self.bn2(conv2)

    skip = self.skip(x)
    return self.relu(bn2 + skip)


# inp_dim -> out_dim -> ... -> out_dim
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
  layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
  layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
  return nn.Sequential(*layers)


# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
  layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
  layers.append(layer(kernel_size, inp_dim, out_dim))
  return nn.Sequential(*layers)


# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
  return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))


class kp_module(nn.Module):
  def __init__(self, n, dims, modules):
    super(kp_module, self).__init__()

    self.n = n

    curr_modules = modules[0]
    next_modules = modules[1]

    curr_dim = dims[0]
    next_dim = dims[1]

    # curr_mod x residual，curr_dim -> curr_dim -> ... -> curr_dim
    self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
    self.down = nn.Sequential()
    # curr_mod x residual，curr_dim -> next_dim -> ... -> next_dim
    self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2)
    # next_mod x residual，next_dim -> next_dim -> ... -> next_dim
    if self.n > 1:
      self.low2 = kp_module(n - 1, dims[1:], modules[1:])
    else:
      self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
    # curr_mod x residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
    self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
    self.up = nn.Upsample(scale_factor=2)

  def forward(self, x):
    up1 = self.top(x)
    down = self.down(x)
    low1 = self.low1(down)
    low2 = self.low2(low1)
    low3 = self.low3(low2)
    up2 = self.up(low3)
    return up1 + up2


class exkp(nn.Module):
  def __init__(self, n, nstack, dims, modules, cnv_dim=256, num_classes=1):
    super(exkp, self).__init__()

    self.nstack = nstack
    self.num_classes = num_classes

    curr_dim = dims[0]

    self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))

    self.kps = nn.ModuleList([kp_module(n, dims, modules) for _ in range(nstack)])

    self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])

    self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])

    self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                nn.BatchNorm2d(curr_dim))
                                  for _ in range(nstack - 1)])
    self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                              nn.BatchNorm2d(curr_dim))
                                for _ in range(nstack - 1)])
    # heatmap layers
    self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
    for hmap in self.hmap:
      hmap[-1].bias.data.fill_(-2.19)

    # regression layers
    self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
    self.w_h_ = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])

    self.relu = nn.ReLU(inplace=True)

  def forward(self, image):
    inter = self.pre(image)

    outs = []
    for ind in range(self.nstack):
      kp = self.kps[ind](inter)
      cnv = self.cnvs[ind](kp)

      if self.training or ind == self.nstack - 1:
        outs.append([self.hmap[ind](cnv), self.regs[ind](cnv), self.w_h_[ind](cnv)])

      if ind < self.nstack - 1:
        inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
        inter = self.relu(inter)
        inter = self.inters[ind](inter)
    return outs


get_hourglass =   {'large_hourglass':
     exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
   'small_hourglass':
     exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4])}


# In[6]:


import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss_slow(preds, targets):
    pos_inds = targets == 1  # todo targets > 1-epsilon ?
    neg_inds = targets < 1  # todo targets < 1-epsilon ?

    neg_weights = torch.pow(1 - targets[neg_inds], 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def _reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


# In[7]:


from collections import OrderedDict

def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feature(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feature(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(hmap, regs, w_h_, K=100):
    batch, cat, height, width = hmap.shape
    hmap=torch.sigmoid(hmap)

  # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                      ys - w_h_[..., 1:2] / 2,
                      xs + w_h_[..., 0:1] / 2,
                      ys + w_h_[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


# In[8]:


def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

def calculate_precision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


def calculate_image_precision(gts, preds, thresholds=(0.5,), form='pascal_voc') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


# In[9]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = get_pose_net(num_layers=50, head_conv=64, num_classes=1)
# model.to(device)
model = get_hourglass['small_hourglass']
model.to(device)
# state_path = '../input/0625chkpt/chkpt-resnet50fpn-0626-1.pt'
# state_path = "../input/0625chkpt/chkpt-resnet50-0612-1.pt"
state_path = "../input/0625chkpt/chkpt-smallhourglass-0627-1.pt"
state = torch.load(state_path)
for k in state:
    print(k)
model.load_state_dict(state['net'])


# In[10]:


import pandas as pd
import os
import matplotlib.pyplot as plt

DIR_INPUT = '/kaggle/input'
DIR_TRAIN = f'{DIR_INPUT}/global-wheat-detection/train'
DIR_TEST = f'{DIR_INPUT}/global-wheat-detection/test'

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]

class WheatTest(torch.utils.data.Dataset):
    '''
    return [img, hmap, _w_h, regs, indx, ind_mask, center, scale, img_id]
    '''


    def __init__(self, dataframe, data_dir, fix_size=512):
        super(WheatTest, self).__init__()
        self.num_classes = 1
        self.data_dir = data_dir
        self.fix_size = fix_size

        self.data_rng = np.random.RandomState(123)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.df = dataframe
        self.ids = dataframe['image_id'].unique()

        self.max_objs = 128
        self.padding = 31  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': fix_size, 'w': fix_size}
        self.fmap_size = {'h': fix_size // self.down_ratio, 'w': fix_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.data_dir, 'test', self.ids[idx] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = cv2.resize(img,(self.fix_size,self.fix_size)) # convert to fix_size, default by 512
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image

        scale = max(height, width) * 1.0
    

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])

        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))
     
        img = img.astype(np.float32) / 255.



        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]


        return {'image': img,'c': center, 's': scale, 'img_id': img_id}

    def __len__(self):
        return len(self.ids)


test_df = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

test_dataset = WheatTest(test_df,'../input/global-wheat-detection')

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)


class Wheat(torch.utils.data.Dataset):
    '''
    return [img, hmap, _w_h, regs, indx, ind_mask, center, scale, img_id]
    '''


    def __init__(self, dataframe, data_dir, train=True, transform=None, fix_size=512):
        super(Wheat, self).__init__()
        self.num_classes = 1
        self.transform = transform
        self.data_dir = data_dir
        self.fix_size = fix_size

        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.df = dataframe
        self.ids = dataframe['image_id'].unique()
        self.train = train

        self.max_objs = 128
        self.padding = 31  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': fix_size, 'w': fix_size}
        self.fmap_size = {'h': fix_size // self.down_ratio, 'w': fix_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.data_dir, self.ids[idx] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = cv2.resize(img,(self.fix_size,self.fix_size)) # convert to fix_size, default by 512
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image

        scale = max(height, width) * 1.0

        flipped = False
        if self.train:
            scale = scale * np.random.choice(self.rand_scales)
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])

        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        annos = self.df[self.df['image_id'].isin([self.ids[idx]])]

        bboxes = annos[['x', 'y', 'w', 'h']].values
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        gt = bboxes.copy()
        labels = np.zeros(len(bboxes)).astype(np.uint8)
        img = img.astype(np.float32) / 255.

        if self.train:
            color_aug(self.data_rng, img, self.eig_val, self.eig_vec)


        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])
        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        #         ###
        #         fig = plt.figure()

        #         ax1 = fig.add_subplot(121)
        #         ax1.imshow(trans_img)

        #         ax2 = fig.add_subplot(122)
        #         ax2.imshow(trans_fmap)
        #         ###
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((np.ceil(h), np.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1
        #         ###
        #         fig = plt.figure()

        #         ax1 = fig.add_subplot(121)
        #         ax1.imshow(img.transpose(1,2,0))

        #         ax2 = fig.add_subplot(122)
        #         ax2.imshow(hmap.transpose(1,2,0).squeeze(2))
        #         ###

        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id,'boxes':gt}

    def __len__(self):
        return len(self.ids)
    
    
csv_path = '/kaggle/input/global-wheat-detection/train.csv'
df = pd.read_csv(csv_path)

def process_bbox(df):
    df['bbox'] = df['bbox'].apply(lambda x: eval(x))
    df['x'] = df['bbox'].apply(lambda x: x[0])
    df['y'] = df['bbox'].apply(lambda x: x[1])
    df['w'] = df['bbox'].apply(lambda x: x[2])
    df['h'] = df['bbox'].apply(lambda x: x[3])
    df['x'] = df['x'].astype(np.float)
    df['y'] = df['y'].astype(np.float)
    df['w'] = df['w'].astype(np.float)
    df['h'] = df['h'].astype(np.float)

    df.drop(columns=['bbox'],inplace=True)
#     df.reset_index(drop=True)
    return df


df_new = process_bbox(df)

image_ids = df_new['image_id'].unique()
train_ids = image_ids[0:int(0.8*len(image_ids))]
val_ids = image_ids[int(0.8*len(image_ids)):]
train_df = df_new[df_new['image_id'].isin(train_ids)]
val_df = df_new[df_new['image_id'].isin(val_ids)]

val_set = Wheat(val_df,DIR_TRAIN,False)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=1,
    shuffle=True, 
    num_workers=0)


# In[11]:


import time
results1 = []
results2 = []
socre_bound = 0.2

t0 = time.perf_counter()

for sample in val_loader:
    for k in sample:
        if k != 'img_id' and k != 'boxes':
            sample[k] = sample[k].to(device)

    with torch.no_grad():
        img = sample['image']
        img_id = sample['img_id'][0]
        
        output = model(img)[-1]
#         print('using: {}ms'.format(1000*dur))
            
        dets = ctdet_decode(*output, K=50)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
        dets[:, :2] = transform_preds(dets[:, 0:2],
                                      sample['c'].cpu().numpy(),
                                      sample['s'].cpu().numpy(),
                                      (128,128))
        dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                       sample['c'].cpu().numpy(),
                                       sample['s'].cpu().numpy(),
                                       (128,128))
#         dur =  time.perf_counter() - t0
        
        
        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
        boxes = dets[:,:4]
        scores = dets[:,-2]
        gt = sample['boxes'].numpy()[0]
#         print(gt.shape)

#         c_img = img.cpu().numpy()
#         c_img = c_img[0].transpose(1,2,0)
#         c_img = np.ascontiguousarray(c_img)
        
#         s_boxes = []
#         s_scores = []

        boxes = np.clip(boxes, 0 ,1024).astype(np.int32)
#         print(boxes)
#         print(gt)
        precision1 = calculate_image_precision(gt,boxes,(0.5,))
        results1.append(precision1)
        precision2 = calculate_image_precision(gt,boxes,(0.75,))
        results2.append(precision2)
        
        
        if precision1 >0.7:
            print("0.5mAP: {}".format(np.mean(precision1)))
            print("0.75mAP: {}".format(np.mean(precision2)))
            c_img = img.cpu().numpy()
            c_img = c_img[0].transpose(1,2,0)
            c_img = np.ascontiguousarray(c_img)
            c_img = (c_img*std)+mean
            c_img = cv2.resize(c_img,(1024,1024))
            for i, box in enumerate(boxes):
                box = box.astype(np.int32)
#                 if dets[i-1][-2]<0.2:
#                     continue
                cv2.rectangle(c_img,
                          (box[0],box[1]),
                          (box[2],box[3]),
                          (200,200,200), 2)


            for i, box in enumerate(gt):
                box = box.astype(np.int32)
#                 if dets[i-1][-2]<0.2:
#                     continue
                cv2.rectangle(c_img,
                          (int(box[0]),int(box[1])),
                          (int(box[2]),int(box[3])),
                          (200,0,0), 2)

            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(111)
            ax1.imshow(c_img)
        
print("TOTAL 0.5mAP: {}".format(np.mean(results1)))
print("TOTAL 0.75mAP: {}".format(np.mean(results2)))
#         result = {
#             'image_id': img_id,
#             'PredictionString': format_prediction_string(boxes, scores)
#         }
        
#         results.append(result)


# In[12]:


# test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
# a = test_df.head()
# test_df.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




