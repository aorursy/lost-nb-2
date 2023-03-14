#!/usr/bin/env python
# coding: utf-8

# In[1]:


from set1.util.path import path_dropbox_data
from set1.util.iomanager import join, readpickle, writepickle
import numpy as np
import json
import csv
from tqdm import tqdm

class_len = readpickle(join(path_dropbox_data(), r'QuickDraw\class_length'))
def load_data(filename):
	print('loading class:{}'.format(filename))
	data = []
	with open(join(path_dropbox_data(), 'QuickDraw\\train_simplified\\', filename + '.csv'))as file:
		read = csv.reader(file)
		read.__next__()
		for row in read:
			img_str = row[1]
			img_array = np.array(json.loads(img_str))
			data.append([img_array, sorted(list(class_len.keys())).index(filename)])
	return data

if __name__ == "__main__":
	merged = dict()
	for name in tqdm(list(class_len.keys())):
		merged[name] = load_data(name)
	writepickle(join(path_dropbox_data(), 'QuickDraw'), merged, 'merged')


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F

def res_block(in_channel, out_channel):
	block = nn.Sequential(
		nn.InstanceNorm2d(in_channel),
		nn.ReLU(),
		nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
		nn.InstanceNorm2d(out_channel),
		nn.ReLU(),
		nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
	)
	return block

def big_block(in_channel, out_channel, n_block):
	layers = []
	layers.append(res_block(in_channel, out_channel))
	for i in range(n_block-1):
		layers.append(res_block(out_channel, out_channel))
	return nn.Sequential(*layers)

class Resnet(nn.Module):
	def __init__(self):
		super().__init__()
		self.block1 = big_block(64, 64, 3)
		self.block2 = big_block(128, 128, 4)
		self.block3 = big_block(256, 256, 6)
		self.block4 = big_block(512, 512, 3)
		self.downscale = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=4, padding=2)
		self.downscale1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
		self.downscale2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
		self.downscale3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
		self.FCLayer = nn.Linear(512, 340)

	def forward(self, input):
		skip = [None]*3
		out = self.downscale(input)	#64,128,128
		skip[0] = out	#64,128,128

		out = self.block1(out)	#64,128,128
		out = self.downscale1(torch.cat([out, skip[0]], 1))	#128,128,128 to 128,64,64
		skip[1] = out		#128,64,64

		out = self.block2(out)	#128,64,64
		out = self.downscale2(torch.cat([out, skip[1]], 1)) #256,64,64 to 256,32,32
		skip[2] = out	#256,32,32

		out = self.block3(out)	#256,32,32
		out = self.downscale3(torch.cat([out, skip[2]], 1))	#512,32,32 to 512,16,16

		out = self.block4(out)	#512,16,16
		out = F.avg_pool2d(out, 512)

		out = out.reshape([-1,512])
		out = self.FCLayer(out)
		return out


# In[3]:


import numpy as np
import cv2
from set1.util.iomanager import join, readpickle
from set1.util.path import path_dropbox_data
import torch.utils.data
from torchvision.transforms import transforms
import torch.nn as nn
import torch


def draw_image(img_array):
	img = np.zeros((256, 256, 3), np.uint8)
	for i in range(img_array.shape[0]):
		for j in range(img_array[i][0].__len__() - 1):
			img = cv2.line(img, (img_array[i][0][j], img_array[i][1][j]), (img_array[i][0][j + 1], img_array[i][1][j + 1]), (256, 256, 256), 3)
	return np.transpose(img[:, :, 0:1], (2, 0, 1))


class DoodleDataset(torch.utils.data.Dataset):
	def __init__(self, filenumber):
		self.path = join(path_dropbox_data(), 'QuickDraw')
		self.data = readpickle(join(self.path, 'merged{}'.format(filenumber)))
		self.length = readpickle(join(self.path, 'file_length'))[filenumber]

	def __getitem__(self, index):
		image = draw_image(self.data[index][0])
		image = torch.from_numpy(image).float()
		label = self.data[index][1]
		return image, label

	def __len__(self):
		return self.length


class Doodle_loss(nn.Module):
	def __init__(self):
		super().__init__()
		self.loss = nn.CrossEntropyLoss()

	def forward(self, out, label):
		loss = self.loss(out, label)
		return loss


# In[4]:


import torch
import torch.nn as nn
import argparse
from set1.util.path import path_dropbox_data
from set1.doodle.util import DoodleDataset, Doodle_loss
from set1.doodle.network2 import Resnet
from set1.YOLO.saver import Saver
from set1.util.iomanager import join

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', '-lr', type=float, default=1e-7)
parser.add_argument('--eval_period', '-ep', type=int, default=100)
parser.add_argument('--save_period', '-sp', type=int, default=10000)
parser.add_argument('--num_iter', '-ne', type=int, default=100000)
parser.add_argument('--network_name', '-nn', type=str, default='SR_GAN_24')
parser.add_argument('--batch_size', '-bs', type=int, default=256)
parser.add_argument('--path_root', '-pr', type=str, default=path_dropbox_data())

args = parser.parse_args([])

learning_rate = args.learning_rate
eval_period = args.eval_period
save_period = args.save_period
batch_size = args.batch_size
num_iter = args.num_iter
network_name = args.network_name

device = torch.device("cuda:0")
model = Resnet()
if torch.cuda.device_count() > 1:
	model = nn.DataParallel(model)
model.to(device)

criterion = Doodle_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
doodle_saver = Saver(model, join(path_dropbox_data(), r'doodle', r'save'), "doodle", max_to_keep=100)
doodle_saver.load()

for epoch in range(100):
	data_loader = torch.utils.data.DataLoader(dataset=DoodleDataset,
											  batch_size=batch_size,
											  shuffle=True)
	print("loading {}th dataset".format(epoch % 10))
	for param in optimizer.param_groups:
		param['lr'] *= 0.8
	for i, (images, labels) in enumerate(data_loader):
		images = images.to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i) % eval_period == 0:
			acc = torch.sum(labels == torch.argmax(outputs, 1)).float() / batch_size
			print("Epoch [{}/{}], Step [{}] Loss: {:.4f} Accuracy: {}"
				  .format(epoch + 1, 100, i, loss.item(), acc))

		if (i) % save_period == 0:
			doodle_saver.save(i + epoch * data_loader.__len__())
			print("saved at iter_{}".format(i + epoch * data_loader.__len__()))


# In[5]:


filenumber [3], Step [0] accuracy : [9.0234375%]
filenumber [3], Step [10] accuracy : [89.9609375%]
filenumber [3], Step [20] accuracy : [90.46875%]
filenumber [3], Step [30] accuracy : [90.4296875%]
filenumber [3], Step [40] accuracy : [88.9453125%]
filenumber [3], Step [50] accuracy : [90.1953125%]
filenumber [3], Step [60] accuracy : [89.21875%]
filenumber [3], Step [70] accuracy : [90.625%]
filenumber [3], Step [80] accuracy : [90.3515625%]
filenumber [3], Step [90] accuracy : [89.453125%]
filenumber [3], Step [100] accuracy : [90.078125%]


# In[6]:


filenumber [8], Step [0] accuracy : [0.0%]
filenumber [8], Step [10] accuracy : [0.0%]
filenumber [8], Step [20] accuracy : [0.0%]
filenumber [8], Step [30] accuracy : [0.0%]
filenumber [8], Step [40] accuracy : [0.0%]
filenumber [8], Step [50] accuracy : [0.0%]
filenumber [8], Step [60] accuracy : [0.0%]
filenumber [8], Step [70] accuracy : [0.0%]
filenumber [8], Step [80] accuracy : [0.0%]
filenumber [8], Step [90] accuracy : [0.0%]
filenumber [8], Step [100] accuracy : [0.0%]

