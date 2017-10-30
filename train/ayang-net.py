import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import h5py
import random
import numpy as np
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


f_train = h5py.File('./preprocess/miniplaces_256_train.h5')
f_val = h5py.File('./preprocess/miniplaces_256_val.h5')


num_images = 100000
num_categories = 100
num_epochs = 1000
batch_size = 32

val_size = 10000
val_batch = 16


normalize = transforms.Normalize(mean=[127, 127, 127], std=[64, 64, 64])

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(96, 256, 5)
		self.conv3 = nn.Conv2d(256, 384, 3)
		self.conv4 = nn.Conv2d(384, 384, 3)
		self.conv5 = nn.Conv2d(384, 256, 3)
		self.fc1 = nn.Linear(4096, num_categories)
		# self.fc2 = nn.Linear(4096, num_categories)
		# self.fc1 = nn.Linear(1024, 100)

	def forward(self, x):
		x = nn.functional.relu(self.conv1(x))
		x = self.pool(nn.functional.relu(self.conv2(x)))
		x = self.pool(nn.functional.relu(self.conv3(x)))
		x = nn.functional.relu(self.conv4(x))
		x = self.pool(nn.functional.relu(self.conv5(x)))
		# print(x.size())
		x = x.view(-1, 4096)
		# x = nn.functional.relu(self.fc1(x))
		x = self.fc1(x)
		return x

net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
	# read in a batch of data
	perm = np.random.permutation(num_images)
	for b in range(0, num_images, batch_size):
		X_train = torch.Tensor(np.array([f_train['images'][i] for i in perm[b:b+batch_size]], dtype=np.float32))
		Y_train = torch.LongTensor(np.array([f_train['labels'][i] for i in perm[b:b+batch_size]], dtype=np.int64))

		# preprocessing
		X_train = normalize(X_train)
		X_train = torch.transpose(X_train, 1, 3) # switch channel dimension to second
		# Y_onehot = torch.LongTensor(batch_size, num_categories)
		# Y_onehot.zero_()
		# Y_onehot.scatter_(1, Y_train, 1)


		# forward/back prop
		optimizer.zero_grad()
		inputs = torch.autograd.Variable(X_train.cuda())
		outputs = net(inputs)
		loss = criterion(outputs, torch.autograd.Variable(Y_train.cuda()))
		loss.backward()
		optimizer.step()

		print("Epoch %d Step %d / %d: Loss = %.2f" % (epoch + 1, b / batch_size + 1, num_images / batch_size, loss.data[0]))

		# evaluate val accuracy
		# if b % 20000 == (20000 - 32):
		if b / batch_size % 200 == 199:
			count = 0
			for i in range(0, val_batch * 30, val_batch):
				X_val = torch.Tensor(np.array([f_val['images'][j] for j in range(i, i + val_batch)], dtype=np.float32))
				Y_val = np.array([f_val['labels'][j] for j in range(i, i + val_batch)], dtype=np.int64)
				# Y_val = torch.LongTensor(np.array([f_val['labels'][i] for i in range(i, i + val_batch)], dtype=np.int64))

				X_val = normalize(X_val)
				X_val = torch.transpose(X_val, 1, 3)

				inputs_val = torch.autograd.Variable(X_val.cuda())
				# print(inputs_val.size())
				outputs_val = net(inputs_val)
				rows = outputs_val.split(val_batch)
				rows = rows[0].data.cpu().numpy()
				for j in range(len(rows)):
					tmp = rows[j]
					tmp = tmp.argsort()[-5:][::-1]
					if Y_val[j] in tmp:
						count += 1
			print("Validation accuracy: %f%%" % (count * 100 / (val_batch * 30)))
	count = 0
	for i in range(0, val_size, val_batch):
		X_val = torch.Tensor(np.array([f_val['images'][j] for j in range(i, i + val_batch)], dtype=np.float32))
		Y_val = np.array([f_val['labels'][j] for j in range(i, i + val_batch)], dtype=np.int64)
		# Y_val = torch.LongTensor(np.array([f_val['labels'][i] for i in range(i, i + val_batch)], dtype=np.int64))

		X_val = normalize(X_val)
		X_val = torch.transpose(X_val, 1, 3)

		inputs_val = torch.autograd.Variable(X_val.cuda())
		# print(inputs_val.size())
		outputs_val = net(inputs_val)
		rows = outputs_val.split(val_batch)
		rows = rows[0].data.cpu().numpy()
		for j in range(len(rows)):
			tmp = rows[j]
			tmp = tmp.argsort()[-5:][::-1]
			if Y_val[j] in tmp:
				count += 1
	print("Validation accuracy: %f%%" % (count * 100 / val_size))

print("Finished training")
