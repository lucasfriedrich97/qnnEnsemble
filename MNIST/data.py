import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
from torch.autograd import Variable
import cv2
import os


def dataMNIST(ntrain,ntest,batch_size):

	n_samples = ntrain
	img_transform = transforms.Compose([
	      transforms.ToTensor(),
	      transforms.Normalize([0.5], [0.5])
	    ])
	X_train = datasets.MNIST(root='./data', train=True, download=True,
	                             transform=img_transform)


	idx= np.concatenate((np.where(X_train.targets == 0)[0][:n_samples],
	                      np.where(X_train.targets == 1)[0][:n_samples],
	                      np.where(X_train.targets == 2)[0][:n_samples]))

	X_train.data = X_train.data[idx]
	X_train.targets = X_train.targets[idx]
	train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

	n_samples = ntest
	X_test = datasets.MNIST(root='./data', train=False, download=True,
	                            transform=img_transform)

	idx= np.concatenate((np.where(X_test.targets == 0)[0][:n_samples],
	                      np.where(X_test.targets == 1)[0][:n_samples],
	                      np.where(X_test.targets == 2)[0][:n_samples]))

	X_test.data = X_test.data[idx]
	X_test.targets = X_test.targets[idx]
	test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)
	return train_loader,test_loader
