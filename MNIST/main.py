import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
from torch.autograd import Variable
import cv2
import os

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import math
from pennylane import numpy as qml_np
import torch.nn.functional as F


import data 
import Qmodels as qmd  
import train 


batch_size = 100

class model1(nn.Module):

  	def __init__(self,nq,L):
  		super(model1, self).__init__()
  		self.l1 = qmd.qlayer1(784,nq,L,1)
  		self.l2 = nn.Linear(nq,3)
  		self.f = nn.Softmax(dim=1)

  	def forward(self, x):
  		x = x.reshape(batch_size,784)
  		x = self.l1(x)
  		x = self.l2(x)
  		return self.f(x)



class model2(nn.Module):

  	def __init__(self,nq,L):
  		super(model2, self).__init__()
  		self.l1 = qmd.qlayer2(784,nq,1,L,1)
  		self.l2 = nn.Linear(nq,3)
  		self.f = nn.Softmax(dim=1)

  	def forward(self, x):
  		x = x.reshape(batch_size,784)
  		x = self.l1(x)
  		x = self.l2(x)
  		return self.f(x)





name = 'dataSet4'

if not os.path.exists('./{}'.format(name)):
  	os.mkdir('./{}'.format(name))

'''
if not os.path.exists('./{}/data_train_test'.format(name)):
  	os.mkdir('./{}/data_train_test'.format(name))
'''


#xtrain,xtest = data.dataMNIST(2000,200,batch_size)

xtrain = torch.load('./dataSet3/data_train_test/xtrain.pt')
xtest = torch.load('./dataSet3/data_train_test/xtest.pt')


#torch.save(xtrain, './{}/data_train_test/xtrain.pt'.format(name))
#torch.save(xtest, './{}/data_train_test/xtest.pt'.format(name))



lr = 0.001

for nq in [4,5,6]:

	for L in [2,4,6]:
		loss_hist1 = []
		acc_hist1 = []
		for i in range(6):
			net = model1(nq,L)
			loss,acc=train.train(net,xtrain,xtest,40,lr,i+1,1,nq,L)
			loss_hist1.append(loss)
			acc_hist1.append(acc)

		np.savetxt('./{}/loss_model1_NQ_{}_L_{}_lr_{}.txt'.format(name,nq,L,lr),loss_hist1)
		np.savetxt('./{}/acc_model1_NQ_{}_L_{}_lr_{}.txt'.format(name,nq,L,lr),acc_hist1)

		loss_hist2 = []
		acc_hist2 = []
		for i in range(6):
			net = model2(nq,L)
			loss,acc=train.train(net,xtrain,xtest,40,lr,i+1,2,nq,L)
			loss_hist2.append(loss)
			acc_hist2.append(acc)

		np.savetxt('./{}/loss_model2_NQ_{}_L_{}_lr_{}.txt'.format(name,nq,L,lr),loss_hist2)
		np.savetxt('./{}/acc_model2_NQ_{}_L_{}_lr_{}.txt'.format(name,nq,L,lr),acc_hist2)


