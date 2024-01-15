import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import math
import torch.nn.functional as F


def train(net,xtrain,xtest,epochs,lr,ind,aa,NQ,LL):
  	optimizer = optim.Adam(net.parameters(), lr=lr)
  	Loss = nn.MSELoss()

  	acc = []
  	loss = []

  	for epoch in range(epochs):
  		soma = 0
  		ss = 1
  		net.train()
  		with tqdm(xtrain, unit="batch") as tepoch:
  			for x, y in tepoch:
  				tepoch.set_description(f" Model{aa}:{ind} lr:{lr} NQ:{NQ} L:{LL} {epoch+1}/{epochs} ")
  				out = net(x).float()
  				target = F.one_hot(y, num_classes=3).float()
  				l = Loss(out,target)
  				soma += l.item()
  				optimizer.zero_grad()
  				l.backward()
  				optimizer.step()
  				ss+=1

  			loss.append(soma/len(xtrain))

  		net.eval()
  		soma_acc = 0
  		L = 0
  		for x,y in xtest:
  			out = net(x)
  			_, preds = torch.max(out, 1)
  			soma_acc += torch.sum(preds == y.data)
  			L += len(y.data)

  		acc.append( soma_acc/L )

  	return np.array(loss),np.array(acc)
