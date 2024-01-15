import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import pennylane as qml
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import math
from pennylane import numpy as qml_np
import torch.nn.functional as F


def qlayer_(nq,nl,ent):
	dev = qml.device('default.qubit',wires=nq)
	@qml.qnode(dev,interface='torch')
	def f(inputs,w):
		for i in range(nl):
			for j in range(nq):
				qml.RY(inputs[j],wires=j)

			for j in range(nq):
				qml.RY(w[j][i][0],wires=j)
				qml.RZ(w[j][i][1],wires=j)
			if ent == 0:
				for j in range(nq-1):
					qml.CNOT(wires=[j,j+1])
			elif ent == 1:
				for j in range(nq):
					for k in range(nq):
						if j!=k:
							qml.CNOT(wires=[j,k])

		A = np.array([ [1,0],[0,0] ])
		return [qml.expval(qml.Hermitian(A, wires=i)) for i in range(nq)]

	return f



class qlayer1(nn.Module):

	def __init__(self,N,NQ,NL,ent):
	    super(qlayer1, self).__init__()
	    self.hidden = nn.Linear(N,NQ)
	    self.relu = nn.ReLU()
	    self.ql = qlayer_(NQ,NL,ent)
	    self.param = nn.Parameter(torch.FloatTensor(NQ,NL,2).uniform_(0, 2*math.pi))

	def forward(self, x):
	    if np.ndim(x[0]) == 0:
	      x = x.reshape(1,len(x))
	    x = self.hidden(x)
	    x = self.relu(x)

	    x = x.T

	    x = self.ql(x,self.param)

	    x = torch.stack(x)

	    return x.T.float()





class qlayer2(nn.Module):

	def __init__(self,N,NQ,NL,learness,ent):
	    super(qlayer2, self).__init__()
	    self.learness = learness

	    self.hidden = nn.Linear(N,NQ)
	    self.relu = nn.ReLU()
	    self.ql = qlayer_(NQ,NL,ent)

	    self.param = nn.ParameterList([nn.Parameter( torch.FloatTensor(NQ,NL,2).uniform_(0, 2*math.pi) ) for _ in range(self.learness)])
	        

	    self.p = nn.Parameter(torch.ones(self.learness))

	def forward(self, x):

		if np.ndim(x[0]) == 0:
			x = x.reshape(1,len(x))

		x = self.hidden(x)
		x = self.relu(x)
		Z = torch.sum( torch.exp(-self.p) )
		p = torch.exp(-self.p)/Z
		p = p.reshape(self.learness,1,1)
		y = []
		x = x.T
		for i in range(self.learness):

			out = self.ql(x,self.param[i])
			out = torch.stack(out).T
			y.append(out)

		ys = torch.stack(y)
		zz = p*ys
		return torch.sum(zz, axis=0).float()


