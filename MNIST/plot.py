import numpy as np  
import matplotlib.pyplot as plt  


name = 'acc'
Name = 'Acc'

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(3,3)

lr = 0.001



s1 = 0
for NQ in [4,5,6]:
	s2 = 0
	for L in [2,4,6]:
		model1 = np.loadtxt('./dataSet5/{}_model1_NQ_{}_L_{}_lr_{}.txt'.format(name,NQ,L,lr))
		model2 = np.loadtxt('./dataSet5/{}_model2_NQ_{}_L_{}_lr_{}.txt'.format(name,NQ,L,lr))
			
		dx = np.arange(len( model1.mean(0) ))
		ax[s1][s2].plot(dx, model1.mean(0),label='Ref 2')
		ax[s1][s2].fill_between(dx, model1.min(0), model1.max(0), alpha=0.3)
		ax[s1][s2].plot(dx, model2.mean(0),label='Model 2')
		ax[s1][s2].fill_between(dx, model2.min(0), model2.max(0),alpha=0.3)
		ax[s1][s2].set_title('$NQ = ${}, L = {}'.format(NQ,L))
		ax[s1][s2].legend()
		if s2 ==0:
			ax[s1][s2].set_ylabel('{}'.format(Name))
		if s1 == 2:	

			ax[s1][s2].set_xlabel('Epochs')
	        
		s2 +=1
	s1+=1


if name == 'loss':

	for i in range(3):
		for j in range(3):
			ax[i][j].set_ylim(0,0.3)

plt.subplots_adjust(left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.3)

plt.show()

        
