'''
--------------------------------------------------------
@File    :   trainer.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/16 13:35     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer(object):
	""" The class to train networks"""
	def __init__(self,
	             dataloader,
	             network,
	             optimizer=torch.optim.Adam,
	             learning_rate=1.0e-4,
	             epoch=1000,
	             loss_function=nn.MSELoss()):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.dataloader = dataloader
		self.net = network.to(self.device)
		self.lr = learning_rate
		self.opt = optimizer(self.net.parameters(), lr=self.lr)
		self.epoch = epoch
		self.loss_F = loss_function

	def train(self):
		self.net.train()
		for epoch in range(1, self.epoch + 1):
			for batch_idx, (data, target) in enumerate(self.dataloader):
				data, target = data.to(self.device), target.to(self.device)
				self.opt.zero_grad()
				output = self.net(data)
				loss = F.nll_loss(output, target)
				loss.backward()
				self.opt.step()
				if (batch_idx + 1) % 10 == 0:
					print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
						epoch, batch_idx * len(data), len(self.dataloader.dataset),
						100. * batch_idx / len(self.dataloader), loss.item())
					)