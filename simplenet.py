import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable


class simpleNet(nn.Module):
	def __init__(self, Y=True, loss_type='mse'):
		super(simpleNet, self).__init__()
		in_d = 1
		if not Y:
			in_d = 1

		out_d = in_d
		if loss_type == 'ce':
			out_d = 256

		self.input = nn.Conv2d(in_channels=in_d, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)

		self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)

		self.output = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.final1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)
		self.final2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)
		self.final3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)
		self.final4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)

		self.relu = nn.ReLU(inplace=False)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))

		out = inputs
		out = self.conv1(self.relu(out))
		out = self.conv2(self.relu(out))
		out = self.conv3(self.relu(out))
		out = self.conv4(self.relu(out))
		out = self.conv5(self.relu(out))
		out = self.conv6(self.relu(out))

		out = self.output(self.relu(out))
		out = torch.add(out, residual)
		out = self.final1(self.relu(out))
		out = self.final2(self.relu(out))
		out = self.final3(self.relu(out))
		out = self.final4(self.relu(out))





		# out = torch.einsum('bcwh,ck->bkwh', out, self.fc)
		


		return out
