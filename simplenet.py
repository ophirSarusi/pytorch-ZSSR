import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
from torch import fft
import numpy as np


class simpleNet(nn.Module):
	def __init__(self, Y=True, loss_type='mse', out_values=256):
		super(simpleNet, self).__init__()

		self.loss_type = loss_type
		channels = 128

		in_d = 1
		if not Y:
			in_d = 3

		out_d = in_d
		if loss_type == 'ce':
			out_d = 101

		self.input = nn.Conv2d(in_channels=in_d, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)

		self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv6 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)

		if self.loss_type == 'ce':
			self.output = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
									padding_mode='reflect', bias=False)
			self.final1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)
			self.final2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)
			self.final3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)
			self.final4 = nn.Conv2d(in_channels=channels, out_channels=out_d, kernel_size=1, stride=1, padding=0, bias=False)
		else:
			self.output = nn.Conv2d(in_channels=channels, out_channels=out_d, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
			# self.output = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
			# self.fc = nn.Parameter(torch.normal(0.0, sqrt(2. / channels), size=(channels, 256)), requires_grad=True)

		self.relu = nn.ReLU(inplace=False)
		# self.range = nn.Parameter(torch.Tensor(np.array(range(256))/255).unsqueeze(dim=1), requires_grad=False)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		# residual = x
		residual = self.relu(self.input(x))

		out = residual
		out = self.relu(self.conv1(out))
		out = self.relu(self.conv2(out))
		out = self.relu(self.conv3(out))
		out = self.relu(self.conv4(out))
		out = self.relu(self.conv5(out))
		out = self.relu(self.conv6(out))

		out = self.output(out)
		out1 = torch.add(out, residual)

		if self.loss_type == 'ce':
			out = self.relu(self.final1(out1))
			out = self.relu(self.final2(out))
			out = self.relu(self.final3(out))
			out = self.relu(self.final4(out))

		# out = torch.einsum('bcwh,ck->bkwh', out, self.fc) / 10
		# out = F.softmax(out, dim=1)
		# out = torch.einsum('bcwh,ck->bkwh', out, self.range)

		# if self.loss_type == 'ce':
		# 	out = torch.einsum('bcwh,ck->bkwh', out, self.fc1)
		# 	out = torch.einsum('bcwh,ck->bkwh', out, self.fc2)
		# 	out = torch.einsum('bcwh,ck->bkwh', out, self.fc3)

		return out


class FourierNet(nn.Module):
	def __init__(self, Y=True, loss_type='mse'):
		super(FourierNet, self).__init__()
		in_d = 1
		channels = 128
		fft_channels = 256
		if not Y:
			in_d = 3

		out_d = in_d
		if loss_type == 'ce':
			out_d = 101

		self.input = nn.Conv2d(in_channels=in_d, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)

		self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		# self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		# self.conv5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		# self.conv6 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)

		self.f_conv_re1 = nn.Conv2d(in_channels=in_d, out_channels=fft_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.f_conv_re2 = nn.Conv2d(in_channels=fft_channels, out_channels=fft_channels, kernel_size=3, stride=1, padding=1, bias=False)
		# self.f_conv_re3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
		# self.f_conv_re4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.f_conv_re5 = nn.Conv2d(in_channels=fft_channels, out_channels=in_d, kernel_size=3, stride=1, padding=1, bias=False)
		self.f_conv_im1 = nn.Conv2d(in_channels=in_d, out_channels=fft_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.f_conv_im2 = nn.Conv2d(in_channels=fft_channels, out_channels=fft_channels, kernel_size=3, stride=1, padding=1, bias=False)
		# self.f_conv_im3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
		# self.f_conv_im4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.f_conv_im5 = nn.Conv2d(in_channels=fft_channels, out_channels=in_d, kernel_size=3, stride=1, padding=1, bias=False)

		self.output = nn.Conv2d(in_channels=channels, out_channels=out_d, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
		self.relu = nn.ReLU(inplace=False)

		self.param = nn.parameter.Parameter(0.1 * torch.rand(1))

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.relu(self.input(x))

		out = inputs
		out = self.relu(self.conv1(out))
		out = self.relu(self.conv2(out))
		out = self.relu(self.conv3(out))
		# out = self.relu(self.conv4(out))
		# out = self.relu(self.conv5(out))
		# out = self.relu(self.conv6(out))

		out = self.output(out)

		out1 = torch.add(out, residual)

		out = fft.rfft2(out1)
		out_real = self.relu(self.f_conv_re1(out.real))
		out_real = self.relu(self.f_conv_re2(out_real))
		# out_real = self.relu(self.f_conv_re3(out_real))
		# out_real = self.relu(self.f_conv_re4(out_real))
		out_real = self.f_conv_re5(out_real)
		out_imag = self.relu(self.f_conv_im1(out.imag))
		out_imag = self.relu(self.f_conv_im2(out_imag))
		# out_imag = self.relu(self.f_conv_im3(out_imag))
		# out_imag = self.relu(self.f_conv_im4(out_imag))
		out_imag = self.f_conv_im5(out_imag)
		out = torch.complex(out_real, out_imag)

		# out = torch.add(out1, torch.abs(fft.irfft2(out)))
		out = torch.add(out1, torch.mul(torch.abs(fft.irfft2(out)), self.param))

		return out
