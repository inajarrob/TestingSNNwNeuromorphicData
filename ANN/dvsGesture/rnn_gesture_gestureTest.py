# -*- coding: utf-8 -*-

"""
@author: Weihua He
@email: hewh16@gmail.com
"""

# Please install dcll pkgs from below
# https://github.com/nmi-lab/dcll
# and then enjoy yourself.
# If there is any question please mail me.

from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
from tqdm import tqdm

import argparse, pickle, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# For how many ms do we present a sample during classification
n_iters = 60
n_iters_test = 60 

# How epochs to run before testing
n_test_interval = 20

batch_size = 36
dt = 25000 #us
ds = 4
target_size = 11 # num_classes
n_epochs = 3000 # 4500
in_channels = 2 # Green and Red
thresh = 0.3
lens = 0.25
decay = 0.3
learning_rate = 1e-4
time_window = 60
im_dims = im_width, im_height = (128//ds, 128//ds)
names = 'dvsGesture_fcorigrnn_count'

parser = argparse.ArgumentParser(description='STDP for DVS gestures')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load data
_, gen_test = create_data(
        batch_size = batch_size,
        chunk_size = n_iters_test,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)

def generate_test(gen_test, n_test:int, offset=0):
    input_test, labels_test = gen_test.next(offset=offset)
    input_tests = []
    labels1h_tests = []
    n_test = min(n_test,int(np.ceil(input_test.shape[0]/batch_size)))
    for i in range(n_test):
        input_tests.append( torch.Tensor(input_test.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters_test,-1,in_channels,im_width,im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:,i*batch_size:(i+1)*batch_size]))
    return n_test, input_tests, labels1h_tests

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)


cfg_fc = [512, 512, 11]


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


best_acc = 0
acc = 0
acc_record = list([])


class RNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNN(input_size=32*32*2,
                          hidden_size=cfg_fc[0],
                          num_layers=2)


        self.readout = nn.Linear(cfg_fc[0],cfg_fc[-1])

    def zeros_hidden_state(self):
        h_state = []
        for i in range(2):
            h_state.append(torch.zeros(cfg_fc[0],cfg_fc[0],device=device))

        return h_state

    def forward(self, input, h_state, win = 15):

        outs = []

        x = input[:,:,:,:,:win].view(batch_size, -1, win).permute([0,2,1])

        r_out, (h_n, h_c) = self.rnn(x)
        #out = torch.mean(self.readout(r_out), 1)
        out = self.readout(r_out[:,-1,:])
        return out,(h_n, h_c)


rnn_model = RNN_Model()
rnn_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)
import torchfile, sys
try:
    checkpoint = torchfile.load("3000epochs.t7")
    print("aaaaaa")
    rnn_model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    max_test_acc = checkpoint['max_test_acc']
    print("MODELO CARGADO")
except:
    print("modelo no cargado")
    pass

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(rnn_model))
act_fun = ActFun.apply
print('Generating test...')
n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset = 0)
print('n_test %d' % (n_test))

for epoch in range(1):
    rnn_model.zero_grad()
    optimizer.zero_grad()
    
    running_loss = 0
    start_time = time.time()
    #correct = 0
    #total = 0
    for i in range(len(input_tests)):
        inputTest = input_tests[1].float().to(device)
        inputTest = inputTest.permute([1,2,3,4,0])
        outputs, h_state = rnn_model(inputTest, time_window)

        _, predicted = torch.max(outputs.data, 1)
        _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)
        labelTest, _ = torch.max(labelTestTmp.data, 0)
        #total = total + labelTest.size(0)
        #correct = correct + (predicted.cpu() == labelTest).sum()
        print("RESULT: ", predicted.cpu())
print("-----------------------------------------------------------------------------------------------")
fin = time.time()
final_time = fin-start_time
print("FINAL TIME: ", str(final_time))