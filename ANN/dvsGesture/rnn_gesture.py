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
from dcll.load_animals_sparse import *
from tqdm import tqdm

import argparse, pickle, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import time
inicio = time.time()



os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# For how many ms do we present a sample during classification
n_iters = 60
n_iters_test = 60 

# How epochs to run before testing
n_test_interval = 20

batch_size = 36
dt = 25000 #us
ds = 4
target_size = 19 # num_classes
n_epochs = 30 # 4500
in_channels = 2 # Green and Red
thresh = 0.3
lens = 0.25
decay = 0.3
learning_rate = 1e-4
time_window = 60
im_dims = im_width, im_height = (128//ds, 128//ds)
names = 'animals_fcorigrnn_count'
wandb.init(project="Animals 19 RNN")
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": n_epochs,
    "batch_size": batch_size
}

parser = argparse.ArgumentParser(description='STDP for DVS gestures')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load data
gen_train, _ = create_data(
        batch_size = batch_size,
        chunk_size = n_iters,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


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


cfg_fc = [512, 512, 19]


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

    def __init__(self, num_classes=18):
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
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(rnn_model))
act_fun = ActFun.apply
print('Generating test...')
n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset = 0)
print('n_test %d' % (n_test))

wandb.init(project="DVS128 Gesture RNN")   
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": n_epochs,
    "batch_size": batch_size
    }
final_time=0.0
best_accuracy = 0.0
best_epoch=0
best_accuracy_test = 0.0
best_epoch_test=0
for epoch in range(n_epochs):
    rnn_model.zero_grad()
    optimizer.zero_grad()
    
    running_loss = 0
    running_acc = 0.0
    train_correct_sum= 0.0
    train_sum = 0.0
    start_time = time.time()

    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
    input = input.float().to(device)
    input = input.permute([1,2,3,4,0])
    labels = torch.from_numpy(labels).float()
    labels = labels[1, :, :]
    outputs, h_state = rnn_model(input, time_window)   
     
    loss = criterion(outputs.cpu(), labels)
    running_loss = running_loss + loss.item()
    loss.backward()
    for name, parms in rnn_model.named_parameters():
        print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad.shape)

    optimizer.step()
    print('Epoch [%d/%d], Loss:%.5f' % (epoch + 1, n_epochs, running_loss))

    if (epoch + 1) % n_test_interval == 0:
        # se testea como de bien lo hace la red con el test
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 400)
        for i in range(len(input_tests)):
            inputTest = input_tests[1].float().to(device)
            inputTest = inputTest.permute([1,2,3,4,0])
            outputs, h_state = rnn_model(inputTest, time_window)

            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)
            labelTest, _ = torch.max(labelTestTmp.data, 0)
            total = total + labelTest.size(0)
            correct = correct + (predicted.cpu() == labelTest).sum()
            # inputTest, labelsTest = gen_train.next()
            # inputTest = inputTest.float().to(device)
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct.float() / total))
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)

    wandb.log({"acc": acc,
                "train_loss": running_loss})
    if best_accuracy < acc:
        best_accuracy = acc
        best_epoch    = epoch
    #if best_accuracy_test < test_acc:
    #    best_accuracy_test = test_acc
    #    best_epoch_test    = epoch

    print(acc)
    print('Saving..')
    state = {
        'net': rnn_model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'acc_record': acc_record,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt' + names + '.t7')
    best_acc = acc
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(best_acc)
print("-----------------------------------------------------------------------------------------------")
fin = time.time()
final_time = fin-inicio
print("FINAL TIME TRAIN: ", str(final_time))
print("BEST EPOCH TEST: ", best_epoch, " BEST ACC TRAIN: ", best_accuracy)
print("BEST EPOCH TEST: ", best_epoch_test, " BEST ACC TEST: ", best_accuracy_test)

    


