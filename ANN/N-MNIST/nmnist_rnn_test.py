# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:34:12 2018
# testing the effect of batch
@author: yjwu
"""
import sys
import torch,time,os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from MyLargeDataset import*

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import wandb

num_classes = 10
batch_size  = 50
num_epochs = 10
learning_rate = 5e-4
time_window = 15

probs = 0.0
names = 'nmnist_rnn_test'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#train_path = r'./NMNIST_PYTHON_TRAIN_I.mat'
#test_path = r'./NMNIST_PYTHON_TEST.mat'
test_path = sys.argv[2]
test_dataset = MyDataset(test_path,'r')

# train_dataset = test_dataset

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,drop_last = True)
cfg_fc = [512, 512, 10]


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch>1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


best_acc = 0
acc_record = list([])

class RNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNN(input_size=34*34*2,
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

        out = self.readout(r_out[:,-1,:])


        return out,(h_n, h_c)





rnn_model = RNN_Model()
rnn_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)
import torchfile
try:
    checkpoint = torchfile.load(sys.argv[1])
    rnn_model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    max_test_acc = checkpoint['max_test_acc']
    print("MODELO CARGADO")
except:
    print("modelo no cargado")
    pass

inicio = time.time()
for epoch in range(1):
    running_loss = 0
    h_state = rnn_model.zeros_hidden_state()
    correct = 0
    total = 0

    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
    for images, labels in test_loader:
        images  = images.float().to(device)
        outputs, h_state = rnn_model(images, h_state, time_window)  
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        total += float(labels.size(0))
        correct += (predicted.cpu() == labels).sum()
        print("RESULT: ", predicted.cpu())
print("-----------------------------------------------------------------------------------------------")
fin = time.time()
final_time = fin-inicio
print("FINAL TIME TRAIN: ", str(final_time))

