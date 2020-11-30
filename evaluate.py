import time
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import data_loader_evaluate
from torch.autograd import Variable
from model import _G_xvz, _G_vzx
from itertools import *
import pdb
import warnings

warnings.filterwarnings('ignore')

dd = pdb.set_trace

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_list", type=str, default="./list_test.txt")
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument('--outf', default='./evaluate', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./output', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=False)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# need initialize!!
G_xvz = _G_xvz()
G_vzx = _G_vzx()

train_list = args.data_list

train_loader = torch.utils.data.DataLoader(
    data_loader_evaluate.ImageList( train_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)


def L1_loss(x, y):
    return torch.sum(torch.abs(x-y))


x = torch.FloatTensor(args.batch_size, 3, 128, 128)
x_bar_bar_out = torch.FloatTensor(10, 3, 128, 128)

v_siz = 9
z_siz = 128-v_siz
v = torch.FloatTensor(args.batch_size, v_siz)
z = torch.FloatTensor(args.batch_size, z_siz)

if args.cuda:
    G_xvz = torch.nn.DataParallel(G_xvz)
    G_vzx = torch.nn.DataParallel(G_vzx)

    x = x
    x_bar_bar_out = x_bar_bar_out
    v = v
    z = z

x = Variable(x)
x_bar_bar_out = Variable(x_bar_bar_out)
v = Variable(v)
z = Variable(z)


def load_model(net, path, name):
    state_dict = torch.load('%s/%s' % (path,name), map_location=torch.device('cpu'))
    own_state = net.state_dict()

    for name, param in state_dict.items():
        # if name not in own_state:
        #     print('not load weights %s' % name)
        #     continue
        own_state[name.replace('module.', '')].copy_(param)
        print('load weights %s' % name)

def load_model_ensemble(net, path1, path2, name, lambda1, lambda2):
    state_dict1 = torch.load('%s/%s' % (path1,name), map_location=torch.device('cpu'))
    state_dict2 = torch.load('%s/%s' % (path2, name), map_location=torch.device('cpu'))
    own_state = net.state_dict()

    for name, param1 in state_dict1.items():
        param2 = state_dict2[name]
        own_state[name.replace('module.', '')].copy_((lambda1*param1+lambda2*param2))

batch_size = args.batch_size
cudnn.benchmark = True
G_xvz.eval()
G_vzx.eval()

lambda11_grid = np.linspace(0, 0.5, 5)
lambda21_grid = np.linspace(0, 0.5, 5)
best_lam11 = lambda11_grid[0]
best_lam12 = 1-best_lam11
best_lam21 = lambda21_grid[0]
best_lam22 = 1-best_lam21

min_loss = float("Inf")
for lam11 in lambda11_grid:
    lam12 = 1-lam11
    for lam21 in lambda21_grid:
        loss = 0
        lam22 = 1-lam21
        load_model_ensemble(G_xvz, './pretrained_model', './output', 'netG_xvz.pth', lam11, lam12)
        load_model_ensemble(G_vzx, './pretrained_model', './output', 'netG_vzx.pth', lam21, lam22)
        for i, (data) in enumerate(train_loader):
            img = data
            x.data.resize_(img.size()).copy_(img)

            x_bar_bar_out.data.zero_()
            v_bar, z_bar = G_xvz(x)

            for one_view in range(9):
                v.data.zero_()
                for d in range(data.size(0)):
                    v.data[d][one_view] = 1
                exec('x_bar_bar_%d = G_vzx(v, z_bar)' % (one_view))

            for d in range(batch_size):
                x_bar_bar_out.data[0] = x.data[d]
                for one_view in range(9):
                    exec('x_bar_bar_out.data[1+one_view] = x_bar_bar_%d.data[d]' % (one_view))
                loss += L1_loss(x.data, x_bar_bar_out)

    if loss<min_loss:
        min_loss = loss
        best_lam11 = lam11
        best_lam12 = lam12
        best_lam21 = lam21
        best_lam22 = lam22

print('best_lam11: {:.6f}, best_lam12: {:.6f}'.format(best_lam11, best_lam12))
print('best_lam21: {:.6f}, best_lam22: {:.6f}'.format(best_lam21, best_lam22))

load_model_ensemble(G_xvz, './pretrained_model', './output', 'netG_xvz.pth', best_lam11, best_lam12)
load_model_ensemble(G_vzx, './pretrained_model', './output', 'netG_vzx.pth', best_lam21, best_lam22)

# load_model(G_xvz, args.modelf, 'netG_xvz.pth')
# load_model(G_vzx, args.modelf, 'netG_vzx.pth')

for i, (data) in enumerate(train_loader):
    img = data
    x.data.resize_(img.size()).copy_(img)

    x_bar_bar_out.data.zero_()
    v_bar, z_bar = G_xvz(x)

    for one_view in range(9):
        v.data.zero_()
        for d in range(data.size(0)):
            v.data[d][one_view] = 1
        exec ('x_bar_bar_%d = G_vzx(v, z_bar)' % (one_view))

    for d in range(batch_size):
        x_bar_bar_out.data[0] = x.data[d]
        for one_view in range(9):
            exec ('x_bar_bar_out.data[1+one_view] = x_bar_bar_%d.data[d]' % (one_view))
        vutils.save_image(x_bar_bar_out.data,
                    '%s/%d_x_bar_bar.png' % (args.outf, i*batch_size+d), nrow = 10, normalize=True, pad_value=255)