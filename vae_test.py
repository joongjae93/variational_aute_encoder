#-*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import copy
import os

from data import CIFData
from data import collate_pool, get_train_val_test_loader


parser = argparse.ArgumentParser(description='Crystal CVAE')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.0, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.0)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.2)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')
parser.add_argument('--atom-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--nbr-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden bond features in conv layers')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers (default 3)')
parser.add_argument('--l-dim', default=2, type=int, metavar='N',
                    help='number of latent dimension (default 2)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#device = torch.device("cuda" if args.cuda else "cpu")


dataset = CIFData(*args.data_options)
collate_fn = collate_pool
train_loader, _, test_loader, _ = get_train_val_test_loader(
    dataset=dataset,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    train_ratio=args.train_ratio,
    num_workers=args.workers,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    pin_memory=args.cuda,
    train_size=args.train_size,
    val_size=args.val_size,
    test_size=args.test_size,
    return_test=True)


class ConvLayer(nn.Module):

    def __init__(self, atom_fea_len, nbr_fea_len):

        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len, bias=False)
        self.fc_full2 = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.nbr_fea_len, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        
        self.bn1_atom = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn1_bond = nn.BatchNorm1d(2*self.nbr_fea_len)
        self.bn2_atom = nn.BatchNorm1d(self.atom_fea_len)
        self.bn2_bond = nn.BatchNorm1d(self.nbr_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):

        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :] # shape는 N, M, afl
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
    
        total_gated_atom = self.fc_full(total_nbr_fea)
        total_gated_atom = self.bn1_atom(total_gated_atom.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        atom_filter, atom_core = total_gated_atom.chunk(2, dim=2)
        atom_filter = self.sigmoid(atom_filter)
        atom_core = self.softplus1(atom_core)
        atom_sumed = torch.sum(atom_filter * atom_core, dim=1) #여기서 곱은 ew 곱. shape는 N, afl
        atom_sumed = self.bn2_atom(atom_sumed)
        out1 = self.softplus2(atom_in_fea + atom_sumed)
        
        total_gated_bond = self.fc_full2(total_nbr_fea)
        total_gated_bond = self.bn1_bond(total_gated_bond.view(
            -1, self.nbr_fea_len*2)).view(N, M, self.nbr_fea_len*2)
        bond_filter, bond_core = total_gated_bond.chunk(2, dim=2)
        bond_filter = self.sigmoid(bond_filter)
        bond_core = self.softplus1(bond_core)
        bond_sumed = bond_filter * bond_core #여기서 곱은 ew 곱. shape는 N, M, nfl
#        bond_sumed = self.bn2_bond(bond_sumed.view(
#           -1, self.nbr_fea_len)).view(N, M, self.nbr_fea_len) # 배치 노말라이제이션이 필요할까?
        out2 = self.softplus2(nbr_fea + bond_sumed)
        
        return out1, out2


class CrystalGraphConvNet(nn.Module):

    def __init__(self, orig_atom_fea_len, orig_nbr_fea_len, number_of_bonds, 
                 atom_fea_len=128, nbr_fea_len=64, l_dim=2, n_conv=3):
        
        super(CrystalGraphConvNet, self).__init__()
        
        self.embedding_atom = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embedding_atom2 = nn.Linear(atom_fea_len, orig_atom_fea_len)
        self.embedding_bond = nn.Linear(orig_nbr_fea_len, nbr_fea_len)
        self.embedding_bond2 = nn.Linear(nbr_fea_len, orig_nbr_fea_len)
        self.embedding_mu = nn.Linear(atom_fea_len+nbr_fea_len*number_of_bonds, l_dim)
        self.embedding_logvar = nn.Linear(atom_fea_len+nbr_fea_len*number_of_bonds, l_dim)
        self.decode = nn.Linear(l_dim, atom_fea_len+nbr_fea_len*number_of_bonds)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc_softplus = nn.Softplus()
        self.convs_2 = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.fc_to_conv_softplus = nn.Softplus()
        self.output1 = nn.Softplus()
        self.output2 = nn.Sigmoid()
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        
        N1, _ = atom_fea.shape
        atom_nbr_fea = atom_fea[nbr_fea_idx, :]
        F1, F2, F3 = atom_nbr_fea.shape
        total_input_fea = torch.cat(
            [atom_fea.unsqueeze(1).expand(F1, F2, F3),
             atom_nbr_fea, nbr_fea], dim=2)
        atom_fea = self.embedding_atom(atom_fea)
        nbr_fea = self.embedding_bond(nbr_fea)
        _, L1 = atom_fea.shape
        _, M1, L2 = nbr_fea.shape
        for conv_func in self.convs:
            atom_fea, nbr_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        non_pooled_fea = torch.cat([atom_fea, nbr_fea.view(N1, -1)], dim=1)
        atom_fea = self.pooling(atom_fea, crystal_atom_idx)
        atom_fea = self.conv_to_fc_softplus(atom_fea)
        N2, _ = atom_fea.shape
        nbr_fea = self.pooling(nbr_fea, crystal_atom_idx)
        nbr_fea = self.conv_to_fc_softplus(nbr_fea)
        nbr_fea = nbr_fea.view(N2, -1)
        pooled_fea = torch.cat([atom_fea, nbr_fea], dim=1)
        mu = self.embedding_mu(pooled_fea)
        logvar = self.embedding_logvar(pooled_fea)
        z = self.reparameterize(mu, logvar) # sampling
        z_decoded = self.fc_to_conv_softplus(self.decoder(z, non_pooled_fea, pooled_fea, crystal_atom_idx)) # 활성화 함수 없앨까?
        z_decoded, decoded_nbr = torch.split(z_decoded, [L1, M1*L2], dim=1)
        decoded_nbr = decoded_nbr.view(N1, -1, L2)
        for conv_func_2 in self.convs_2:
            z_decoded, decoded_nbr = conv_func_2(z_decoded, decoded_nbr, nbr_fea_idx)
        z_decoded = self.embedding_atom2(z_decoded)
        z_decoded = self.output2(z_decoded)
        decoded_nbr = self.embedding_bond2(decoded_nbr)
        decoded_nbr = self.output2(decoded_nbr)
        z_nbr_fea = z_decoded[nbr_fea_idx, :]
        z_decoded = torch.cat(
            [z_decoded.unsqueeze(1).expand(F1, F2, F3),
             z_nbr_fea, decoded_nbr], dim=2)
        
        return z_decoded, mu, logvar, z, total_input_fea
    
    def decoder(self, z, atom_fea, crys_fea, crystal_atom_idx):
        
        z_decoded = self.decode(z)
        for i in range(len(crystal_atom_idx)):
            l = len(crystal_atom_idx[i])
            crys_decoded = atom_fea[crystal_atom_idx[i]]*z_decoded[i].expand(l, -1)/crys_fea[i].expand(l, -1)
            if i == 0:
                total_decoded = crys_decoded
            else:
                total_decoded = torch.cat((total_decoded, crys_decoded), dim=0)
                
        return total_decoded
    
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def pooling(self, atom_fea, crystal_atom_idx):
        
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
orig_nbr_fea_len = structures[1].shape[-1]
number_of_bonds = structures[1].shape[-2]
model = CrystalGraphConvNet(orig_atom_fea_len, orig_nbr_fea_len, number_of_bonds, 
                            atom_fea_len=args.atom_fea_len, nbr_fea_len=args.nbr_fea_len, 
                            l_dim=args.l_dim, n_conv=args.n_conv)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if args.cuda:
    model.cuda()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -12 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def train(epoch, start_time):
    model.train()
    for batch_idx, (input, _, _) in enumerate(train_loader):
        if args.cuda:
            data = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            data = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        
        recon_batch, mu, logvar, _, batch = model(*data)
        loss, _, _ = loss_function(recon_batch, batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {:.2f} min'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.sampler),
                100. * batch_idx / len(train_loader),
                loss.item() / args.batch_size, (time.time()-start_time)/60))


def evaluate(epoch):
    model.eval()
    with torch.no_grad():
        train_loss = 0
        train_BCE = 0
        train_KLD = 0
        for batch_idx, (input, _, _) in enumerate(train_loader):
            if args.cuda:
                data = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                data = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        
            recon_batch, mu, logvar, _, batch = model(*data)
            loss, BCE, KLD = loss_function(recon_batch, batch, mu, logvar)
            train_loss += loss.item()
            train_BCE += BCE.item()
            train_KLD += KLD.item()
            
        train_loss /= len(train_loader.sampler)
        trainlosslist.append(train_loss)
        train_BCE /= len(train_loader.sampler)
        trainBCElist.append(train_BCE)
        train_KLD /= len(train_loader.sampler)
        trainKLDlist.append(train_KLD)
        print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))


def test(epoch):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_BCE = 0
        test_KLD = 0
        for i, (input, _, _) in enumerate(test_loader):
            if args.cuda:
                data = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                data = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
            
            recon_batch, mu, logvar, _, batch = model(*data)
            loss, BCE, KLD = loss_function(recon_batch, batch, mu, logvar)
            test_loss += loss.item()
            test_BCE += BCE.item()
            test_KLD += KLD.item()

        test_loss /= len(test_loader.sampler)
        testlosslist.append(test_loss)
        test_BCE /= len(test_loader.sampler)
        testBCElist.append(test_BCE)
        test_KLD /= len(test_loader.sampler)
        testKLDlist.append(test_KLD)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


def save_checkpoint(epoch, model, optimizer, path):
    state = {
	'state_dict': model.state_dict(),
	'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def createFolder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSerror:
        print('Error')


if __name__ == "__main__":
    
    start_time = time.time()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    trainlosslist = []
    trainBCElist = []
    trainKLDlist = []
    testlosslist = []
    testBCElist = []
    testKLDlist = []
    epochs = []
    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    current_lr = get_lr(optimizer)
    createFolder('./full_beta_%d' % args.seed)
    for epoch in range(1, args.epochs + 1):
        train(epoch, start_time)
        evaluate(epoch)
        test_loss = test(epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = copy.deepcopy(model.state_dict())
            save_checkpoint(epoch, model, optimizer, './full_beta_%d/%d.pth.tar' % (args.seed, epoch))
            print('Copied best model')
#        save_checkpoint(epoch, model, optimizer, './full_beta_%d/%d.pth.tar' % (args.seed, epoch))
        scheduler.step(test_loss)
        lr = get_lr(optimizer)
        if current_lr != lr:
            current_lr = lr
            print('Loading best model')
            model.load_state_dict(best_model)

        epochs.append(epoch)
    with open("./full_beta_%d/loss.txt" % args.seed, "w") as f:
        f.write("Train Loss\n")
        f.write(str(trainlosslist))
        f.write("\n\n")
        f.write("Test Loss\n")
        f.write(str(testlosslist))
    with open("./full_beta_%d/BCE.txt" % args.seed, "w") as f:
        f.write("Train\n")
        f.write(str(trainBCElist))
        f.write("\n\n")
        f.write("Test\n")
        f.write(str(testBCElist))
    with open("./full_beta_%d/KLD.txt" % args.seed, "w") as f:
        f.write("Train\n")
        f.write(str(trainKLDlist))
        f.write("\n\n")
        f.write("Test\n")
        f.write(str(testKLDlist))
    plt.figure(figsize=(12.8, 9.6))
    plt.plot(epochs, trainlosslist, 'o', linestyle='-', markersize=3, linewidth=1.5, color='blue', label='Train loss')
    plt.plot(epochs, testlosslist, 'o', linestyle='-', markersize=3, linewidth=1.5, color='orange', label='Test loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.savefig('./full_beta_%d/loss.jpg' % args.seed)
    plt.figure(figsize=(12.8, 9.6))
    plt.plot(epochs, trainBCElist, 'o', linestyle='-', markersize=3, linewidth=1.5, color='blue', label='Train BCE')
    plt.plot(epochs, testBCElist, 'o', linestyle='-', markersize=3, linewidth=1.5, color='orange', label='Test BCE')
    plt.plot(epochs, trainKLDlist, 'v', linestyle='--', markersize=3, linewidth=1.5, color='blue', label='Train KLD')
    plt.plot(epochs, testKLDlist, 'v', linestyle='--', markersize=3, linewidth=1.5, color='orange', label='Test KLD')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.savefig('./full_beta_%d/loss2.jpg' % args.seed)
