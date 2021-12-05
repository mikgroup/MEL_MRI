#!/usr/bin/env python
#Borrowed from Jon Tamir Basic 
import sigpy.plot as pl
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import scipy.io
from torch.autograd import Variable
# import UFNet
import os
# import bart
from torch import optim
import torch_utils as flare
from pytorch3dunet.unet3d.model import UNet3D
from collections import OrderedDict
class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)
    
    
class ConvSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dims, bias=True):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        if dims == 3:
            padding_layer=torch.nn.ReplicationPad3d
            conv_layer=torch.nn.Conv3d
            pad_params=(ka,kb,ka,kb,ka,kb)
        elif dims == 2:
            padding_layer=torch.nn.ReflectionPad2d
            conv_layer=torch.nn.Conv2d
            pad_params=(ka,kb,ka,kb)
            
        conv_params={'in_channels':in_channels, 'out_channels':out_channels, 'kernel_size':kernel_size, 'bias':bias}
        self.net = torch.nn.Sequential(
            padding_layer(pad_params),
            conv_layer(**conv_params)
        )
    def forward(self, x):
        return self.net(x)

def bart_valid(knee_ksp,knee_mps,knee_masks):
    ksp_under = (knee_ksp*knee_masks).transpose((1,2,0))[None,...]
    ksp_mps = knee_mps.transpose((1,2,0))[None,...]
    return bart.bart(1,"pics -l1 -r 0.02 -S",ksp_under,ksp_mps)
def CG_adj(ksp,mps,mask):
    SenseModel = flare.SenseModel(mps,mask) 
    adj = SenseModel.adjoint(ksp)
    return SenseModel,adj
def CG_adj_3D(ksp,mps,mask):
    SenseModel = flare.SenseModel_3D(mps,mask) 
    adj = SenseModel.adjoint(ksp)
    return SenseModel,adj
def CG_MoDL_3D(ksp,mps,mask,lam = 0):
    SenseModel = flare.SenseModel_3D(mps,mask)    
    adj = SenseModel.adjoint(ksp)
    CG_alg = flare.ConjGrad(Aop_fun=SenseModel.normal,b=adj,verbose=False,l2lam=lam)
    return CG_alg.forward(adj)
def CG_MoDL(ksp,mps,mask,lam = 0):
    SenseModel = flare.SenseModel(mps,mask)    
    adj = SenseModel.adjoint(ksp)
    CG_alg = flare.ConjGrad(Aop_fun=SenseModel.normal,b=adj,verbose=False,l2lam=lam)
    return CG_alg.forward(adj)

class MoDL_3D(nn.Module):
    def __init__(self,M=None,A = None,lam_l2 = 0,unroll = 3,cg_max = 10):
        super(MoDL_3D, self).__init__()
        self.Model = M
        self.A = A
        self.lam2 = lam_l2
        self.urnum = unroll
        self.cg = cg_max

    def forward(self, adj):
        out = adj
#         print(out.shape)
        for i in range(self.urnum):
            print(i)
            out = self.Model(out).squeeze(0).permute(1,2,3,0)
#             out1 = out
#             print(adj.shape)
#             print(out.shape)
#             print(out.shape)
#             print(adj.squeeze(0).permute(1,2,3,0).shape)
            rhs = adj.squeeze(0).permute(1,2,3,0) + self.lam2 * out
            CG_alg = flare.ConjGrad(Aop_fun=self.A.normal,b=rhs,verbose=False,l2lam=self.lam2,max_iter=self.cg)
            out = CG_alg.forward(rhs).permute(3,0,1,2).unsqueeze(0)
#             print(out.shape)
        return out

class CG_module(nn.Module):
    def __init__(self,A=None,adjoint=None,verbose = False,lam_l2 = 0,cg_max = 10):
        super(CG_module,self).__init__()
        self.A = None
        self.adj = None
        self.lam = lam_l2
        self.cg = cg_max
        self.verbose = verbose
    def initiate(self,A,adjoint):
        self.A = A
        self.adj = adjoint
    def forward(self,x):
        rhs = self.adj + self.lam * x
        out = flare.ConjGrad(Aop_fun=self.A.normal,b=rhs,verbose=self.verbose,l2lam=self.lam,max_iter=self.cg).forward(rhs)
        return out
    def reverse(self,x):
        out = (1/self.lam) * ((self.A.normal(x)+self.lam*x)-self.adj)
        return out

    
    
class ResNet4(nn.Module):
    def __init__(self, dims, num_channels=32, kernel_size=3, T=5, num_layers=3,device='cpu'):
        super(ResNet4, self).__init__()
        
        conv = lambda in_channels, out_channels, filter_size: ConvSame(in_channels, out_channels, filter_size, dims)
        self.dims = dims
        self.device = device
        layer_dict = OrderedDict()
        layer_dict['conv1'] = conv(2,num_channels,kernel_size)
        layer_dict['relu1'] = nn.ReLU()
        for i in range(num_layers-2):
            layer_dict[f'conv{i+2}'] = conv(num_channels, num_channels, kernel_size)
            layer_dict[f'relu{i+2}'] = nn.ReLU()
        layer_dict[f'conv{num_layers}'] = conv(num_channels,2,kernel_size)
        
        self.model = nn.Sequential(layer_dict).to(self.device)
        self.T = T
        
    def forward(self,x):
        return x + self.step(x)
    
    def step(self,x):
        y = self.model(x)
        # reshape (batch,channel=2,x,y) -> (batch,x,y,channel=2)
        return y
    
    def reverse(self, x):
        z = x
        for _ in range(self.T):
            z = x - self.step(z)
        return z
    
class RUet_module(nn.Module):
    def __init__(self,T=5,device='cpu'):
        super(RUet_module,self).__init__()
        self.model =UNet3D(in_channels=2,out_channels=2,is_segmentation=False,final_sigmoid=False).to(device)
        self.T = T
    def forward(self,x):
        return x+self.model(x)
    def reverse(self,x):
        z = x
        for _ in range(self.T):
            z = x-self.model(z)
        return z
    
    
def genNetwork(layer,N):
    return nn.ModuleList([layer for _ in range(N)])

def makeNetwork(opList, N):
    return genNetwork(nn.ModuleList(opList), N)


class MoDL_model():
    def __init__(self,metadata,device='cpu',num_layers=5, num_filters=64):
        self.metadata = metadata
        self.unrolls = metadata['num_unrolls']
        self.lamb = metadata['lamb']
        self.cg = metadata['cg']
        self.num_layers, self.num_filters= num_layers, num_filters
        self.device = device                   
        self._make_model()    

    def _make_model(self):
        self.RU = ResNet4(dims=3,num_channels=self.num_filters,T=5,num_layers=self.num_layers,device=self.device)
        self.CGM = CG_module(False,False,False,self.lamb,self.cg)
        self.network = makeNetwork([self.RU, self.CGM], self.unrolls) 
    def initialize(self,Sense,adj):
        self.CGM.initiate(Sense,adj)
    def forward(self,x):
        for ii in range(len(self.network)):
            for layer in self.network[ii]:
                x = layer.forward(x)
        return x
# network evaluator (forward/backward)
class UnrolledNetwork():
    def __init__(self, model, xtest, memlimit, loss=None, setupFlag=True, ckptFlag=0, device='cpu',dtype=torch.float32):
        super(UnrolledNetwork, self).__init__()
        self.model = model
        self.network = self.model.network
        self.xtest = xtest
        self.memlimit = memlimit
        self.gpu_device = device
        self.dtype = dtype

        # setup hybrid checkpointing
        self.meldFlag = True # default
        self.cpList = [-1] # default
        
        # setup loss function
        self.unsuperFlag = False
        if loss is None:
            self.lossFunc = lambda x,truth,device : torch.mean((x-truth)**2)
        else:
            self.lossFunc = loss
            
        if setupFlag: 
            self.setup()
        
        if ckptFlag != 0:
            N = len(self.network)
            self.cpList = np.sort(list(np.linspace(int(N-ckptFlag),1,int(ckptFlag), dtype=np.int32)))
        
    def setup(self):
        for p_ in self.network.parameters(): p_.requires_grad_(True)
            
        # compute storage memory of single checkpoint
        torch.cuda.empty_cache()
        startmem = torch.cuda.memory_cached(self.gpu_device)
        self.xtest = self.xtest.to(self.gpu_device)
        endmem = torch.cuda.memory_cached(self.gpu_device)
        mem3 = (endmem - startmem) / 1024**2
        print('Memory per checkpoint: {0:d}MB'.format(int(mem3)))
        torch.cuda.empty_cache()
        

        # test memory requirements (offset + single layer)
        torch.cuda.empty_cache()
        startmem = torch.cuda.memory_cached(self.gpu_device)
        
        x = self.xtest
        for sub in self.network[0]:
            x = sub(x,device=self.gpu_device)
        if not self.unsuperFlag:
            loss = self.lossFunc(x, self.xtest, self.gpu_device)
        else:
            loss = self.lossFunc(x, self.gpu_device)
        loss.backward()

        endmem = torch.cuda.memory_cached(self.gpu_device)
        mem1 = (endmem - startmem) / 1024**2
        print('Memory per layer: {0:d}MB'.format(int(mem1)))
        
#         # test memory requirements (offset + two layer)
#         torch.cuda.empty_cache()
#         startmem = torch.cuda.memory_cached(self.gpu_device)
#         x = self.xtest
#         for layers in self.network[:2]:
#             for sub in layers:
#                 x = sub(x,device=self.gpu_device)
#         loss = self.lossFunc(x,self.xtest)
#         loss.backward()
#         endmem = torch.cuda.memory_cached(self.gpu_device)
#         mem2 = (endmem - startmem) / 1024**2
#         print('Memory per two layer: {0:d}MB'.format(int(mem2)))
#         torch.cuda.empty_cache()
        
        # assess how many checkpoints
        N = len(self.network)
        totalmem = (mem1) * N
        print('Total memory:', totalmem, 'MB')

        if totalmem > self.memlimit:
            print('Requires memory-efficient learning!')
            self.M = np.ceil(totalmem/self.memlimit)
            print('Checkpointing every:',int(self.M))
            self.cpList = list(range(1,int(N-self.M),int(self.M)))
            self.meldFlag = True
            print('Checkpoints:',self.cpList)
        else:
            self.cpList = [-1]
            self.M = 1
            self.meldFlag = False
            
            
    def evaluate(self, x0, interFlag=False, testFlag=True):
        # setup storage (for debugging)
        if interFlag:
            size = [len(self.network)] + [a for a in x0.shape]
            Xall = torch.zeros(size,device=self.gpu_device,dtype=self.dtype)
        else:
            Xall = None

        # setup checkpointing
        if self.cpList is not []:
            size = [len(self.cpList)] + [a for a in x0.shape]
            self.Xcp = torch.zeros(size,device=self.gpu_device,dtype=self.dtype)
        else:
            self.Xcp = None
        cp = 0

        for p_ in self.network.parameters(): p_.requires_grad_(not testFlag)

        x = x0
        for ii in range(len(self.network)):
            if cp < len(self.cpList) and ii == self.cpList[cp]:
                self.Xcp[cp,...] = x
                cp += 1

            for layer in self.network[ii]:
                x = layer.forward(x)

            if interFlag:
                Xall[ii,...] = x
        return x, self.Xcp, Xall
    
    
    def loss_eval(self, x0, truth, testFlag=True):
        with torch.no_grad():
            x, _, _ = self.evaluate(x0, testFlag=True)
            if not self.unsuperFlag:
                loss = self.lossFunc(x, truth, self.gpu_device)
            else:
                loss = self.lossFunc(x, self.gpu_device)
            return x, loss

    
    def differentiate(self,xN,qN,interFlag=False):
        if interFlag:
            size = [len(self.network)] + [a for a in xN.shape]
            X = torch.zeros(size, device=self.gpu_device,dtype=self.dtype)
        else:
            X = None

        for p_ in self.network.parameters(): p_.requires_grad_(True) 

        # checkpointing flag
        cp = len(self.cpList)-1

        xkm1 = xN
        qk = qN
        for ii in range(len(self.network)-1,-1,-1):
            # reverse sequence
            with torch.no_grad():
                
                if interFlag:
                    X[ii,...] = xkm1

                # checkpointing
#                 print(cp,ii,self.cpList[cp])
                if cp >= 0 and ii == self.cpList[cp]:
#                     print('Using Checkpoint:',ii)
                    xkm1 = self.Xcp[cp,...]
                    cp -= 1

                # calculate inverse
                else:
                    for jj in range(len(self.network[ii])-1,-1,-1):
#                         print('Using Reverse:',ii)
                        layer = self.network[ii][jj]
                        xkm1 = layer.reverse(xkm1)

            # forward sequece
            xkm1 = xkm1.detach().requires_grad_(True)
            xk = xkm1
            for layer in self.network[ii]:
                xk = layer.forward(xk)

            # backward call
            xk.backward(qk, retain_graph=True)
            with torch.no_grad():
                qk = xkm1.grad
        return X
    
    
    def forward(self, x0, truth, interFlag=False, testFlag=False):
        # memory-efficient learned design 
        if self.meldFlag:
#             print("haha")
            # evaluate network
            with torch.no_grad():
                print("Hi Ke, How are you!")
                xN,Xcp,Xforward = self.evaluate(x0, interFlag=interFlag, testFlag=testFlag)
        
            # evaluate loss
            xN = xN.detach().requires_grad_(True)
            
            # evaluate loss function
            if not self.unsuperFlag:
                loss = self.lossFunc(xN, truth)
            else:
                loss = self.lossFunc(xN)
            
            loss.backward()
            qN = xN.grad

            # reverse-mode differentiation
            Xbackward = self.differentiate(xN,qN,interFlag=interFlag)
            
        # standard backpropagation
        else:
            # evaluate network
            xN,Xcp,Xforward = self.evaluate(x0, interFlag=interFlag, testFlag=False)
            # evaluate loss function
            if not self.unsuperFlag:
                loss = self.lossFunc(xN, truth)
            else:
                loss = self.lossFunc(xN, self.gpu_device)
            # reverse-mode differentiation
            loss.backward()
            Xbackward = None
            Xforward = None
            
        return xN, loss, Xforward, Xbackward # returned for testing/debugging purposes
class UnrolledNetwork_clean():
    def __init__(self, model,loss=None,device='cpu',flag_meld=True):
        super(UnrolledNetwork_clean, self).__init__()
        self.model = model
        self.network = self.model.network
        self.gpu_device = device

        # setup hybrid checkpointing
        self.meldFlag = flag_meld # default
        
        # setup loss function
        self.lossFunc = loss
            
    def evaluate(self, x0):
        # setup storage (for debugging)
 
        Xall = None
        for p_ in self.network.parameters(): p_.requires_grad_(True)

        x = x0
        for ii in range(len(self.network)):
            for layer in self.network[ii]:
                x = layer.forward(x)
        return x
    
    
    def loss_eval(self, x0, truth):
        with torch.no_grad():
            x, _, _ = self.evaluate(x0)

            loss = self.lossFunc(x, truth)
            return x, loss

    
    def differentiate(self,xN,qN):

        X = None

        for p_ in self.network.parameters(): p_.requires_grad_(True) 

        # checkpointing flag
#         cp = len(self.cpList)-1

        xkm1 = xN
        qk = qN
        for ii in range(len(self.network)-1,-1,-1):
            # reverse sequence
            with torch.no_grad():
                for jj in range(len(self.network[ii])-1,-1,-1):
                    layer = self.network[ii][jj]
                    xkm1 = layer.reverse(xkm1)
            
            # forward sequece
            xkm1 = xkm1.detach().requires_grad_(True)
            xk = xkm1
            for layer in self.network[ii]:
                xk = layer.forward(xk)

            # backward call
            xk.backward(qk, retain_graph=True)
            with torch.no_grad():
                qk = xkm1.grad
        return qk
    
    
    def forward(self, x0, truth):
        # memory-efficient learned design 
        if self.meldFlag:
            # evaluate network
            with torch.no_grad():
                xN = self.evaluate(x0)
        
            # evaluate loss
            xN = xN.detach().requires_grad_(True)
            
            # evaluate loss function
            loss = self.lossFunc(xN, truth)
            loss.backward()
            qN = xN.grad

            # reverse-mode differentiation
            Xbackward = self.differentiate(xN,qN)
            
        # standard backpropagation
        else:
            # evaluate network
            xN= self.evaluate(x0)
            # evaluate loss function

            loss = self.lossFunc(xN, self.gpu_device)
            # reverse-mode differentiation
            loss.backward()
            Xbackward = None
            Xforward = None
            
        return xN, loss,Xbackward # returned for testing/debugging purposes