import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np



import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self,dim,max_seq_len=197, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.,dp_rank = 8,mustInPos = None):
        
        super().__init__()
        
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        
        self.seq_len = max_seq_len
        self.dp_rank = dp_rank
        self.fold = 8
        self.fourierSampling = 16
        
        self.weights_fft = nn.Parameter(torch.empty(self.seq_len//2+1,dim,2))
        # self.weights_fft = nn.Parameter(torch.empty(32,dim,2))
        nn.init.kaiming_uniform_(self.weights_fft, mode='fan_in', nonlinearity='relu')
        
        self.tiny_conv_linear =  torch.nn.Conv1d(in_channels = dim*2 , out_channels = dim, kernel_size = 3,padding=  1, groups = 1)
    
        self.dropout1 = torch.nn.Dropout(p=attn_drop)
        self.bn_1 = nn.BatchNorm1d(self.seq_len)

         
        self.index_set_right =   torch.randperm(self.head_dim)
        if self.dp_rank <self.head_dim:
            self.index_set_right = self.index_set_right[:self.dp_rank]

        self.index_set_left =   torch.randperm(self.seq_len)
        if self.dp_rank <self.seq_len:
            self.index_set_left = self.index_set_left[:self.dp_rank]
            
            
            
        self.indx_set_Fourier =   torch.randperm(self.seq_len//2+1)
        if self.fourierSampling <self.seq_len:
            self.indx_set_Fourier = self.indx_set_Fourier[:self.fourierSampling]



        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)
        self.ln_pre = nn.LayerNorm(dim)
        
        
        
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_heads*self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_heads, self.head_dim)
        X = X.transpose(1, 2)
        return X
        
    def forward(self, x):
        x0 = x
        B, N, C = x.shape
        
        if C//self.fold >0:
            u = x.reshape(B,N,self.fold,C//self.fold)
            u = torch.mean(u,dim = -1)
        #### Fourier Convolution ####
            fft_u = torch.fft.rfft(u, n = self.seq_len, axis = -2,norm = 'ortho')
            fft_u = torch.view_as_real(fft_u)
            fft_u = fft_u.repeat(1,1,C//self.fold,1)
        else:
            fft_u = torch.fft.rfft(x, n = self.seq_len, axis = -2,norm = 'ortho')
            fft_u = torch.view_as_real(fft_u)


        # weights_fft = torch.cat((self.weights_fft,torch.zeros(self.seq_len//2+1 - self.weights_fft.shape[0],self.weights_fft.shape[1],self.weights_fft.shape[2]).to('cuda')))
        weight_used =self.weights_fft.unsqueeze(0)

        '''we may also use low-rank fft matrix'''
        # weight_used = torch.einsum("lkt,kdt->ldt",self.weights_fft1,self.weights_fft2).unsqueeze(0)#self.weights_fft

        temp_real = fft_u[...,0]*weight_used[...,0] - fft_u[...,1]*weight_used[...,1]
        temp_imag = fft_u[...,0]*weight_used[...,1] + fft_u[...,1]*weight_used[...,0]


        out_ft1 = torch.cat([temp_real.unsqueeze(-1),temp_imag.unsqueeze(-1)],dim =  -1)
        # print(out_ft.shape,len(self.indx_set_Fourier),x.shape)
        # out_ft1 = torch.zeros(out_ft.shape).to('cuda')
        # out_ft1[:,self.indx_set_Fourier,:,:] = out_ft[:,self.indx_set_Fourier,:,:]
        # out_ft[:,self.indx_set_Fourier,:,:] = (out_ft[:,self.indx_set_Fourier,:,:]*0).detach()
        # out_ft = out_ft*0
        out_ft1 = torch.view_as_complex(out_ft1)
        
        
        m = torch.fft.irfft(out_ft1, n =  self.seq_len, axis = -2,norm = 'ortho')

        input_h = torch.cat((m, x), dim = -1)



        h =  self.tiny_conv_linear(input_h.permute(0,2,1)).permute(0,2,1)


        x = self.dropout1(F.elu(self.bn_1(h)))

        
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
    
        
        
        #### left part ####
        if self.dp_rank <= self.seq_len:
            k1 = k[:,:,self.index_set_left,:]
            v1 = v[:,:,self.index_set_left,:]
        else:
            k1 = k
            v1 = v



        dots = q @ k1.transpose(-1,-2)
        dots = dots / math.sqrt(self.head_dim)
        attn = nn.functional.softmax(dots,dim=-1)
        attn = self.attn_drop(attn)

        #### right part ####
        q2 = q.transpose(-1,-2)
        if self.dp_rank <= self.head_dim:

            k2 = k[:,:,:,self.index_set_right]
            v2 = v[:,:,:,self.index_set_right]
        else:
            k2 = k
            v2 = v
  
        dots_r = q2 @ k2
        dots_r = dots_r / math.sqrt(self.seq_len)
        attn_r = nn.functional.softmax(dots_r,dim=-1).transpose(-1,-2)
        attn_r = self.attn_drop(attn_r)

        X = self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,v1))))/2 + self.split_heads(self.ln_2(self.combine_heads(torch.matmul(v2,attn_r))))/2
      
        x = X.transpose(1, 2).reshape(B, N, C)

        x = self.proj_drop(x)
        return x #+ x0
    
    



class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None
    
    def get_emb(self,sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        # print(tensor.shape)
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
        
    

    
class DataEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, embed_type='fixed', freq='h', dropout=0.1,seq_len = 96):
        super(DataEmbedding, self).__init__()
        
        self.tiny_conv_linear =  torch.nn.Conv1d(in_channels =  enc_in, out_channels = d_model*1, kernel_size = 1,padding=  0, groups = 1)
      

        self.dropout = nn.Dropout(p=dropout)        
        self.posembeding =  PositionalEncoding1D(channels = d_model)


    def forward(self, x, x_mark = None):
        x_mean = torch.mean(x,dim = -1,keepdim = True)
        B,H,N = x.shape
     
        x1 = self.dropout(self.tiny_conv_linear(x.permute(0,2,1)).permute(0,2,1) ) 
        return self.posembeding(x1)+ x1
                        
                        
    
    
class SKTLinear(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(SKTLinear, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.embd_dim = configs.enc_in
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len

        
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,configs.dropout,configs.seq_len)
                
        
        self.encoder =  nn.ModuleList(
            [    
                Attention(dim = configs.d_model, max_seq_len=configs.seq_len, num_heads=configs.n_heads, qkv_bias=False, attn_drop=configs.dropout, proj_drop=configs.dropout,dp_rank = 8,mustInPos = None)  
                 for l in range(configs.e_layers)
            ],
        )

        self.Postmlp=  nn.ModuleList(
            [    
                nn.Linear(configs.d_model,self.embd_dim)
                 for l in range(configs.e_layers)
            ],
        )
        
        self.fourierExtrapolation = fourierExtrapolation(inputSize = configs.seq_len,n_harm = self.pred_len,n_predict = self.pred_len)
        

        
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        B1,H1,C1 = x_enc.shape
        # dec_out = []
        for i in range(len(self.encoder)):
            attn_layer,post = self.encoder[i],self.Postmlp[i]
            
            if i == 0:
                tmp_mean = torch.mean(x_enc[:,:,:],dim = 1,keepdim = True)#.detach()
                tmp_std = torch.sqrt(torch.var(x_enc[:,:,:],dim = 1,keepdim = True)+1e0)#.detach()
                x_enc = (x_enc - tmp_mean)/(tmp_std) 

                enc_out1 = self.enc_embedding(x_enc)
         
            enc_out1= attn_layer(enc_out1) + enc_out1 

             

     
        
        tmp1 = self.fourierExtrapolation.fourierExtrapolation(post(enc_out1[:,:self.seq_len,:]))
       
        # dec_out.append(tmp1[:,:,:].unsqueeze(-1))
        dec_out = tmp1[:,:,:].unsqueeze(-1)
 

        # dec_out = torch.mean(torch.cat(dec_out,axis = -1),axis = -1)
        # dec_out = dec_out[-1].squeeze(-1)
     
        output = (dec_out.reshape(B1,-1,C1))*(tmp_std)+tmp_mean 
        
        
        return output[:,-self.pred_len:,:], output[:,:self.seq_len,:]
    
    
    


class Exp_Main(object):
    def __init__(self, args):
        # super(Exp_Main, self).__init__(args)
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.batchIdx = torch.randperm(self.args.enc_in)

        
        
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = SKTLinear(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion =nn.MSELoss()#.L1Loss()#(beta = 1e-2)#nn.MSELoss()#nn.MSELoss()#L1Loss()##nn.MSELoss() nn.MSELoss()+
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs#/x_mean #+x_mean
                # batch_y = batch_y#/x_mean
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                inerr_reshuffle = 1#self.args.enc_in
                
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                while inerr_reshuffle >0:
                    inerr_reshuffle -= 1
                    model_optim.zero_grad()
                    # self.batchIdx = torch.randperm(self.args.enc_in)
                    # self.batchIdx = torch.randperm(self.args.enc_in)
                    # print(batch_x_mark.shape,batch_y_mark.shape,batchIdx)
                    # batch_x = batch_x[:,:,self.batchIdx]
                    # batch_y = batch_x[:,:,self.batchIdx]
                    # batch_x_mark = batch_x_mark[:,:,batchIdx]
                    # batch_y_mark =batch_y_mark[:,:,batchIdx]
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # print('dec_inp', dec_inp.shape)
                    # print('batch_x', batch_x.shape)
                    # print('batch_y',batch_y.shape)
                    # print('batch_x_mark',batch_x_mark.shape)
                    # print('batch_y_mark',batch_y_mark.shape)
                    # return
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y+torch.randn(outputs.shape).to('cuda')*0)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss =  criterion(outputs, batch_y)#+criterion(x_mean, batch_x)/2#/2#/2 +/2 #2 + 
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0 and inerr_reshuffle == 1:
                        # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs= outputs#+x_mean
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs#+x_mean
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
class fourierExtrapolation(nn.Module):
    def __init__(self,inputSize,n_harm = 8,n_predict = 96):
        super().__init__()
        self.n = inputSize
        self.n_harm = n_harm
        self.t = torch.arange(0, self.n)
        # self.t0 = self.t.unsqueeze(0).unsqueeze(-1).float().to('cuda')
        self.t0 = self.t.float().to('cuda')
        self.x_mean = torch.mean(self.t0)
        self.x_square = torch.mean((self.t0 - self.x_mean)*(self.t0 - self.x_mean))
        self.t0 = self.t.unsqueeze(0).unsqueeze(-1).float().to('cuda')

        
        self.f = torch.fft.fftfreq(self.n)              # frequencies

        self.indexes = list(range(self.n))
        # sort indexes by frequency, lower -> higher
        self.indexes.sort(key = lambda i: torch.absolute(self.f[i]))
        self.indexes = self.indexes[:1 + self.n_harm * 2]
        # self.indexes = torch.tensor(self.indexes).to('cuda')
        self.n_predict = n_predict
        self.t = torch.arange(0, self.n + self.n_predict)#.to('cuda')
        self.t1 = self.t.unsqueeze(0).unsqueeze(-1).float().to('cuda')
        # self.restored_sig = torch.zeros(self.t.shape).to('cuda')
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.f = self.f.unsqueeze(0).unsqueeze(-1).to('cuda')
        self.t = self.t.unsqueeze(0).unsqueeze(-1).to('cuda')
        self.t = self.t.unsqueeze(-1)
        
        self.g = self.f[:,self.indexes,:].permute(0,2,1).unsqueeze(1)
        self.decoder = torch.nn.Linear(96*2,96)
        self.phase_init = 2 * 3.1415 * self.g * self.t

    def linearfit(self,y,Lambda = 1e2):
    # H,C = x.shape
        B,H,C = y.shape
 
        y_mean = torch.mean(y,dim = 1,keepdim = True)#.repeat(B,1,1)

        b = torch.mean((self.t0 - self.x_mean)*(y-y_mean),dim = 1,keepdim = True) / (self.x_square+Lambda)

        return b.detach()#.unsqueeze(1)
        
        
   
        
        
    def fourierExtrapolation(self,x,notrend = True):

        # p = self.linearfit(x)

        
        x_notrend = x #- p * self.t0


        
        x_freqdom = torch.fft.fft(x_notrend,dim = -2)  # detrended x in frequency domain
  
        
        x_freqdom = torch.view_as_real(x_freqdom)
        x_freqdom = x_freqdom[:,self.indexes ,:,:]
        x_freqdom = torch.view_as_complex(x_freqdom)
        ampli = torch.absolute(x_freqdom) / self.n   # amplitude
        phase = torch.angle(x_freqdom)          # phase

        ampli = ampli.permute(0,2,1).unsqueeze(1)
        phase = phase.permute(0,2,1).unsqueeze(1)

        self.restored_sig = ampli * torch.cos(self.phase_init + phase)

        return torch.sum(self.restored_sig,dim = -1)# + p * self.t1 
    
    



def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    