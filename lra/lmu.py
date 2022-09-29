import numpy as np

import torch
from torch import nn
from torch import fft
from torch.nn import functional as F



 
    
class LMUFFT_nccl(nn.Module):


    def __init__(self, hidden_size, seq_len, rnn_dropout = 0.5, num_head = 2,transformer_dim = 64, fold = 4):

        super(LMUFFT_nccl, self).__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn_dropout = rnn_dropout
        self.num_head = num_head
        self.dim = transformer_dim
        self.fold = fold
        
        
        self.weights_fft = nn.Parameter(torch.empty(self.seq_len//2+1, self.hidden_size,2))
        nn.init.kaiming_uniform_(self.weights_fft, mode='fan_in', nonlinearity='relu')
        
        
        self.weights_v = nn.Parameter(torch.rand(1,self.hidden_size))
        
        self.weights_u = nn.Parameter(torch.rand(1,self.hidden_size))
        
        
        self.tiny_conv_fft =  torch.nn.Conv1d(in_channels = 2 , out_channels = self.hidden_size*2, kernel_size = 1, padding=  0, groups = 2)# self.num_head)
        
        # self.weights_fft0 = nn.Parameter(torch.rand(1,self.hidden_size))
        
        weight_range = torch.tensor([i*1.0 for i in range(self.seq_len//2+1)]).reshape(1,1,self.seq_len//2+1)#.to('cuda')
        # print(self.weight_range.shape )
        # input()
        self.weight_range = nn.Parameter(torch.empty(1,2,self.seq_len//2+1))
        nn.init.kaiming_uniform_(self.weight_range, mode='fan_in', nonlinearity='relu')
        
        
        # self.register_buffer("weight_range", weight_range) # [memory_size, seq_len + 1]
        
#         #############3
        self.weight_list = [1 + self.weights_u + self.weights_v*(i) for i in range(self.seq_len) ]
        
        self.weight_ori = torch.cat(self.weight_list,dim = 0)
          
#         # print(self.weight_ori.type, self.weight_ori.shape)
#         # input()
#         self.weight_used = torch.fft.rfft(self.weight_ori, n = self.seq_len,dim = -2)
#         # # print(self.weight_ori.shape, self.weights_fft.shape)
#         # # input()
#         # print(self.weight_used.type, self.weight_used.shape)
#         # input()
#         self.weight_used = torch.view_as_real(self.weight_used)
#         #####################



        # # self.weights_fft = torch.fft.rfft(self.weight_ori, n = self.seq_len)
        # # print(self.weight_ori.shape, self.weights_fft.shape)
        # # input()
        # self.weights_fft = torch.view_as_real(self.weights_fft)
     
        self.index_set =   torch.randperm(self.seq_len//2+1)
        self.index_set = self.index_set[(self.seq_len//2+1)//self.fold:]
           
        
        self.tiny_conv_linear =  torch.nn.Conv1d(in_channels = self.hidden_size*2 , out_channels = self.hidden_size, kernel_size = 1, padding=  0, groups = 1)# self.num_head)
        
        
        self.lmudropout = torch.nn.Dropout(p=self.rnn_dropout)#dropout_prob
        self.bn_1 = nn.BatchNorm1d(self.seq_len)#,affine = False)
        
        self.lmudropout2 = torch.nn.Dropout(p=self.rnn_dropout)#dropout_prob
        self.bn_2 = nn.BatchNorm1d(self.seq_len)#,affine = False)
                

    def forward(self, x):
  
 
        # [batch_size, seq_len, hidden_size]
        fft_u = fft.rfft(x, n =  self.seq_len, axis = -2)#,norm = 'ortho')
        fft_u = torch.view_as_real(fft_u)
            
            
        # self.weight_list = [1 + self.weights_v + self.weights_v*(i) for i in range(self.seq_len) ]
        
        # self.weight_ori = torch.cat(self.weight_list,dim = 0)
        # print('1', self.weight_ori.shape)
        # input()
        # self.weight_used self.weights_fft#= self.tiny_conv_fft(self.weight_range).permute(0,2,1).reshape(1,self.seq_len//2+1,self.hidden_size,2)
        # print('2', self.weight_ori.shape)
        # input() 
        # print(self.weight_ori.type, self.weight_ori.shape)
        # input()
        # self.weight_used = torch.fft.rfft(self.weight_ori, n = self.seq_len,dim = -2)
        # # print(self.weight_ori.shape, self.weights_fft.shape)
        # # input()
        # print(self.weight_used.type, self.weight_used.shape)
        # input()
        # self.weight_used = torch.view_as_real(self.weight_used)
            
        self.weight_used = self.weights_fft.unsqueeze(0)
        temp_real = fft_u[...,0]*self.weight_used[...,0] - fft_u[...,1]*self.weight_used[...,1]
        temp_imag = fft_u[...,0]*self.weight_used[...,1] + fft_u[...,1]*self.weight_used[...,0]
        
                

        out_ft = torch.cat([temp_real.unsqueeze(-1),temp_imag.unsqueeze(-1)],dim =  -1)
        out_ft = torch.view_as_complex(out_ft) 
        
        
        # out_ft[:,self.index_set,:] = 0
        m = fft.irfft(out_ft, n =  self.seq_len, axis = -2)#,norm = 'ortho')
                      
                      
        input_h = torch.cat((m, x), dim = -1) 
        h =  self.tiny_conv_linear(input_h.permute(0,2,1)).permute(0,2,1)


        
        h = self.lmudropout(F.elu(self.bn_1(h)))
        
        
        m = self.lmudropout2(F.elu(self.bn_2(m)))
 
        return   h,m
    
    
    
    
    

    
    
class LMUFFT_nccl1(nn.Module):


    def __init__(self, hidden_size, seq_len, rnn_dropout = 0.5, num_head = 2):

        super(LMUFFT_nccl1, self).__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn_dropout = rnn_dropout
        self.num_head = num_head
        
        # self.weights_fft = nn.Parameter(torch.randn(self.seq_len//2+1, self.hidden_size,2))
        
        self.weights_fft = nn.Parameter(torch.zeros(self.seq_len//2+1, self.hidden_size,2))
        # nn.init.kaiming_uniform_(self.weights_fft, mode='fan_in', nonlinearity='relu')
        # self.weights_fft = nn.Parameter(torch.randn(self.num_head, self.seq_len//2+1, self.hidden_size // self.num_head,2))
        # self.weights_fft = nn.Parameter(torch.rand(1, self.seq_len//2+1, self.hidden_size // self.num_head,2))
    
    
    
        self.kappa = 1
   
        # self.weights_fft1 = nn.Parameter(torch.rand(self.seq_len//2 + 1, self.kappa , 2)/np.sqrt(self.seq_len//2 + 1))
        # self.weights_fft2 = nn.Parameter(torch.rand(self.kappa, self.hidden_size, 2)/np.sqrt(self.hidden_size))
        
        
        
        
        
        
        # self.weights_fft1 = nn.Parameter(torch.rand(self.seq_len//2 + 1, self.kappa , 2))# /np.sqrt(self.seq_len))

# 
        self.weights_fft1 = torch.empty(self.seq_len//2 + 1, self.kappa , 2)
#         
        nn.init.kaiming_uniform_(self.weights_fft1, mode='fan_in', nonlinearity='relu')
        self.weights_fft1 = nn.Parameter(self.weights_fft1)


        # self.weights_fft2 = nn.Parameter(torch.rand(self.kappa, self.hidden_size, 2)/np.sqrt(self.hidden_size))

        # self.weights_fft2 = nn.Parameter(torch.rand(self.kappa, self.hidden_size, 2) )#/np.sqrt(self.hidden_size))
# 
        self.weights_fft2 = torch.empty(self.kappa, self.hidden_size, 2)
#         
        nn.init.kaiming_uniform_(self.weights_fft2, mode='fan_in', nonlinearity='relu')
        self.weights_fft2 = nn.Parameter(self.weights_fft2)
        
        
        
        
        
        self.tiny_conv_linear =  torch.nn.Conv1d(in_channels = self.hidden_size*2 , out_channels = self.hidden_size, kernel_size = 3, padding=  1, groups = 1)# self.num_head)
        
        
        self.lmudropout = torch.nn.Dropout(p=self.rnn_dropout)#dropout_prob
        self.bn_1 = nn.BatchNorm1d(self.seq_len)#,affine = False)
        
        

    def combine_heads_local(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.hidden_size*2)
        return X
    
#     def combine_heads_local(self, X):
#         X = X.permute(3, 2,0,1)
#         X = X.reshape(X.size(0), X.size(1), X.size(2)*X.size(3))
#         return X

#     def split_heads_local(self, X):
#         X = X.reshape(X.size(0), X.size(1), X.size(2)//self.num_head, self.num_head)
#         X = X.permute(2, 3,0,1)
#         return X


    def forward(self, x):
  
        # batch_size, num_head, seq_len, input_size = x.shape
    
        # print(x.shape)
        # torch.Size([8, 8, 2048, 64])
        # input()
        
        # x = x.reshape(batch_size*num_head, seq_len, input_size)
        # fft_input = x.permute(0,1, 3, 2)
        # fft_input  = self.split_heads(x)
        
        
        # x = self.combine_heads_local(x)
        # fft_input =  # [batch_size, 1, seq_len]
        fft_u = fft.rfft(x, n =  self.seq_len, axis = -2)#,norm = 'ortho')
        # fft_u = fft.rfft2(x, dim=(- 1, - 2),norm = 'ortho')
        fft_u = torch.view_as_real(fft_u)
                
            
        # print(fft_u.shape, self.weights_fft.unsqueeze(0).shape)
        
        # torch.Size([64, 2048, 33, 2]) torch.Size([1, 64, 1025, 2])
        
        # input()
        self.weight_used = torch.einsum("lkt,kdt->ldt",self.weights_fft1,self.weights_fft2).unsqueeze(0)#self.weights_fft
      
        temp_real = fft_u[...,0]*self.weight_used[...,0] - fft_u[...,1]*self.weight_used[...,1]
        temp_image = fft_u[...,0]*self.weight_used[...,1] + fft_u[...,1]*self.weight_used[...,0]
        
        # print(fft_u.shape,self.weights_fft.unsqueeze(0).unsqueeze(0).shape)
        # input()
        # temp_real = fft_u[:,:,:,:,0]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,0] - fft_u[:,:,:,:,1]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,1]
        # temp_image = fft_u[:,:,:,:,0]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,1] + fft_u[:,:,:,:,1]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,0]
        

        out_ft = torch.cat([temp_real.unsqueeze(-1),temp_image.unsqueeze(-1)],dim =  -1)
        out_ft = torch.view_as_complex(out_ft) # [batch_size, memory_size, seq_len+1]
        m = fft.irfft(out_ft, n =  self.seq_len, axis = -2)#,norm = 'ortho')
        # m = fft.irfft2(out_ft,  dim=(- 1, - 2),norm = 'ortho')#s = [seq_len,input_size],
        # m = m.permute(0,1, 3, 2) # [batch_size, seq_len, memory_size]
        # m = m.reshape(x.shape[0],fft_input.shape[1],fft_input.shape[3],fft_input.shape[2])
        # m = self.combine_heads(m)
       
        # print(m.shape,x.shape,out_ft.shape,fft_u.shape,temp_real.shape,temp_image.shape)
        # input()
        
        input_h = torch.cat((m.squeeze(0), x), dim = -1) # [batch_size, seq_len, memory_size + input_size]
        
        # print(input_h.shape)
        # input()
        # input_h = input_h.reshape(batch_size , num_head,  seq_len, input_size)
        # input_h = self.combine_heads_local(input_h)
        h =  self.tiny_conv_linear(input_h.permute(0,2,1)).permute(0,2,1)


        
        h = self.lmudropout(F.elu(self.bn_1(h)))
        # h = self.lmudropout((self.bn_1(h)))
        
        # h = h.reshape(batch_size , num_head,  seq_len, input_size)
        
        
        # print(batch_size, num_head, seq_len, input_size , h.shape)
        # input()
        # h = h.reshape(x.size(0),x.size(1), x.size(2), x.size(3))
        # h = self.combine_heads(h)
        return h#*np.sqrt(self.seq_len  / 2) #h#*np.sqrt(self.seq_len  / 2)   m+m_ori# 
