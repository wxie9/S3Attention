import torch
import numpy as np
import time
import math
def dct_real(x_ori, norm='ortho',dim = -1):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    
    # x_ori = torch.from_numpy(x_ori)
    # if not torch.is_tensor(x_ori):
        # x_ori = torch.from_numpy(x_ori)
        
    t0 = [] 
    t0.append(time.time()) #0
    pi = 3.141592653589793
    x_shape = list(x_ori.shape)
    N = x_shape[dim]
    
    t0.append(time.time()) #1
    # Move the transformed dimesion to the last position.
    x_permute  = [i for i in range(len(x_shape))]
    x_permute[dim], x_permute[-1] = x_permute[-1], x_permute[dim]
    x_shape[dim], x_shape[-1] = x_shape[-1], x_shape[dim]
    
    
    t0.append(time.time()) #2
    x = x_ori.contiguous().permute(x_permute).float()
    
    
    t0.append(time.time()) #3
    x = x.reshape((-1,N))
    
    
    
    
    t0.append(time.time()) #4
    # DCT trasnform based on FFT
    
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=-1)
    
    # v = x
    
    
    
    
    t0.append(time.time()) #5
    Vc = torch.fft.fft(v.float(),dim = -1, norm = norm)
    # V = Vc.real * W_r - Vc.imag
    
    
    t0.append(time.time()) #6
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * pi / (2 * N)
    
    
    
    t0.append(time.time()) #7
    W_r = torch.cos(k)
    
    
    
    t0.append(time.time()) #8
    W_i = torch.sin(k)
    
    
    
    t0.append(time.time()) #9
    V = Vc.real * W_r - Vc.imag * W_i
    
    
    
    
    t0.append(time.time()) #10
#     if norm == 'ortho':
#         V[:, 0]  /= math.sqrt(N) * 2
#         V[:, 1:] /= math.sqrt(N / 2) * 2
    
    t0.append(time.time()) #11
    V = 2 * V.reshape(x_shape).permute(x_permute)
    
    
    
    
    t0.append(time.time()) #12
    
    
#     for i in range(len(t0)-1):
#         print(f"from {i} to {i+1}: ", t0[i+1]-t0[i])
        
#     print(f'total time: ', t0[-1]-t0[0])
#     print("*"*100)
    return V.to(x)



def idct_real(X_ori, norm='ortho',dim = -1):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """
    
    # if not torch.is_tensor(X_ori):
        # X_ori = torch.from_numpy(X_ori)
    # X_ori = torch.from_numpy(X_ori)
    pi = 3.141592653589793
    X_shape = list(X_ori.shape)
#     X = X.float()
    N = X_shape[dim]
    
    
    
    # Move the transformed dimesion to the last position.
    X_permute  = [i for i in range(len(X_shape))]
    X_permute[dim], X_permute[-1] = X_permute[-1], X_permute[dim]
    X_shape[dim], X_shape[-1] = X_shape[-1], X_shape[dim]
    X = X_ori.contiguous().permute(X_permute).float()
    X_v = X.reshape((-1,N)) / 2
    
    
    

#     X_v = X.contiguous().view(-1, N) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] *pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=-1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(-1), V_i.unsqueeze(-1)], dim=-1)
    V = torch.view_as_complex(V)
    

    v = torch.fft.ifft(V, dim = -1)

    x = v.new_zeros(v.shape)

    # x[:, ::2] += v[:, :N - (N // 2)]
    # x[:, 1::2] += v.flip([1])[:, :N // 2]

    
    x = x.reshape(X_shape).permute(X_permute)
           
    return x.to(X_ori)




def dct(x, norm=None,dim = -1):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    pi = 3.141592653589793
    x_shape = x.shape
    x = x.float()
    N = x_shape[dim]
#     total_N = 
#     print(N,x_shape,x.contiguous().shape)
    x = x.contiguous().view(-1, N)
#     print('x shape:', x.shape)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=-1)
#     print('v shape:', v.shape)
    Vc = torch.fft.fft(v.float(),dim = -1)
#     print('Vc shape:', Vc.shape)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
#     print(x.shape,Vc.shape)
    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V.half()


def idct(X, norm=None,dim = -1):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """
    pi = 3.141592653589793
    x_shape = X.shape
    X = X.float()
    N = x_shape[dim]

    X_v = X.contiguous().view(-1, N) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] *pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=-1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(-1), V_i.unsqueeze(-1)], dim=-1)
    V = torch.view_as_complex(V.float())
    

    v = torch.fft.ifft(V, dim = -1)
#     print('asdfasdfasdfasdfasdfdsafsadfasdfsa')
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.half().view(*x_shape)



class LinearDCT(torch.nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features,dim = -1, type = 'dct', norm='ortho', bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        self.dim = dim
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct_real(I, norm=self.norm, dim = self.dim).data.t()
        elif self.type == 'idct':
            self.weight.data = idct_real(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!