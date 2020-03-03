import torch
import math
import numpy as np
import scipy.special

class IveFunction(torch.autograd.Function):
    '''alternative implementation of scipy.special.ive for pytorch, allowing backward.
    Reference:
        https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/ops/ive.py
    '''

    @staticmethod
    def forward(self, v, z):
        self.save_for_backward(z)
        self.v = v
        # scipy.special.ive requires double type
        z_cpu = z.data.cpu().numpy().astype(np.float64)

        output = scipy.special.ive(v, z_cpu)
        return torch.DoubleTensor(output).type(z.dtype).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1].double()
        return None, grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z)

class LogVmfConstFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, m, k):
        self.save_for_backward(k)
        self.m = m
        dtype = k.dtype
        device = k.device

        # ive returns small values out of float range
        k_np = k.data.cpu().numpy().astype(np.float64)
        ive_np = scipy.special.ive(m // 2 - 1, k_np)
        log_ive = torch.log(torch.from_numpy(ive_np))

        cm_k = (m // 2 - 1) * torch.log(k) \
            - m // 2 * math.log(2 * math.pi) \
            - log_ive.to(device).type(dtype)

        # k < 1 makes 'inf' in cm_k
        cm_k[cm_k == float('inf')] = cm_k[cm_k != float('inf')].max().item()

        return cm_k
    
    @staticmethod
    def backward(self, grad_output):
        k = self.saved_tensors[-1]
        m = self.m
        dtype = k.dtype
        device = k.device
        
        # ive returns small values out of float range
        k_np = k.data.cpu().numpy().astype(np.float64)
        grad_np = -1 * scipy.special.ive(m // 2, k_np) / scipy.special.ive(m // 2 - 1, k_np)

        k_grad = torch.from_numpy(grad_np).to(device).type(dtype)

        # k < 1 makes 'nan' in k_grad
        k_grad[k_grad != k_grad] = k_grad[k_grad == k_grad].min().item()
        
        # first term for m and m will not require grad
        return None, grad_output * k_grad

class LogVmfConstApproxFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, m, k):
        self.save_for_backward(k)
        self.m = m
        dtype = k.dtype
        device = k.device

        cm_k = ((m//2 + 1)**2 + k**2).sqrt() \
            - (m//2 - 1) * (m//2 - 1 + ((m//2 +1)**2 + k**2).sqrt()).log()

        return cm_k
    
    @staticmethod
    def backward(self, grad_output):
        k = self.saved_tensors[-1]
        m = self.m
        dtype = k.dtype
        device = k.device
        
        k_grad = -1 * k / (m//2 - 1 + ((m//2 +1)**2 + k**2).sqrt())
        
        # first term for m and m will not require grad
        return None, grad_output * k_grad

ive = IveFunction.apply
log_vmf_c = LogVmfConstFunction.apply
log_vmf_c_approx = LogVmfConstApproxFunction.apply
