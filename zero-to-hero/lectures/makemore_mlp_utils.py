import torch

SEED = 2147483647
generator = torch.Generator().manual_seed(SEED)

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):

        self.out = x @ self.weight
        if self.bias:
            self.out += self.bias
        
        return self.out
    
    def parameters(self):
        if self.bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]
        
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.dim = dim
        
        # Train-able parameters.
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
        # Buffers
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)

    def __call__(self, x):

        if self.training:
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True)
        else:
            xmean = self.mean
            xvar = self.var

        xhat = (x-xmean)/(xvar+self.eps)**0.5
        self.out = self.gamma*xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.mean = self.momentum*xmean + (1-self.momentum)*self.mean
                self.var = self.momentum*xvar + (1-self.momentum)*self.var

        return self.out
    
    def parameters(self):

        return [self.gamma, self.beta]
    
class Tanh:

    def __init__(self):
        pass

    def __call__(self, x):
            
            self.out = x.tanh()
            return self.out
    
    def parameters(self):
        return []