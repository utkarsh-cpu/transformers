''' All the Layer which are fundamental'''


from comp_graph import Layer
import torch as th
from Operations import *
import math as mt



''' Linear Layer and Other Archtype'''
class Linear(Layer):
    def __init__(self,input_weight : int ,out_weight : int, bias: bool=True):
        super().__init__()
        self.input_weight=input_weight
        self.out_weight=out_weight
        self.weight= th.rand(out_weight,input_weight)
        if bias:
            self.bias=th.rand(out_weight)
    
    def __call__(self, input: th.Tensor):
        
        if self.bias is not None:
            super().__call__([input,self.weight,self.bias])
            return add(matmul(input,self.weight.T),self._bias)
        else:
            super().__call__([input,self.weight])
            return matmul(input,self.weight.T)
    
    def info(self):
        if self.bias is not None:
            return [self.weight,self.bias,self.input_weight,self.out_weight]
        else:
            return [self.weight,self.input_weight,self.out_weight]

class LazyLinear(Layer):
    def __init__(self,out_weight : int, bias: bool=True):
        super().__init__()
        self.out_weight=out_weight
        if bias:
            self.bias=th.rand(out_weight)
    
    def __call__(self, input: th.Tensor):
        self.weight=th.rand(self.out_weight,input.size()[-1])
        if self.bias is not None:
            super().__call__([input,self.weight,self.bias])
            return add(matmul(input,self.weight.T),self._bias)
        else:
            super().__call__([input,self.weight])
            return matmul(input,self.weight.T)
    
    def info(self):
        if self.bias is not None:
            return [self.weight,self.bias,self.input_weight,self.out_weight]
        else:
            return [self.weight,self.input_weight,self.out_weight]


'''Dropout Layer'''
##Inverse Dropout

class Dropout(Layer):
    def __init__(self,p: float):
        super().__init__()
        self.p=p

    def __call__(self,x):
        super().__call__([x])
        tmp=th.mul(th.Tensor.bernoulli_(x.dim(),self.p),(1/(1-self.p)))
        return mul(tmp,x)
    
''' Layer Normalization'''
class LayerNorm(Layer):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = (th.ones(d_model))
        self.beta = (th.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        super().__call__([self.gamma,self.beta,x])
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = add(x,-mean)/th.sqrt(var + self.eps)
        out = add(mul(self.gamma,out),self.beta)
        return out


''' Activation Function Layer Section'''
class RElu(Layer):
    def __init__(self,inplace:bool = False):
        super().__init__()
        self.inplace=inplace
    
    def __call__(self, input):
        super().__call__([input])
        if self.inplace:
            input=mul(input,(input>0))
            return input
        else:
            tmp=mul(input,(input>0))
            return tmp

class Sigmoid(Layer):
    def __init__(self,inplace:bool = False):
        super().__init__()
        self.inplace=inplace
    
    def __call__(self, input):
        super().__call__([input])
        if self.inplace:
            input=pow(add(1,th.exp(-input)),-1)
            return input
        else:
            tmp=pow(add(1,th.exp(-input)),-1)
            return tmp

class LRElu(Layer):
    def __init__(self,inplace:bool = False):
        super().__init__()
        self.inplace=inplace
    
    def __call__(self, input,alpha:float):
        super().__call__([input])
        if self.inplace:
            input=th.maximum(input,mul(input*alpha))
        else:
            tmp=th.maximum(input,mul(input*alpha))
            return tmp

class Elu(Layer):
    def __init__(self,inplace:bool = False):
        super().__init__()
        self.inplace=inplace
    
    def __call__(self, input,alpha:float):
        super().__call__([input])
        if self.inplace:
            input=th.maximum(input,mul((th.exp(input)-1),alpha))
        else:
            tmp=th.maximum(input,pow(add(1,th.exp(-input)),-1))
            return tmp

class HardSigmoid(Layer):
    def __init__(self,inplace:bool = False):
        super().__init__()
        self.inplace=inplace
    
    def __call__(self, input):
        super().__call__([input])
        if self.inplace:
            input=add(mul(input,pow(6,-1),0.5))*(add(th.logical_and(input<=3,input>=-3)),mul(input,th.logical_and(input>=3,input<=-3)))
        else:
            tmp=add(mul(input,pow(6,-1),0.5))*(add(th.logical_and(input<=3,input>=-3)),mul(input,th.logical_and(input>=3,input<=-3)))
            return tmp

class Tanh(Layer):
    def __init__(self,inplace:bool = False):
        super().__init__()
        self.inplace=inplace
    
    def __call__(self, input):
        super().__call__([input])
        if self.inplace:
            input=th.tanh(input)
            return input
        else:
            tmp=th.tanh(input)
            return tmp