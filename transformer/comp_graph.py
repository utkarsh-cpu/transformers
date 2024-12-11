'''
        Computational Graph from Scratch        
        --> Backend for storing various mathematical expression
        --> Important for gradient descent and backpropagation
'''

import torch as th

class Module:
    def __init__(self):
        graph=ComputationalGraph()
        graph()
    
    def __call__(self):


class ComputationalGraph:
    def __init__(self):
        self.operations=[]
        self.layer=[]
        self.inputs=[]
        self.parameters=[]

    def __call__(self):
        global _default_graph
        _default_graph=self

class Operation:
    def __init__(self,input_nodes):
        self.input_nodes=input_nodes
        self.consumers=[]
        for input_node in input_nodes:
            input_node.consumers.append(self)
        
        _default_graph.operations.append(self)
    
    def __call__(self):
        pass

class Layer:
    def __init__(self,*args):
        pass
        
    def __call__(self,input_nodes):
        self.input_nodes=input_nodes
        self.consumers=[]
        for input_node in input_nodes:
            input_node.consumers.append(self)
        
        _default_graph.layer.append(self)
    
