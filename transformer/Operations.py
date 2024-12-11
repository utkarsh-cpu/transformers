from lib2to3.pgen2.token import OP
from comp_graph import Operation
import torch as th

class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def __call__(self, x_value, y_value):
        return x_value + y_value

class matmul(Operation):
    def __init__(self,x,y):
        super().__init__([x, y])

    def __call__(self, x_value, y_value):
        return th.matmul(x_value,y_value)

class mul(Operation):
    def __init__(self,x,y):
        super().__init__([x, y])

    def __call__(self, x_value, y_value):
        return x_value*y_value

class pow(Operation):
    def __init__(self,x,y):
        super().__init__([x, y])

    def __call__(self, x_value, y_value):
        return x_value**y_value