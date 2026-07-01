import numpy as np
import torch as tc
import math as mt

#get all oprators

class Operators:
     
     def __init__(self, name, device, tensor):
          super().__init__()
          self.name = name
          self.device = device
          self.tensor = tensor

     def get_single_ope(self):
        
         if self.name == 'Sz':
             return tc.tensor([[1, 0], [0, -1]], device=self.device, dtype=self.tensor.dtype) / 2