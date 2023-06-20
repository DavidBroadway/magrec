
import torch
from abc import ABC, abstractmethod


class Condition(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
    
    @abstractmethod
    def sample(self):
        pass
    
    @abstractmethod
    def loss(self):
        pass

    
    
class PDE(Condition):
    def __init__(self, pde, name) -> None:
        super().__init__()
        self.pde = pde
        self.name = name
    
    def sample(self):
        pass
    
    def loss(self):
        pass
    
    def in_region(self, x) -> bool:
        pass
    