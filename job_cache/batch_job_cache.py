from abc import ABC, abstractmethod
from typing import Dict, Any

class BatchJobCache(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJobCache':
        pass