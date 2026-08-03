from typing import Dict, Type
from response import get_response_cls, Response
from .batch_job_cache import BatchJobCache

class GeminiBatchJobCache(BatchJobCache):
    def __init__(self, save_to: str, batch_job_name: str, ResponseClass: str | Type[Response]):
        self.save_to = save_to
        self.batch_job_name = batch_job_name
        if isinstance(ResponseClass, str):
            self.ResponseClass = get_response_cls(ResponseClass)
            self.ResponseClass_name = ResponseClass
        else:
            self.ResponseClass = ResponseClass
            self.ResponseClass_name = ResponseClass.__name__
        
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "GeminiBatchJobCache":
        return cls(
            save_to=d['save_to'],
            batch_job_name=d['batch_job_name'],
            ResponseClass=d['ResponseClass']
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'save_to': self.save_to,
            'batch_job_name': self.batch_job_name,
            'ResponseClass': self.ResponseClass_name
        }
