from typing import Dict, Type
from job_cache import BatchJobCache
from .direct_qa_operator import DirectQAOperator, BatchDirectQAOperator, get_operator_cls

class GeminiRunDirectQABatchJobCache(BatchJobCache):
    save_to: str
    batch_job_name: str
    
    def __init__(
        self,
        save_to: str,
        batch_job_name: str,
        OperatorClass: str | Type[DirectQAOperator] | Type[BatchDirectQAOperator]
    ):
        self.save_to = save_to
        self.batch_job_name = batch_job_name
        if isinstance(OperatorClass, str):
            self.OperatorClass = get_operator_cls(OperatorClass)
            self.OperatorClass_name = OperatorClass
        else:
            self.OperatorClass = OperatorClass
            self.OperatorClass_name = OperatorClass.__name__

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "GeminiRunDirectQABatchJobCache":
        return cls(
            save_to=d['save_to'],
            batch_job_name=d['batch_job_name'],
            OperatorClass=d['OperatorClass']
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'save_to': self.save_to,
            'batch_job_name': self.batch_job_name,
            'OperatorClass': self.OperatorClass_name
        }