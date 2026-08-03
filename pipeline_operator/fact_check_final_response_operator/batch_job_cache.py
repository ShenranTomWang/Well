from typing import Dict, Type
from job_cache import BatchJobCache
from .fact_check_final_response_operator import FactCheckFinalResponseOperator, BatchFactCheckFinalResponseOperator, get_operator_cls

class GeminiRunFactCheckFinalResponseBatchJobCache(BatchJobCache):
    save_to: str
    batch_job_name: str
    
    def __init__(
        self,
        save_to: str,
        batch_job_name: str,
        OperatorClass: str | Type[FactCheckFinalResponseOperator] | Type[BatchFactCheckFinalResponseOperator],
        dataset_name: str,
    ):
        self.save_to = save_to
        self.batch_job_name = batch_job_name
        self.dataset_name = dataset_name
        if isinstance(OperatorClass, str):
            self.OperatorClass = get_operator_cls(OperatorClass)
            self.OperatorClass_name = OperatorClass
        else:
            self.OperatorClass = OperatorClass
            self.OperatorClass_name = OperatorClass.__name__

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "GeminiRunFactCheckFinalResponseBatchJobCache":
        return cls(
            save_to=d['save_to'],
            batch_job_name=d['batch_job_name'],
            OperatorClass=d['OperatorClass'],
            dataset_name=d['dataset_name']
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'save_to': self.save_to,
            'batch_job_name': self.batch_job_name,
            'OperatorClass': self.OperatorClass_name,
            'dataset_name': self.dataset_name
        }