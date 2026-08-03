from job_cache import GeminiBatchJobCache
from .registry import get_evaluator_cls
from response import Response
from typing import Type, Dict, Any

class GeminiScoreBatchJobCache(GeminiBatchJobCache):
    def __init__(
        self,
        save_to: str,
        batch_job_name: str,
        ResponseClass: str | Type[Response],
        task: str,
        EvaluatorClass: str | Type[Any]
    ):
        super().__init__(
            save_to=save_to,
            batch_job_name=batch_job_name,
            ResponseClass=ResponseClass
        )
        self.task = task
        if isinstance(EvaluatorClass, str):
            self.EvaluatorClass = get_evaluator_cls(EvaluatorClass)
            self.EvaluatorClass_name = EvaluatorClass
        else:
            self.EvaluatorClass = EvaluatorClass
            self.EvaluatorClass_name = EvaluatorClass.__name__
        
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "GeminiScoreBatchJobCache":
        return cls(
            save_to=d['save_to'],
            batch_job_name=d['batch_job_name'],
            ResponseClass=d['ResponseClass'],
            task=d['task'],
            EvaluatorClass=d['EvaluatorClass']
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'save_to': self.save_to,
            'batch_job_name': self.batch_job_name,
            'ResponseClass': self.ResponseClass_name,
            'task': self.task,
            'EvaluatorClass': self.EvaluatorClass_name
        }
        
class ResponseLevelScoreBatchJobCache(GeminiBatchJobCache):
    def __init__(
        self,
        save_to: str,
        batch_job_name: str,
        ResponseClass: str | Type[Response],
        EvaluatorClass: str | Type[Any]
    ):
        super().__init__(
            save_to=save_to,
            batch_job_name=batch_job_name,
            ResponseClass=ResponseClass
        )
        if isinstance(EvaluatorClass, str):
            self.EvaluatorClass = get_evaluator_cls(EvaluatorClass)
            self.EvaluatorClass_name = EvaluatorClass
        else:
            self.EvaluatorClass = EvaluatorClass
            self.EvaluatorClass_name = EvaluatorClass.__name__
        
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "ResponseLevelScoreBatchJobCache":
        return cls(
            save_to=d['save_to'],
            batch_job_name=d['batch_job_name'],
            ResponseClass=d['ResponseClass'],
            EvaluatorClass=d['EvaluatorClass']
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'save_to': self.save_to,
            'batch_job_name': self.batch_job_name,
            'ResponseClass': self.ResponseClass_name,
            'EvaluatorClass': self.EvaluatorClass_name
        }