from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Union, Type
from job_cache import BatchJobCache

_OPERATOR_REGISTRY: Dict[str, Union["QuestionToStatementOperator", "BatchQuestionToStatementOperator"]] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: Union["QuestionToStatementOperator", "BatchQuestionToStatementOperator"]) -> Union["QuestionToStatementOperator", "BatchQuestionToStatementOperator"]:
        op_name = name if name is not None else cls.__name__
        _OPERATOR_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_operator(name: str, **kwargs) -> Union["QuestionToStatementOperator", "BatchQuestionToStatementOperator"]:
    return _OPERATOR_REGISTRY[name](**kwargs)

def get_operator_cls(name: str) -> Union[Type["QuestionToStatementOperator"], Type["BatchQuestionToStatementOperator"]]:
    return _OPERATOR_REGISTRY[name]

class QuestionToStatementOperator(ABC):
    @abstractmethod
    def extract(self, dp: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass

class BatchQuestionToStatementOperator(ABC):
    @abstractmethod
    def submit_job(self, dps: Dict[str, Any], **kwargs) -> BatchJobCache:
        pass

    @abstractmethod
    def checkback(self, job: BatchJobCache, **kwargs) -> Dict[str, Any]:
        pass
