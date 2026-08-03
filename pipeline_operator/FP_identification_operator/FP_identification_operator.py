from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type, Union
from job_cache import BatchJobCache

_OPERATOR_REGISTRY: Dict[str, Union["FPIdentificationOperator", "BatchFPIdentificationOperator"]] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: Union["FPIdentificationOperator", "BatchFPIdentificationOperator"]) -> Union["FPIdentificationOperator", "BatchFPIdentificationOperator"]:
        op_name = name if name is not None else cls.__name__
        _OPERATOR_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_operator(name: str, **kwargs) -> Union["FPIdentificationOperator", "BatchFPIdentificationOperator"]:
    return _OPERATOR_REGISTRY[name](**kwargs)

def get_operator_cls(name: str) -> Union[Type["FPIdentificationOperator"], Type["BatchFPIdentificationOperator"]]:
    return _OPERATOR_REGISTRY[name]

class FPIdentificationOperator(ABC):
    @abstractmethod
    def extract(self, dp: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass

class BatchFPIdentificationOperator(ABC):
    @abstractmethod
    def submit_job(self, dps: Dict[str, Any], **kwargs) -> BatchJobCache:
        pass

    @abstractmethod
    def checkback(self, job: BatchJobCache, **kwargs) -> Dict[str, Any]:
        pass
