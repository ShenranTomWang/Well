from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Union, Type
from job_cache import BatchJobCache

_OPERATOR_REGISTRY: Dict[str, Union["FactCheckFinalResponseOperator", "BatchFactCheckFinalResponseOperator"]] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: Union["FactCheckFinalResponseOperator", "BatchFactCheckFinalResponseOperator"]) -> Union["FactCheckFinalResponseOperator", "BatchFactCheckFinalResponseOperator"]:
        op_name = name if name is not None else cls.__name__
        _OPERATOR_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_operator(name: str, **kwargs) -> Union["FactCheckFinalResponseOperator", "BatchFactCheckFinalResponseOperator"]:
    """
    Get a data operator by name.
    Args:
        name (str): The name of the data operator.
        **kwargs: Additional arguments to pass to the operator constructor.
    Returns:
        FactCheckFinalResponseOperator | BatchFactCheckFinalResponseOperator: The operator instance.
    """
    return _OPERATOR_REGISTRY[name](**kwargs)

def get_operator_cls(name: str) -> Union[Type["FactCheckFinalResponseOperator"], Type["BatchFactCheckFinalResponseOperator"]]:
    """
    Get a data operator class by name.
    Args:
        name (str): The name of the data operator.
    Returns:
        Type[FactCheckFinalResponseOperator] | Type[BatchFactCheckFinalResponseOperator]: The operator class.
    """
    return _OPERATOR_REGISTRY[name]

class FactCheckFinalResponseOperator(ABC):
    @abstractmethod
    def respond(self, dp: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass
    
class BatchFactCheckFinalResponseOperator(ABC):
    @abstractmethod
    def submit_job(self, dps: Dict[str, Any], **kwargs) -> BatchJobCache:
        pass
    
    @abstractmethod
    def checkback(self, job: BatchJobCache, **kwargs) -> Dict[str, Any]:
        pass