from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Union, Type
from .batch_job_cache import GeminiRunFactCheckBatchJobCache

_OPERATOR_REGISTRY: Dict[str, Union["CheckOperator", "BatchCheckOperator"]] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: Union["CheckOperator", "BatchCheckOperator"]) -> Union["CheckOperator", "BatchCheckOperator"]:
        op_name = name if name is not None else cls.__name__
        _OPERATOR_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_operator(name: str, **kwargs) -> Union["CheckOperator", "BatchCheckOperator"]:
    """
    Get a data operator by name.
    Args:
        name (str): The name of the data operator.
        **kwargs: Additional arguments to pass to the operator constructor.
    Returns:
        CheckOperator | BatchCheckOperator: The operator instance.
    """
    return _OPERATOR_REGISTRY[name](**kwargs)

def get_operator_cls(name: str) -> Union[Type["CheckOperator"], Type["BatchCheckOperator"]]:
    """
    Get a data operator class by name.
    Args:
        name (str): The name of the data operator.
    Returns:
        Type[CheckOperator] | Type[BatchCheckOperator]: The operator class.
    """
    return _OPERATOR_REGISTRY[name]

class CheckOperator(ABC):
    @abstractmethod
    def check(self, dp: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass
    
class BatchCheckOperator(ABC):
    @abstractmethod
    def submit_job(self, dps: Dict[str, Any], **kwargs) -> GeminiRunFactCheckBatchJobCache:
        pass
    
    @abstractmethod
    def checkback(self, job: GeminiRunFactCheckBatchJobCache, **kwargs) -> Dict[str, Any]:
        pass