from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Union, Type
from job_cache import BatchJobCache

_OPERATOR_REGISTRY: Dict[str, Union["DirectQAOperator", "BatchDirectQAOperator"]] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: Union["DirectQAOperator", "BatchDirectQAOperator"]) -> Union["DirectQAOperator", "BatchDirectQAOperator"]:
        op_name = name if name is not None else cls.__name__
        _OPERATOR_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_operator(name: str, **kwargs) -> Union["DirectQAOperator", "BatchDirectQAOperator"]:
    """
    Get a data operator by name.
    Args:
        name (str): The name of the data operator.
        **kwargs: Additional arguments to pass to the operator constructor.
    Returns:
        DirectQAOperator | BatchDirectQAOperator: The operator instance.
    """
    return _OPERATOR_REGISTRY[name](**kwargs)

def get_operator_cls(name: str) -> Union[Type["DirectQAOperator"], Type["BatchDirectQAOperator"]]:
    """
    Get a data operator class by name.
    Args:
        name (str): The name of the data operator.
    Returns:
        Type[DirectQAOperator] | Type[BatchDirectQAOperator]: The operator class.
    """
    return _OPERATOR_REGISTRY[name]

class DirectQAOperator(ABC):
    @abstractmethod
    def qa(self, dp: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass
    
class BatchDirectQAOperator(ABC):
    @abstractmethod
    def submit_job(self, dps: Dict[str, Any], **kwargs) -> BatchJobCache:
        pass
    
    @abstractmethod
    def checkback(self, job: BatchJobCache, **kwargs) -> Dict[str, Any]:
        pass