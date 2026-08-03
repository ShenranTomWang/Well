from collections.abc import Callable
from typing import Dict, Any, Type

_EVALUATOR_REGISTRY: Dict[str, Any] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: Any) -> Any:
        op_name = name if name is not None else cls.__name__
        _EVALUATOR_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_evaluator(name: str, **kwargs) -> Any:
    """
    Get a evaluator by name.
    Args:
        name (str): The name of the data evaluator.
        **kwargs: Additional arguments to pass to the evaluator constructor.
    Returns:
        GeminiScoreBatchedEvaluator | ResponseLevelScoreBatchedEvaluator: The evaluator instance.
    """
    return _EVALUATOR_REGISTRY[name](**kwargs)

def get_evaluator_cls(name: str) -> Type[Any]:
    """
    Get the evaluator class by name.
    Args:
        name (str): The name of the data evaluator.
    Returns:
        Type[GeminiScoreBatchedEvaluator | ResponseLevelScoreBatchedEvaluator]: The evaluator class.
    """
    return _EVALUATOR_REGISTRY[name]