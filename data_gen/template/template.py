from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Type
from response import Response

_TEMPLATE_REGISTRY: Dict[str, "Template"] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: "Template") -> "Template":
        op_name = name if name is not None else cls.__name__
        _TEMPLATE_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_template(name: str, **kwargs) -> "Template":
    """
    Get a template by name.
    Args:
        name (str): The name of the template.
        **kwargs: Additional arguments to pass to the template constructor.
    Returns:
        Template: The template instance.
    """
    return _TEMPLATE_REGISTRY[name](**kwargs)

def get_template_cls(name: str, **kwargs) -> Type["Template"]:
    """
    Get a template class by name.
    Args:
        name (str): The name of the template.
        **kwargs: Additional arguments to pass to the template constructor.
    Returns:
        Type[Template]: The template class.
    """
    return _TEMPLATE_REGISTRY[name]

class FewShotExampleTemplate(ABC):
    @abstractmethod
    def generate(self, **kwargs) -> List[Dict]:
        pass
    
class Template(ABC):
    ResponseClass: Type[Response]

    @abstractmethod
    def generate(self, **kwargs) -> str | List[Dict]:
        pass

    def parse(self, response: str, **kwargs) -> Any:
        """
        Parse the response from the model and return the result in the desired format.
        This calls .get() already
        """
        return self.ResponseClass.model_validate_plain_text(response, **kwargs).get()

class TemplateWithKey(Template):
    answer_key: str