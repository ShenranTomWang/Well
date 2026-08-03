from pydantic import BaseModel
from abc import abstractmethod
from typing import Dict, Callable, Any, List
import re

_RESPONSE_REGISTRY: Dict[str, "Response"] = {}

def register(name: str | None = None) -> Callable:
    def decorator(cls: "Response") -> "Response":
        op_name = name if name is not None else cls.__name__
        _RESPONSE_REGISTRY[op_name] = cls
        return cls
    return decorator

def get_response(name: str, **kwargs) -> "Response":
    """
    Get a response by name.
    Args:
        name (str): The name of the response class.
        **kwargs: Additional arguments to pass to the response constructor.
    Returns:
        Response: The response instance.
    """
    return _RESPONSE_REGISTRY[name](**kwargs)

def get_response_cls(name: str, **kwargs) -> "Response":
    """
    Get a response class by name.
    Args:
        name (str): The name of the response class.
        **kwargs: Additional arguments to pass to the response constructor.
    Returns:
        Response: The response instance.
    """
    return _RESPONSE_REGISTRY[name]

class Response(BaseModel):
    @classmethod
    @abstractmethod
    def model_validate_plain_text(cls, text: str, **kwargs) -> "Response":
        pass
    
    @abstractmethod
    def get(self) -> Any:
        pass
    
@register()
class PresuppositionExtractionResponse(Response):
    presuppositions: List[str]
    
    @classmethod
    def model_validate_plain_text(
        cls,
        text: str | None,
        model_role: str = "assistant",
        filter_presuppositions: bool = False,
        **kwargs
    ) -> "PresuppositionExtractionResponse":
        if text is None:
            presuppositions = []
        else:
            text = text.lower().replace(f"{model_role}\n", "")
            text = text.lower().replace(f"{model_role}: ", "")
            if filter_presuppositions:
                presuppositions = [line.strip() for line in text.split("\n") if len(line.strip()) > 0 and "presupposition" not in line.lower()]
            else:
                presuppositions = [line.strip() for line in text.split("\n")]
            presuppositions = [line for line in presuppositions if "assistant" not in line.lower()]
        return cls(presuppositions=presuppositions)
    
    def get(self) -> List[str]:
        return self.presuppositions

@register()
class QuestionToStatementResponse(Response):
    statement: str

    @classmethod
    def model_validate_plain_text(
        cls,
        text: str | None,
        model_role: str = "assistant",
        **kwargs
    ) -> "QuestionToStatementResponse":
        if text is None:
            statement = ""
        else:
            statement = text.strip()
            statement = statement.replace(f"{model_role}\n", "")
            statement = statement.replace(f"{model_role}: ", "")
            if statement.lower().startswith("statement:"):
                statement = statement.split(":", 1)[1].strip()
            statement = statement.split("\n")[0].strip()
        return cls(statement=statement)

    def get(self) -> str:
        return self.statement
    

@register()
class FPIdentificationResponse(Response):
    has_false_assumption: int

    @classmethod
    def model_validate_plain_text(
        cls,
        text: str | None,
        model_role: str = "assistant",
        **kwargs
    ) -> "FPIdentificationResponse":
        if text is None:
            return cls(has_false_assumption=0)
        text = text.strip().lower()
        text = text.replace(f"{model_role}\n", "")
        text = text.replace(f"{model_role}: ", "")
        match = re.search(r'\b(yes|true|no|false)\b', text)
        result = int(match.group(1) in ['yes', 'true']) if match else 0
        return cls(has_false_assumption=result)

    def get(self) -> int:
        return self.has_false_assumption
    
@register()
class LogicalFormExtractionResponse(Response):
    logical_form: List[str]
    
    @classmethod
    def model_validate_plain_text(
        cls,
        text: str | None,
        model_role: str = "assistant",
        **kwargs
    ) -> "LogicalFormExtractionResponse":
        if text is None:
            logical_form = []
        else:
            text = text.lower().replace(f"{model_role}\n", "")
            text = text.lower().replace(f"{model_role}: ", "")
            logical_form = text.split('\n')
        return cls(logical_form=logical_form)
    
    def get(self) -> List[str]:
        return self.logical_form
    
@register()
class FeedbackActionResponse(Response):
    feedback_action: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str, **kwargs) -> "FeedbackActionResponse":
        if not text:
            text = ""
        return cls(feedback_action=text)
    
    def get(self) -> str:
        return self.feedback_action

@register()
class FinalAnswerResponse(Response):
    answer: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str, model_role: str = "assistant", **kwargs) -> "FinalAnswerResponse":
        text = text.lower().replace(f"{model_role}\n", "") if text else ""
        return cls(answer=text.strip())
    
    def get(self) -> str:
        return self.answer

@register()
class ClaimCoverageResponse(Response):
    coverage: int | str
    
    @classmethod
    def model_validate_plain_text(cls, text: str, **kwargs) -> "ClaimCoverageResponse":
        try:
            if text.strip().lower() in ['true', 'yes']:
                coverage = 1
            elif text.strip().lower() in ['false', 'no']:
                coverage = 0
            else:
                coverage = int(text.strip())
        except ValueError:
            coverage = 0
        return cls(coverage=coverage)
    
    def get(self) -> int | str:
        return self.coverage

@register()
class LLMCheckResponse(Response):
    LLM_check_results: int
    
    @classmethod
    def model_validate_plain_text(cls, text: str, **kwargs) -> "LLMCheckResponse":
        if not text:
            return cls(LLM_check_results=0)
        text = text.strip().lower()
        match = re.search(r'\b(yes|true|no|false)\b', text)
        result = int(match.group(1) in ['yes', 'true']) if match else 0
        return cls(LLM_check_results=result)
    
    def get(self) -> int:
        return self.LLM_check_results
    
@register()
class ResponseLevelScoreResponse(Response):
    score: float
    explanation: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str, **kwargs) -> "ResponseLevelScoreResponse":
        try:
            score = re.search(r'\b(?:Score|Rating):\s*([+-]?\d+)\b', text.replace("*", "")).group(1)
            score = float(score.strip())
            explanation = text
        except Exception:
            explanation = text if text is not None else ""
            score = 0
        return cls(score=score, explanation=explanation.strip())
    
    def get(self) -> Dict[str, Any]:
        return {"score": self.score, "explanation": self.explanation}
    
    def get_normalized_score(self) -> float:
            return self.score / 6