from .template import Template, FewShotExampleTemplate
from typing import List, Dict, Type
from response import LLMCheckResponse, Response

class LLMCheckFewShotExample(FewShotExampleTemplate):
    presupposition: str
    user_role: str
    model_role: str
    
    def __init__(self, presuppositions: List[str], is_normal: bool, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.presupposition = presuppositions[0]
        self.is_normal = is_normal
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        if self.is_normal:
            feedback = "true"
        else:
            feedback = "false"
        user_content = f"Presupposition: {self.presupposition}"
        return [
            f"{self.user_role}: {user_content}\n{self.model_role}: {feedback}"
        ]

class LLMCheckTemplate(Template):
    ResponseClass: Type[Response] = LLMCheckResponse
    passages: List[str]
    model_detected_presupposition: str
    few_shot_data: List[LLMCheckFewShotExample]
    system_role: str
    user_role: str
    model_role: str
    
    def __init__(
            self,
            passages: List[str],
            model_detected_presupposition: str,
            few_shot_data: List[Dict],
            system_role: str = "system",
            user_role: str = "user",
            model_role: str = "assistant",
            **kwargs
        ):
        self.model_detected_presupposition = model_detected_presupposition
        self.passages = passages
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += LLMCheckFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()

    def generate(self, **kwargs) -> List[Dict]:
        user_content = f"Presupposition: {self.model_detected_presupposition}"
        if len(self.passages) > 0:
            RAG_content = f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}"
        else:
            RAG_content = None
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that fact-checks a presupposition.
                    You will be given a presupposition.
                    Your task is to determine whether it is true or false.
                    You should just return one word "true" or "false" as your answer, without any additional explanation.
                    {"Below are some examples to help you understand the task." if len(self.few_shot_data) > 0 else ""}
                    {'\n\n'.join(self.few_shot_data)}
                    {RAG_content if RAG_content is not None else ""}
                """
            },
            {
                "role": self.user_role,
                "content": user_content
            }
        ]
        return messages