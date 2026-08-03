from .template import TemplateWithKey, FewShotExampleTemplate
from typing import List, Dict, Type
from response import PresuppositionExtractionResponse, Response
from constant.constant import MODEL_DETECTED_PRESUPPOSITIONS_KEY

class PresuppositionExtractionFewShotExample(FewShotExampleTemplate):
    question: str
    presuppositions: List[str]
    user_role: str
    model_role: str
    
    def __init__(self, question: str, presuppositions: List[str], user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        content = "\n".join(self.presuppositions) + "\n"
        return [
            f"{self.user_role}: {self.question}\n{self.model_role}: {content}"
        ]

class PresuppositionExtractionTemplate(TemplateWithKey):
    ResponseClass: Type[Response] = PresuppositionExtractionResponse
    answer_key: str = MODEL_DETECTED_PRESUPPOSITIONS_KEY
    question: str
    few_shot_data: List[PresuppositionExtractionFewShotExample]
    passages: int
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(self, question: str, few_shot_data: List[Dict], passages: int = 0, system_role: str = "system", model_role: str = "assistant", user_role: str = "user", **kwargs):
        self.question = question
        self.passages = passages
        self.system_role = system_role
        self.model_role = model_role
        self.user_role = user_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += PresuppositionExtractionFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()
    
    def generate(self, **kwargs) -> List[Dict]:
        if len(self.passages) > 0:
            RAG_content = f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}"
        else:
            RAG_content = None
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that analyzes the given question.
                    Your task is to extract presuppositions in the given question.
                    Notice that the presuppositions in a question could be true or false, and may be explicit or implicit.
                    There could be multiple presuppositions in a question, but there will always be at least one presupposition in the question.
                    Format your response as a list of presuppositions, separated by newlines.
                    {"Below are some examples to help you understand the task." if len(self.few_shot_data) > 0 else ""}
                    {'\n\n'.join(self.few_shot_data)}
                    {RAG_content if RAG_content is not None else ""}
                """
            },
            {
                "role": self.user_role,
                "content": self.question
            }
        ]
        return messages
    
    def parse(self, response: str, filter_presuppositions: bool = True, **kwargs):
        return super().parse(response, filter_presuppositions=filter_presuppositions, **kwargs)