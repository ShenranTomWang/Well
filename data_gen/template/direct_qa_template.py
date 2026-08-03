from .template import TemplateWithKey, FewShotExampleTemplate
from typing import List, Dict, Type
from response import FinalAnswerResponse, Response
from constant.constant import MODEL_FINAL_ANSWER_KEY

class DirectQAFewShotExample(FewShotExampleTemplate):
    question: str
    answer: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, answer: str, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.answer = answer
        self.user_role = user_role
        self.model_role = model_role
        
    def generate(self, **kwargs) -> List[Dict]:
        user_content = self.question
        return [
            f"{self.user_role}: {user_content}\n{self.model_role}: {self.answer}"
        ]

class DirectQATemplate(TemplateWithKey):
    ResponseClass: Type[Response] = FinalAnswerResponse
    answer_key: str = MODEL_FINAL_ANSWER_KEY
    
    def __init__(self, question: str, passages: List[str], few_shot_data: List[Dict], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", system_prompt: str | None = None, **kwargs):
        self.question = question
        self.passages = passages
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.system_prompt = system_prompt
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += DirectQAFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()
        
    def generate(self, **kwargs):
        user_content = self.question
        if len(self.passages) > 0:
            RAG_content = f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}"
        else:
            RAG_content = None
        default_system_prompt = (
            "You are a helpful assistant that answer questions based on your knowledge.\n"
            "The user will ask a question, and you need to provide the answer to that question."
        )
        system_prompt = self.system_prompt if self.system_prompt is not None else default_system_prompt
        content_parts = [system_prompt]
        if len(self.few_shot_data) > 0:
            content_parts.append("Below are some examples to help you understand the task.")
            content_parts.append("\n\n".join(self.few_shot_data))
        if RAG_content is not None:
            content_parts.append(RAG_content)
        messages = [
            {"role": self.system_role, "content": "\n\n".join(content_parts)},
            {"role": self.user_role, "content": user_content}
        ]
        return messages
