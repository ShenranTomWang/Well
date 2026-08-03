from .template import TemplateWithKey, FewShotExampleTemplate
from typing import List, Dict, Type
from response import QuestionToStatementResponse, Response
from constant.constant import MODEL_CONVERTED_STATEMENT_KEY


class QuestionToStatementFewShotExample(FewShotExampleTemplate):
    question: str
    statement: str
    user_role: str
    model_role: str

    def __init__(self, question: str, statement: str, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.statement = statement
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[str]:
        return [
            f"{self.user_role}: Question: {self.question}\n{self.model_role}: Statement: {self.statement}"
        ]


class QuestionToStatementTemplate(TemplateWithKey):
    ResponseClass: Type[Response] = QuestionToStatementResponse
    answer_key: str = MODEL_CONVERTED_STATEMENT_KEY
    question: str
    few_shot_data: List[QuestionToStatementFewShotExample]
    passages: List[str]
    system_role: str
    model_role: str
    user_role: str

    def __init__(
        self,
        question: str,
        few_shot_data: List[Dict],
        passages: List[str] = None,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        self.question = question
        self.passages = passages if passages is not None else []
        self.system_role = system_role
        self.model_role = model_role
        self.user_role = user_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += QuestionToStatementFewShotExample(
                **dp,
                user_role=self.user_role,
                model_role=self.model_role
            ).generate()

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
                    You will be provided with a question. Your task is to transform the question into a statement and keep its original meaning.
                    Return exactly one statement. Do not answer the question or add explanations.
                    {"Below are some examples to help you understand the task." if len(self.few_shot_data) > 0 else ""}
                    {'\n\n'.join(self.few_shot_data)}
                    {RAG_content if RAG_content is not None else ""}
                """
            },
            {
                "role": self.user_role,
                "content": f"Question: {self.question}"
            }
        ]
        return messages
