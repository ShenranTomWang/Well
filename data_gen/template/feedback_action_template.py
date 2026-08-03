from .template import TemplateWithKey, FewShotExampleTemplate
from typing import List, Dict, Type
from response import FeedbackActionResponse, Response
from constant.constant import MODEL_FEEDBACK_ACTION_KEY

class FeedbackActionFewShotExample(FewShotExampleTemplate):
    question: str
    presuppositions: List[str]
    user_role: str
    model_role: str
    
    def __init__(self, question: str, presuppositions: List[str], is_normal: bool, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.is_normal = is_normal
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        presuppositions = self.presuppositions
        presuppositions.append("There is a clear and single answer to the question.")
        if self.is_normal:
            feedback = "The question is valid and does not contain false presuppositions."
            action = "Answer the question directly based on the presuppositions."
        else:
            feedback = f"The question contains false presuppositions that {'; '.join(self.presuppositions)}."
            action = f"Correct the false assumptions that {'; '.join(self.presuppositions)} and respond based on the corrected assumption."
        model_content = f"Feedback: {feedback}\nAction: {action}"
        user_content = f"Question: {self.question}\nPresuppositions: {'; '.join(self.presuppositions)}"
        return [
            f"{self.user_role}: {user_content}\n{self.model_role}: {model_content}"
        ]

class FeedbackActionTemplate(TemplateWithKey):
    ResponseClass: Type[Response] = FeedbackActionResponse
    answer_key: str = MODEL_FEEDBACK_ACTION_KEY
    question: str
    passages: List[str]
    model_detected_presuppositions: List[str]
    few_shot_data: List[FeedbackActionFewShotExample]
    system_role: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, passages: List[str], model_detected_presuppositions: List[str], few_shot_data: List[Dict], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.passages = passages
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.model_detected_presuppositions = model_detected_presuppositions
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FeedbackActionFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()

    def generate(self, **kwargs) -> List[Dict]:
        user_content = f"Question: {self.question}\nPresuppositions: {'; '.join(self.model_detected_presuppositions)}; There is a clear and single answer to the question"
        if len(self.passages) > 0:
            RAG_content = f"For the following question, you are given these additional information: \nAdditional Information: {' ||| '.join(self.passages)}"
        else:
            RAG_content = None
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides feedback on the question and a guideline for answering the question.
                    You will be given a question and the assumptions that are implicit in the question.
                    Your task is to first, provide feedback on the question based on whether it contains any false assumptions and then provide a guideline for answering the question.
                    Separate your feedback and action with a newline, and format your response as:
                    Feedback: <your feedback>\nAction: <your action>.
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