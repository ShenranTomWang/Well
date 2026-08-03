from .template import TemplateWithKey, FewShotExampleTemplate
from typing import List, Dict, Type
from response import FinalAnswerResponse, Response
from constant.constant import MODEL_FINAL_ANSWER_KEY

class FinalAnswerFewShotExample(FewShotExampleTemplate):
    question: str
    answer: str
    user_role: str
    model_role: str
    
    def __init__(
        self,
        question: str,
        is_normal: bool,
        answer: str,
        presuppositions: List[str] = None,
        user_role: str = "user",
        model_role: str = "assistant",
        **kwargs
    ):
        self.question = question
        self.presuppositions = presuppositions
        self.is_normal = is_normal
        self.answer = answer
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        if self.is_normal:
            feedback = "The question is valid and does not contain false presuppositions."
            action = "Answer the question directly based on the presuppositions."
        else:
            feedback = f"The question contains false presuppositions that {'; '.join(self.presuppositions)}."
            action = f"Correct the false assumptions that {'; '.join(self.presuppositions)} and respond based on the corrected assumption."
        return [
            f"{self.user_role}: Question: {self.question}\nFeedback: {feedback}\nAction: {action}\n{self.model_role}: {self.answer}"
        ]
        

class FPInterpretationFewShotExample(FewShotExampleTemplate):
    question: str
    answer: str
    user_role: str
    model_role: str
    
    def __init__(
        self,
        question: str,
        is_normal: bool,
        answer: str,
        user_role: str = "user",
        model_role: str = "assistant",
        **kwargs
    ):
        self.question = question
        self.is_normal = is_normal
        self.answer = answer
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        return [
            f"{self.user_role}: Question: {self.question}\n{self.model_role}: {self.answer}"
        ] if not self.is_normal else []


class FinalAnswerTemplate(TemplateWithKey):
    ResponseClass: Type[Response] = FinalAnswerResponse
    answer_key: str = MODEL_FINAL_ANSWER_KEY
    question: str
    model_feedback_action: str
    few_shot_data: List[FinalAnswerFewShotExample]
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(
        self,
        question: str,
        model_feedback_action: str,
        few_shot_data: List[Dict],
        system_role: str = "system",
        user_role: str = "user",
        model_role: str = "assistant",
        **kwargs
    ):
        self.question = question
        self.model_feedback_action = model_feedback_action
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FinalAnswerFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()
        
    def generate(self, **kwargs) -> List[Dict]:
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides a response to a question based on the feedback and action guideline.
                    You will be given a question and feedback and action guideline on how to answer the question.
                    Your task is to provide a final answer to the question based on the feedback and action guideline.
                    {"Below are some examples to help you understand the task." if len(self.few_shot_data) > 0 else ""}
                    {"\n\n".join(self.few_shot_data)}
                """},
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\n{self.model_feedback_action}\n"
            }
        ]
        return messages
    
class FactCheckFinalAnswerTemplate(TemplateWithKey):
    ResponseClass: Type[Response] = FinalAnswerResponse
    answer_key: str = MODEL_FINAL_ANSWER_KEY
    question: str
    factcheck_results: List[int]
    model_detected_presuppositions: List[str]
    few_shot_data: List[FinalAnswerFewShotExample]
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(
        self,
        question: str,
        factcheck_results: List[int],
        few_shot_data: List[Dict],
        model_detected_presuppositions: List[str] = None,
        system_role: str = "system",
        user_role: str = "user",
        model_role: str = "assistant",
        **kwargs
    ):
        self.question = question
        self.factcheck_results = factcheck_results
        if model_detected_presuppositions:
            self.model_detected_presuppositions = model_detected_presuppositions
        else:
            raise ValueError("model_detected_presuppositions must be provided")
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FinalAnswerFewShotExample(
                **dp,
                user_role=self.user_role,
                model_role=self.model_role
            ).generate()
        
    def generate(self, **kwargs) -> List[Dict]:
        false_presuppositions = [self.model_detected_presuppositions[i] for i, res in enumerate(self.factcheck_results) if res == 0]
        if len(false_presuppositions) == 0:
            feedback_action = "Feedback: The question is valid and does not contain false presuppositions.\nAction: Answer the question directly based on the presuppositions."
        else:
            feedback_action = f"Feedback: The question contains false presuppositions that {'; '.join(false_presuppositions)}.\nAction: Correct the false assumptions that {'; '.join(false_presuppositions)} and respond based on the corrected assumption."
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides a response to a question based on the feedback and action guideline.
                    You will be given a question and feedback and action guideline on how to answer the question.
                    Your task is to provide a final answer to the question based on the feedback and action guideline.
                    {"Below are some examples to help you understand the task." if len(self.few_shot_data) > 0 else ""}
                    {"\n\n".join(self.few_shot_data)}
                """},
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\n{feedback_action}\n"
            }
        ]
        return messages


class FactCheckFPInterpretationTemplate(TemplateWithKey):
    ResponseClass: Type[Response] = FinalAnswerResponse
    answer_key: str = MODEL_FINAL_ANSWER_KEY
    question: str
    model_FP_identification: int
    few_shot_data: List[FinalAnswerFewShotExample]
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(
        self,
        question: str,
        model_FP_identification: int,
        few_shot_data: List[Dict],
        system_role: str = "system",
        user_role: str = "user",
        model_role: str = "assistant",
        **kwargs
    ):
        self.question = question
        self.model_FP_identification = model_FP_identification
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FPInterpretationFewShotExample(
                **dp,
                user_role=self.user_role,
                model_role=self.model_role
            ).generate()
        
    def generate(self, **kwargs) -> List[Dict]:
        has_FP = self.model_FP_identification == 0
        hint = """
            You will be provided with a question that contains at least 1 false assumption.
            Your task is to help me understand what are the false assumptions.
            Write an explanation to pinpoint the false assumptions.
        """ if has_FP else ""
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that answers questions. \
                    The user will ask a question, and you need to provide the answer to that question.
                    {hint}
                    {"Below are some examples to help you understand the task." if len(self.few_shot_data) > 0 and has_FP else ""}
                    {"\n\n".join(self.few_shot_data) if len(self.few_shot_data) > 0 and has_FP else ""}
                """},
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\n"
            }
        ]
        return messages
