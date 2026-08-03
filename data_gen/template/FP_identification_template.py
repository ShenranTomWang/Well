from .template import TemplateWithKey, FewShotExampleTemplate
from typing import Dict, List, Type
from response import FPIdentificationResponse, Response
from constant.constant import MODEL_FP_IDENTIFICATION_KEY


class FPIdentificationFewShotExample(FewShotExampleTemplate):
    question: str
    is_normal: bool
    user_role: str
    model_role: str

    def __init__(
        self,
        question: str,
        is_normal: bool,
        user_role: str = "user",
        model_role: str = "assistant",
        **kwargs
    ):
        self.question = question
        self.is_normal = is_normal
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[str]:
        answer = "No" if self.is_normal else "Yes"
        return [
            (
                f"{self.user_role}: Input: {self.question}\n"
                "Question: Does the input contain any false assumptions?\n"
                f"{self.model_role}: {answer}"
            )
        ]


class FPIdentificationTemplate(TemplateWithKey):
    ResponseClass: Type[Response] = FPIdentificationResponse
    answer_key: str = MODEL_FP_IDENTIFICATION_KEY
    question: str
    few_shot_data: List[FPIdentificationFewShotExample]
    system_role: str
    model_role: str
    user_role: str

    def __init__(
        self,
        question: str,
        few_shot_data: List[Dict],
        passages: List[str],
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        self.question = question
        self.system_role = system_role
        self.model_role = model_role
        self.user_role = user_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FPIdentificationFewShotExample(
                **dp,
                user_role=self.user_role,
                model_role=self.model_role
            ).generate()
        self.passages = passages

    def generate(self, **kwargs) -> List[Dict]:
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that helps identify false assumptions.
                    {f"Use the information from the evidence to help you identify the false assumption." if len(self.passages) > 0 else ""}
                    Output Yes if the question has false assumptions; otherwise, output No.
                    {"Below are some examples to help you understand the task." if len(self.few_shot_data) > 0 else ""}
                    {'\n\n'.join(self.few_shot_data)}
                """
            },
            {
                "role": self.user_role,
                "content": (
                    f"Input: {self.question}\n"
                    "Question: Does the input contain any false assumptions?\n"
                    f"Evidence: {' ||| '.join(self.passages)}" if len(self.passages) > 0 else ""
                )
            }
        ]
        return messages
