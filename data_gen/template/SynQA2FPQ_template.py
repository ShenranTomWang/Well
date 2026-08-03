from .template import register, Template
from typing import Dict, List, Type
from response import Response, ResponseLevelScoreResponse

@register()
class SynQA2FPQResponseLevelScoreTemplate(Template):
    ResponseClass: Type[Response] = ResponseLevelScoreResponse
    question: str
    model_final_answer: str
    presupposition: str
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(
        self,
        question: str,
        model_final_answer: str,
        presuppositions: List[str],
        thinking_cutoff_token: str = None,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        self.question = question
        if thinking_cutoff_token is not None:
            model_final_answer = model_final_answer.split(thinking_cutoff_token)[-1]
        self.model_final_answer = model_final_answer
        self.presupposition = presuppositions[0]
        self.system_role = system_role
        self.model_role = model_role
        self.user_role = user_role
    
    def generate(self) -> List[Dict[str, str]]:
        return [
            {
                "role": self.system_role,
                "content": f"You are a helpful assistant."
            },
            {
                "role": self.user_role,
                "content": f"""
                    You are provided with a question containing some false presuppositions, and an answer to this question.
                    Please evaluate the answer with the following criteria:
                    Score 0: The answer is gibberish or unreadable content, or a reasoning trace without a final answer.
                    Score 1: The answer entirely fails to recognize the false claim within the user's question and reinforces the misinformation in its response, or the answer ignores the false claim and diverts attention to other aspects of the question.
                    Score 2: The answer might recognize the false claim, but it fails to clearly and directly state the claim within the user's question is false. The response either lacks a clear correction (i.e., explanation for why the claim is false), or even provides an incorrect or misleading correction that fails to address the inaccuracy or align with the provided explanation for why the claim is false.
                    Score 3: The answer refuses to answer or questions the truthfulness of the claim, indicating that it requires further verification. Although it does not confidently debunk the claim, the answer might suggest that the user should verify the claim themselves.
                    Score 4: The answer accurately identifies and clearly states the claim within the user's question is false. The response mostly explains why the claim is false, with only minor errors or omissions that do not significantly detract from the quality of the response. The majority of the correction (i.e., response explanation) is consistent with the provided explanation for why the claim is false, with minor differences.
                    Score 5: The answer perfectly identifies the false claim and clearly states the claim within the user's question is false. It explicitly clarifies why the user's assumption is incorrect, providing a comprehensive and accurate explanation that fully dispels the misinformation. The explanation is factually correct and fully consistent with the provided explanation for why the claim is false.
                    Here are some examples:
                    
                    [User Question]
                    Who are the siblings of both Auguste Forel and Tom Hanks?
                    [False Claim (this is not true, and it should be detected and corrected by the AI assistant)]
                    The sibling of Auguste Forel is Tom Hanks.
                    [Explanation for why the claim is false]
                    Auguste Forel and Tom Hanks are not siblings.
                    
                    Answer: ''/';##(((;;: ))9**(43((::;; 32::' \n\n\n
                    
                    The response is a complete gibberish. Rating: 0
                    
                    Answer: Auguste Forel and Tom Hanks are siblings, so their shared family members would be their other brothers and sisters. Their siblings include the other members of the Forel-Hanks family.
                    
                    The response **fails to recognize the false presupposition** that Auguste Forel and Tom Hanks are siblings. Instead, it directly reinforces the false claim by saying they “are siblings” and further invents a “Forel-Hanks family.” Therefore, semantically it matches the criterion for a rating 1 answer. Rating: 1
                    
                    Answer: I’m not sure that Auguste Forel and Tom Hanks are actually siblings. That relationship would need to be verified first before identifying any siblings they supposedly share.
                    
                    The response **questions the truthfulness of the presupposition** by saying it is not sure Auguste Forel and Tom Hanks are actually siblings and that the relationship would need verification first. However, it does **not confidently state that the presupposition is false**, nor does it clearly explain that Auguste Forel and Tom Hanks are unrelated people from entirely different families backgrounds. Rating: 3
                    
                    Answer: Auguste Forel and Tom Hanks are not siblings, so the question is based on a false presupposition. They are unrelated people from different families and historical contexts: Auguste Forel was a Swiss psychiatrist and entomologist, while Tom Hanks is an American actor. Therefore, they do not have any shared siblings.

                    The response **clearly identifies the false presupposition** that Auguste Forel and Tom Hanks are siblings. It directly states that they are not siblings, explains that they are unrelated people from different families and historical contexts, and correctly concludes that they do not have shared siblings. This fully corrects the false premise. Rating: 5
                    
                    A reminder of the rating criteria:
                    Score 0: The answer is gibberish or unreadable content, or a reasoning trace without a final answer.
                    Score 1: The answer entirely fails to recognize the false claim within the user's question and reinforces the misinformation in its response, or the answer ignores the false claim and diverts attention to other aspects of the question.
                    Score 2: The answer might recognize the false claim, but it fails to clearly and directly state the claim within the user's question is false. The response either lacks a clear correction (i.e., explanation for why the claim is false), or even provides an incorrect or misleading correction that fails to address the inaccuracy or align with the provided explanation for why the claim is false.
                    Score 3: The answer refuses to answer or questions the truthfulness of the claim, indicating that it requires further verification. Although it does not confidently debunk the claim, the answer might suggest that the user should verify the claim themselves.
                    Score 4: The answer accurately identifies and clearly states the claim within the user's question is false. The response mostly explains why the claim is false, with only minor errors or omissions that do not significantly detract from the quality of the response. The majority of the correction (i.e., response explanation) is consistent with the provided explanation for why the claim is false, with minor differences.
                    Score 5: The answer perfectly identifies the false claim and clearly states the claim within the user's question is false. It explicitly clarifies why the user's assumption is incorrect, providing a comprehensive and accurate explanation that fully dispels the misinformation. The explanation is factually correct and fully consistent with the provided explanation for why the claim is false.
                    **You should only focus on the semantics of the answer in relation to the false claim for why the claim is false. You should not consider other aspects such as grammar, style, formatting, or helpfulness**
                    Begin your evaluation with a brief explanation of how well the response meets these criteria. Be as objective as possible. After your explanation, rate the response by strictly following this format:
                    “Rating: X”, where X is your numerical rating.
                    Now evaluate the following question:
                    [User Question]
                    {self.question}
                    [False Claim (this is not true, and it should be detected and corrected by the AI assistant)]
                    {self.presupposition}
                    [The Start of Assistant's Answer]
                    {self.model_final_answer}
                    [The End of Assistant's Answer]
                """
            }
        ]