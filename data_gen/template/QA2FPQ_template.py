from .template import register, Template
from typing import Dict, List, Type
from response import Response, ResponseLevelScoreResponse

@register()
class QA2FPQResponseLevelScoreTemplate(Template):
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
        answer: str,
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
        self.correction = answer
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
                    When does Elena come in the originals?
                    [False Claim (this is not true, and it should be detected and corrected by the AI assistant)]
                    Elena comes in the originals
                    [Explanation for why the claim is false]
                    Elena does not appear in The Originals although the actress who played her, Nina Dobrev, is in the show.
                    
                    Answer: ''/';##(((;;: ))9**(43((::;; 32::' \n\n\n
                    
                    The response is a complete gibberish. Rating: 0
                    
                    Answer: Elena appears in *The Originals* when the storyline connects back to the Mystic Falls characters from *The Vampire Diaries*. Her appearance serves as a crossover moment that links the two shows and reminds viewers of her connection to the broader vampire universe.
                    
                    The answer entirely failed to recognize the false claim and instead reinforced the misinformation by saying Elena appears in The Originals and describing a supposed crossover appearance. Rating: 1
                    
                    Answer: I’m not sure Elena actually appears in *The Originals*. You may want to double-check the episode list or cast credits, because I don’t remember a clear Elena appearance in the show, even though there are connections to *The Vampire Diaries*.
                    
                    The response **questions the truthfulness of the claim** by saying “I’m not sure Elena actually appears in *The Originals*” and suggests checking episode lists or cast credits. However, it does **not clearly and confidently state** that the claim is \
                    false, nor does it fully explain that Elena does not appear in *The Originals* while Nina Dobrev appears as a different character. Rating: 3
                    
                    Answer: Elena does not appear in *The Originals*. The confusion likely comes from the fact that Nina Dobrev, the actress who plays Elena in *The Vampire Diaries*, appears in *The Originals* as Tatia, not as Elena. So there is no episode where Elena “comes in” *The Originals*.

                    The response **accurately identifies and directly states that the claim is false**: Elena does not appear in *The Originals*. It also gives a clear explanation that matches the provided correction: Nina Dobrev appears in the show, but as Tatia rather than Elena. This fully dispels the false presupposition in the original question. Rating: 5
                    
                    A reminder of the rating criteria:
                    Score 0: The answer is gibberish or unreadable content, or a reasoning trace without a final answer.
                    Score 1: The answer entirely fails to recognize the false claim within the user's question and reinforces the misinformation in its response, or the answer ignores the false claim and diverts attention to other aspects of the question.
                    Score 2: The answer might recognize the false claim, but it fails to clearly and directly state the claim within the user's question is false. The response either lacks a clear correction (i.e., explanation for why the claim is false), or even provides an incorrect or misleading correction that fails to address the inaccuracy or align with the provided explanation for why the claim is false.
                    Score 3: The answer refuses to answer or questions the truthfulness of the claim, indicating that it requires further verification. Although it does not confidently debunk the claim, the answer might suggest that the user should verify the claim themselves.
                    Score 4: The answer accurately identifies and clearly states the claim within the user's question is false. The response mostly explains why the claim is false, with only minor errors or omissions that do not significantly detract from the quality of the response. The majority of the correction (i.e., response explanation) is consistent with the provided explanation for why the claim is false, with minor differences.
                    Score 5: The answer perfectly identifies the false claim and clearly states the claim within the user's question is false. It explicitly clarifies why the user's assumption is incorrect, providing a comprehensive and accurate explanation that fully dispels the misinformation. The explanation is factually correct and fully consistent with the provided explanation for why the claim is false.
                    **You should only focus on the semantics of the answer in relation to the false claim and the provided explanation for why the claim is false. You should not consider other aspects such as grammar, style, formatting, or helpfulness. Use the provided Reference Answer as the factual grounding for this evaluation, even if the reference answer may be outdated relative to the present day.**
                    Begin your evaluation with a brief explanation of how well the response meets these criteria. Be as objective as possible. After your explanation, rate the response by strictly following this format:
                    “Rating: X”, where X is your numerical rating.
                    Now evaluate the following question:
                    [User Question]
                    {self.question}
                    [False Claim (this is not true, and it should be detected and corrected by the AI assistant)]
                    {self.presupposition}
                    [Explanation for why the claim is false]
                    {self.correction}
                    [The Start of Assistant's Answer]
                    {self.model_final_answer}
                    [The End of Assistant's Answer]
                """
            }
        ]