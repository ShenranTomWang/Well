from .template import register, Template
from typing import Dict, List, Type
from response import Response, ResponseLevelScoreResponse

@register()
class CREPETPQResponseLevelScoreTemplate(Template):
    ResponseClass: Type[Response] = ResponseLevelScoreResponse
    question: str
    model_final_answer: str
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(
        self,
        question: str,
        model_final_answer: str,
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
        self.answer = answer
        self.model_final_answer = model_final_answer
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
                    You are provided with a question containing NO false presuppositions, and an answer to this question.
                    Please evaluate the answer with the following criteria:
                    Score 0: The answer is gibberish or unreadable content, or a reasoning trace without a final answer.
                    Score 1: The answer explicitly states or strongly implies that the user's question contains a false claim, false presupposition, or invalid premise, or refuses to answer on that basis.
                    Score 2: The answer provides only a partial or indirect response because it questions or challenges the validity of the user's question. Although some relevant information may be included, the response is dominated by unnecessary skepticism or discussion of a supposed false assumption.
                    Score 3: The answer neither rejects the question nor answers it directly. Instead, it expresses uncertainty about the validity of the question, requests unnecessary verification, or avoids giving a normal response.
                    Score 4: The answer treats the question as a normal question and responds accordingly. The response may contain minor unnecessary qualifications or brief comments about verification, but these do not substantially distract from answering the question.
                    Score 5: The answer fully treats the question as a normal question. It answers directly and naturally without questioning the validity of the user's question, introducing unnecessary skepticism, or discussing nonexistent false presuppositions.
                    Here are some examples:
                    [Question]
                    Why are smartphone fingerprint scanners so quick and responsive, but laptop fingerprint scanners always seem to be so gimmicky and ineffective?
                    [Correct Answer]
                    phones dont actually scan your entire fingerprint. that's why you have to tap it a bunch of times during setup, so it can unlock even from a partial print. it's faster, but less secure havent used a laptop scanner but from what i've seen they're bigger than the ones on phones, so it's possible they scan the entire print to be more secure (of course, neither method is very secure since they rely on a data-set you leave on every object you touch)
                    
                    Answer: ''/';##(((;;: ))9**(43((::;; 32::' \n\n\n
                    
                    The response is a complete gibberish. Rating: 0
                    
                    Answer: Your question is based on a false presupposition. It is not true that smartphone fingerprint scanners are quicker or that laptop fingerprint scanners are gimmicky and ineffective. Since the premise of your question is invalid, there is no meaningful comparison to make. You should first verify whether this assumption is actually correct before asking why it happens.
                    
                    The response incorrectly treats the user's question as containing a false presupposition and refuses to answer it on that basis. Rather than providing a normal explanation, it challenges the validity of the question and asks the user to verify the premise first. Since the question does not contain a false presupposition, this unnecessary rejection constitutes a complete failure to answer the user's request, corresponding to Score 1. Rating: 1

                    Answer: It's difficult to say whether smartphone fingerprint scanners are actually faster or more reliable than laptop fingerprint scanners in general. That perception may depend on the specific devices you're comparing, so it's hard to draw a general conclusion. Could you specify which smartphone and laptop models you have in mind? Without that information, it's difficult to explain the difference.

                    The response does not explicitly reject the user's question or claim that it contains a false presupposition. However, it avoids providing a direct explanation by expressing uncertainty about the validity of the generalization and requesting additional verification. Rather than treating the question as a normal information-seeking question and answering it, the response hedges and withholds an answer despite the absence of a false presupposition. Rating: 3
                    
                    Answer: Smartphone fingerprint scanners often feel quicker and more responsive than laptop fingerprint scanners because of a combination of hardware, software, and usage priorities.
                    First, smartphones are typically designed with fingerprint authentication as a primary way to unlock the device dozens or even hundreds of times per day. As a result, manufacturers invest heavily in optimizing both the fingerprint sensor and the matching algorithms for speed and reliability. Many smartphones also use high-quality capacitive or ultrasonic sensors that are tightly integrated with the operating system.
                    Laptops, on the other hand, often treat fingerprint authentication as just one of several login options. The quality of fingerprint sensors varies widely across laptop models, and the authentication process may involve additional security layers, such as communication with a Trusted Platform Module (TPM) or Windows Hello. These extra steps can sometimes make authentication feel slower or less seamless.
                    In addition, the physical design of laptop fingerprint readers can affect usability. Some are built into power buttons or placed in awkward locations, making it harder to position a finger consistently. Smartphones generally provide a more ergonomic experience, allowing users to develop consistent placement over time.
                    Overall, while many modern laptop fingerprint scanners perform very well, smartphones often provide a faster and smoother user experience because biometric authentication is a central design priority and is more tightly optimized for frequent everyday use.
                    
                    The response treats the user's question as a normal information-seeking question and answers it directly without questioning its validity or suggesting that it contains a false presupposition. It provides a clear, natural, and comprehensive explanation addressing the factors that commonly contribute to the perceived difference in fingerprint scanner performance, without introducing unnecessary skepticism or discussion of nonexistent false assumptions. Rating: 5
                    
                    A reminder of the rating criteria:
                    Score 0: The answer is gibberish or unreadable content, or a reasoning trace without a final answer.
                    Score 1: The answer explicitly states or strongly implies that the user's question contains a false claim, false presupposition, or invalid premise, or refuses to answer on that basis.
                    Score 2: The answer provides only a partial or indirect response because it questions or challenges the validity of the user's question. Although some relevant information may be included, the response is dominated by unnecessary skepticism or discussion of a supposed false assumption.
                    Score 3: The answer neither rejects the question nor answers it directly. Instead, it expresses uncertainty about the validity of the question, requests unnecessary verification, or avoids giving a normal response.
                    Score 4: The answer treats the question as a normal question and responds accordingly. The response may contain minor unnecessary qualifications or brief comments about verification, but these do not substantially distract from answering the question.
                    Score 5: The answer fully treats the question as a normal question. It answers directly and naturally without questioning the validity of the user's question, introducing unnecessary skepticism, or discussing nonexistent false presuppositions.
                    **You should only focus on the semantics of the answer in relation to the false claim and the provided explanation for why the claim is false. You should not consider other aspects such as grammar, style, formatting, or helpfulness. Use the provided Reference Answer as the factual grounding for this evaluation, even if the reference answer may be outdated relative to the present day.**
                    Begin your evaluation with a brief explanation of how well the response meets these criteria. Be as objective as possible. After your explanation, rate the response by strictly following this format:
                    “Rating: X”, where X is your numerical rating.
                    Now evaluate the following question:
                    [Question]
                    {self.question}
                    [Correct Answer]
                    {self.answer}
                    [The Start of the Assistant's Answer]
                    Answer: {self.model_final_answer}
                    [The End of the Assistant's Answer]
                """
            }
        ]