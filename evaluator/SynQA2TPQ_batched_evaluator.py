from .response_level_score_batched_evaluator import ResponseLevelScoreBatchedEvaluator
from constant.response_level_score import DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL
from .registry import register

@register()
class SynQA2TPQResponseLevelScoreBatchedEvaluator(ResponseLevelScoreBatchedEvaluator):
    def __init__(
        self,
        evaluator_model_name: str = DEFAULT_RESPONSE_LEVEL_EVALUATOR_MODEL,
        thinking_cutoff_token: str = None,
        **kwargs
    ):
        super().__init__(
            dataset_name="SynQA2TPQ",
            evaluator_model_name=evaluator_model_name,
            thinking_cutoff_token=thinking_cutoff_token,
            **kwargs
        )
