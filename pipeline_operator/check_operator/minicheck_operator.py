from typing import Dict
from constant.constant import (
    FACTCHECK_RESULTS_KEY,
    MODEL_DETECTED_PRESUPPOSITIONS_KEY,
    MODEL_CONVERTED_STATEMENT_KEY,
)
from utils.RAG_utils import get_passages
from .check_operator import CheckOperator, register
from minicheck.minicheck import MiniCheck

@register()
class MiniCheckOperator(CheckOperator):
    def __init__(self, model_name: str, cache_dir: str, statement: bool = False):
        self.model = MiniCheck(model_name=model_name, cache_dir=cache_dir)
        self.statement = statement

    def _get_presuppositions(self, dp: Dict[str, any], check_gold: bool):
        if check_gold:
            presuppositions = dp["presuppositions"]
        else:
            detected = dp[MODEL_DETECTED_PRESUPPOSITIONS_KEY]
            presuppositions = detected["presuppositions"] if isinstance(detected, dict) else detected
        return [presupposition for presupposition in presuppositions if presupposition.strip() != ""]

    def _get_statements(self, dp: Dict[str, any]):
        statement = dp[MODEL_CONVERTED_STATEMENT_KEY]
        statements = [statement] if isinstance(statement, str) else statement
        return [statement for statement in statements if statement.strip() != ""]
    
    def check(self, dp: Dict[str, any], source: str, check_gold: bool, **kwargs) -> Dict[str, any]:
        if self.statement:
            claims = self._get_statements(dp)
            instruction = 'Given a statement, retrieve relevant passages that validate or refute the statement'
        else:
            claims = self._get_presuppositions(dp, check_gold=check_gold)
            instruction = 'Given a list of statements, retrieve relevant passages that validate or refute the statements'

        passages = get_passages(dp, claims=claims, source=source, instruction=instruction, **kwargs)
        passages = " ||| ".join(passages)
        pred_labels = []
        for claim in claims:
            if passages.strip() == "":
                pred_labels.append(0)
            else:
                scored_labels, _, _, _ = self.model.score(docs=[passages], claims=[claim])
                pred_labels.extend(scored_labels)
        dp[FACTCHECK_RESULTS_KEY] = pred_labels
        return dp
