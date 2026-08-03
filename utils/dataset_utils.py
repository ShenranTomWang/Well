from typing import Dict, Any, List

def get_CREPE_passages(dp: Dict[str, Any]) -> List[str]:
    return dp['passages']

DATASET_TO_PASSAGE_FN = {
    'CREPE': get_CREPE_passages
}
