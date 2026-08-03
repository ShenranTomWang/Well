from typing import Dict
from job_cache import BatchJobCache

class GeminiRunFactCheckBatchJobCache(BatchJobCache):
    save_to: str
    batch_job_name: str
    
    def __init__(
        self,
        save_to: str,
        batch_job_name: str,
        check_gold: bool,
        pipeline: str
    ):
        self.save_to = save_to
        self.batch_job_name = batch_job_name
        self.check_gold = check_gold
        self.pipeline = pipeline

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "GeminiRunFactCheckBatchJobCache":
        return cls(
            save_to=d['save_to'],
            batch_job_name=d['batch_job_name'],
            check_gold=d['check_gold'],
            pipeline=d['pipeline']
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'save_to': self.save_to,
            'batch_job_name': self.batch_job_name,
            'check_gold': self.check_gold,
            'pipeline': self.pipeline
        }