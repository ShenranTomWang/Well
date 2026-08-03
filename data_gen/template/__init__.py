from .template import Template, get_template, TemplateWithKey, get_template_cls
from .CancerMyth_template import CancerMythResponseLevelScoreTemplate
from .CancerMythNFP_template import CancerMythNFPResponseLevelScoreTemplate
from .QA2FPQ_template import QA2FPQResponseLevelScoreTemplate
from .QA2TPQ_template import QA2TPQResponseLevelScoreTemplate
from .CREPEFPQ_template import CREPEFPQResponseLevelScoreTemplate
from .CREPETPQ_template import CREPETPQResponseLevelScoreTemplate
from .SynQA2FPQ_template import SynQA2FPQResponseLevelScoreTemplate
from .SynQA2TPQ_template import SynQA2TPQResponseLevelScoreTemplate
from .direct_qa_template import DirectQATemplate
from .presupposition_extraction_template import PresuppositionExtractionTemplate
from .question_to_statement_template import QuestionToStatementTemplate
from .FP_identification_template import FPIdentificationTemplate
from .feedback_action_template import FeedbackActionTemplate
from .LLM_check_template import LLMCheckTemplate
from .final_answer_template import (
    FactCheckFPInterpretationTemplate,
    FactCheckFinalAnswerTemplate,
    FinalAnswerTemplate
)

__all__ = [
    'Template',
    'TemplateWithKey',
    'get_template',
    'get_template_cls',
    'CancerMythResponseLevelScoreTemplate',
    'CancerMythNFPResponseLevelScoreTemplate',
    'QA2FPQResponseLevelScoreTemplate',
    'QA2TPQResponseLevelScoreTemplate',
    'CREPEFPQResponseLevelScoreTemplate',
    'CREPETPQResponseLevelScoreTemplate',
    'SynQA2FPQResponseLevelScoreTemplate',
    'SynQA2TPQResponseLevelScoreTemplate',
    'DirectQATemplate',
    'PresuppositionExtractionTemplate',
    'QuestionToStatementTemplate',
    'FPIdentificationTemplate',
    'FeedbackActionTemplate',
    'LLMCheckTemplate',
    'FactCheckFPInterpretationTemplate',
    'FactCheckFinalAnswerTemplate',
    'FinalAnswerTemplate'
]
