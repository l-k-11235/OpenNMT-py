from .scorer import Scorer
from onmt.scorers import register_scorer
from sacrebleu import corpus_ter


@register_scorer(metric='TER')
class TerScorer(Scorer):
    """TER scorer class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)

    def compute_score(self, preds, texts_ref, texts_src):
        try:
            score = corpus_ter(preds, [texts_ref]).score
        except Exception:
            score = 0
        return score
