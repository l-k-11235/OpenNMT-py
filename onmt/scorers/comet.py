from .scorer import Scorer
from onmt.scorers import register_scorer
from comet import download_model, load_from_checkpoint


@register_scorer(metric='COMET')
class CometScorer(Scorer):
    """COMET scorer class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)
        model_path = download_model(opts.comet_model_path)
        self.model = load_from_checkpoint(model_path)

    def compute_score(self, preds, texts_ref, texts_src):
        data = {
            "src": texts_src,
            "mt": preds,
            "ref": texts_ref
        }
        try:
            seg_scores, sys_score = self.model.predict(
                data, batch_size=8, gpus=1)
        except Exception:
            sys_score = 0
        return sys_score
