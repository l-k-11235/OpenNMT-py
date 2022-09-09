from .scorer import Scorer
from onmt.scorers import register_scorer
# load sentencepiece before comet to avoid comet errors
import sentencepiece
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
        data = []
        for i in range(len(preds)):
            data.append({
                "src": texts_src[i],
                "mt": preds[i],
                "ref": texts_ref[i]
            })
        seg_scores, sys_score = self.model.predict(
            data, batch_size=1, num_workers=0)
        return 100*sys_score
