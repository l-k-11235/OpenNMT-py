import json
import numpy as np
from onmt.constants import CorpusTask, DefaultTokens, ModelTask
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.utils.distributed import ErrorHandler, spawned_infer
from onmt.utils.logging import logger
from onmt.transforms import get_transforms_cls, make_transforms, TransformPipe


class InferenceEngine(object):
    """Wrapper Class to run Inference.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        self.opt = opt

    def translate_batch(self, batch):
        pass

    def _translate(self, infer_iter):
        pass

    def infer_file(self):
        """File inference. Source file must be the opt.src argument"""
        if self.opt.world_size <= 1:
            infer_iter = build_dynamic_dataset_iter(
                self.opt,
                self.transforms_cls,
                self.vocabs,
                task=CorpusTask.INFER,
                device_id=self.device_id,
            )
            translated_bucket = self._translate(infer_iter)
            translated_results = [
                translated_bucket[i] for i in range(len(translated_bucket))
            ]
            print('### translated_results:', translated_results)
            return translated_results
        # else:
        #     scores, preds = self.infer_file_parallel()
        # return scores, preds

    def infer_list(self, src):
        """List of strings inference `src`"""
        print("infer_list")
        print(len(src))
        if self.opt.world_size <= 1:
            infer_iter = build_dynamic_dataset_iter(
                self.opt,
                self.transforms_cls,
                self.vocabs,
                task=CorpusTask.INFER,
                src=src,
                device_id=self.device_id,
            )
            translated_bucket = self._translate(infer_iter)
            translated_results = [
                translated_bucket[i] for i in range(len(translated_bucket))
            ]
            print(len(translated_results))
            return translated_results
        # else:
        #     scores, preds = self.infer_list_parallel(src)
        # return scores, preds

    def score_list(self, src, tgt):
        """List of strings inference `src`"""
        print("score_list")
        print(len(src), len(tgt))
        if self.opt.world_size <= 1:
            infer_iter = build_dynamic_dataset_iter(
                self.opt,
                self.transforms_cls,
                self.vocabs,
                task=CorpusTask.INFER,
                src=src,
                tgt=tgt,
                device_id=self.device_id,
            )
            scored_bucket = self.score(infer_iter)
            score_results = [scored_bucket[i] for i in range(len(scored_bucket))]
        print(len(scored_bucket))
        return score_results

    def infer_file_parallel(self):
        """File inference in mulitprocessing with partitioned models."""
        raise NotImplementedError(
            "The inference in mulitprocessing with partitioned models is not implemented."
        )

    def infer_list_parallel(self, src):
        """The inference in mulitprocessing with partitioned models."""
        raise NotImplementedError(
            "The inference in mulitprocessing with partitioned models is not implemented."
        )

    def terminate(self):
        pass


class InferenceEnginePY(InferenceEngine):
    """Inference engine subclass to run inference with `translate.py`.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        import torch
        from onmt.translate.translator import build_translator

        super().__init__(opt)
        self.opt = opt

        if opt.world_size > 1:
            mp = torch.multiprocessing.get_context("spawn")
            # Create a thread to listen for errors in the child processes.
            self.error_queue = mp.SimpleQueue()
            self.error_handler = ErrorHandler(self.error_queue)
            self.queue_instruct = []
            self.queue_result = []
            self.procs = []

            print("world_size: ", opt.world_size)
            print("gpu_ranks: ", opt.gpu_ranks)
            print("opt.gpu: ", opt.gpu)

            for device_id in range(opt.world_size):
                self.queue_instruct.append(mp.Queue())
                self.queue_result.append(mp.Queue())
                self.procs.append(
                    mp.Process(
                        target=spawned_infer,
                        args=(
                            opt,
                            device_id,
                            self.error_queue,
                            self.queue_instruct[device_id],
                            self.queue_result[device_id],
                        ),
                        daemon=False,
                    )
                )
                self.procs[device_id].start()
                print(" Starting process pid: %d  " % self.procs[device_id].pid)
                self.error_handler.add_child(self.procs[device_id].pid)
        else:
            self.device_id = 0 if opt.world_size == 1 else -1
            self.translator = build_translator(
                opt, self.device_id, logger=logger, report_score=True
            )
            self.vocabs = self.translator.vocabs
            self.transforms_cls = get_transforms_cls(opt._all_transform)
            transforms = make_transforms(opt, self.transforms_cls, self.vocabs)
            self.transform = TransformPipe.build_from(transforms.values())

    # def _translate(self, infer_iter):
    #     scores, preds = self.translator._translate(
    #         infer_iter, infer_iter.transforms, self.opt.attn_debug, self.opt.align_debug
    #     )
    # return scores, preds

    def _translate(self, infer_iter):
        translated_bucket = {}
        for batch, bucket_idx in infer_iter:
            batch_inds_in_bucket = batch["ind_in_bucket"].cpu().tolist()

            batch_data = self.translator.translate_batch(batch, attn_debug=False)
            batch_scores = [_score[0].cpu().tolist() for _score in batch_data["scores"]]
            batch = batch_data["batch"]
            for i, _ in enumerate(batch["src"]):
                ind_in_bucket = batch_inds_in_bucket[i]
                src_tok_ids = batch["src"][i, :, 0].cpu().numpy().tolist()
                src_tokens = [
                    self.vocabs["src"].lookup_index(id)
                    for id in src_tok_ids
                    if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                ]
                hyp_tok_ids = [
                    _pred.cpu().numpy().tolist()
                    for _pred in batch_data["predictions"][i]
                ]
                hyp_tokens = [
                    [
                        self.vocabs["src"].lookup_index(id)
                        for id in ids
                        if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                    ]
                    for ids in hyp_tok_ids
                ]
                hyps = [self.transform.apply_reverse(tokens) for tokens in hyp_tokens]
                translated_bucket[ind_in_bucket] = {
                    "scores": batch_scores[i],
                    "preds": hyps,
                    "hyp_tokens": hyp_tokens,
                    "src_tokens": src_tokens,
                }
                print(ind_in_bucket, src_tokens)
                print(hyp_tok_ids)
                print(hyp_tokens)
        return translated_bucket

    def score(self, infer_iter):
        self.translator.with_scores = True
        scored_bucket = {}
        for batch, bucket_idx in infer_iter:
            batch_data = self.translator.translate_batch(batch, attn_debug=False)
            batch_scores = [_score[0].cpu().tolist() for _score in batch_data["scores"]]
            batch_inds_in_bucket = batch["ind_in_bucket"].cpu().tolist()
            src_lengths = batch["srclen"].cpu().tolist()
            batch = batch_data["batch"]
            for i, _ in enumerate(batch["src"]):
                ind_in_bucket = batch_inds_in_bucket[i]
                tgt_tok_ids = batch["tgt"][i, :, 0].cpu().numpy().tolist()
                tgt_tokens = [
                    self.vocabs["src"].lookup_index(id)
                    for id in tgt_tok_ids
                    if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                ]
                src_tok_ids = batch["src"][i, :, 0].cpu().numpy().tolist()
                src_tokens = [
                    self.vocabs["src"].lookup_index(id)
                    for id in src_tok_ids
                    if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                ]
                ppl = np.exp(-batch_scores[i] / (src_lengths[i] + len(tgt_tokens)))
                scored_bucket[ind_in_bucket] = {
                    "score": batch_scores[i],
                    "ppl": ppl,
                    "tgt_tokens": tgt_tokens,
                    "src_tokens": src_tokens,
                }
        return scored_bucket

    def infer_file_parallel(self):
        assert self.opt.world_size > 1, "World size must be greater than 1."
        for device_id in range(self.opt.world_size):
            self.queue_instruct[device_id].put(("infer_file", self.opt))
        scores, preds = [], []
        for device_id in range(self.opt.world_size):
            scores.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], preds[0]

    def infer_list_parallel(self, src):
        for device_id in range(self.opt.world_size):
            self.queue_instruct[device_id].put(("infer_list", src))
        scores, preds = [], []
        for device_id in range(self.opt.world_size):
            scores.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], preds[0]

    def terminate(self):
        if self.opt.world_size > 1:
            for device_id in range(self.opt.world_size):
                self.queue_instruct[device_id].put(("stop"))
                self.procs[device_id].terminate()


class InferenceEngineCT2(InferenceEngine):
    """Inference engine subclass to run inference with ctranslate2.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        import ctranslate2
        import pyonmttok

        super().__init__(opt)
        self.opt = opt
        self.logger = logger
        assert self.opt.world_size <= 1, "World size must be less than 1."
        self.device_id = 0 if opt.world_size == 1 else -1
        if opt.world_size == 1:
            self.device_index = opt.gpu_ranks
            self.device = "cuda"
        else:
            self.device_index = 0
            self.device = "cpu"
        self.transforms_cls = get_transforms_cls(self.opt._all_transform)
        # Build translator
        if opt.model_task == ModelTask.LANGUAGE_MODEL:
            self.translator = ctranslate2.Generator(
                opt.models[0], device=self.device, device_index=self.device_index
            )
        else:
            self.translator = ctranslate2.Translator(
                self.opt.models[0], device=self.device, device_index=self.device_index
            )
        # Build vocab
        vocab_path = opt.src_subword_vocab
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        vocabs = {}
        src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab
        vocabs["data_task"] = "lm"
        vocabs["decoder_start_token"] = "<s>"
        self.vocabs = vocabs
        # Build transform pipe
        transforms = make_transforms(opt, self.transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def translate_batch(self, batch, opt):
        input_tokens = []
        for i in range(batch["src"].size()[0]):
            start_ids = batch["src"][i, :, 0].cpu().numpy().tolist()
            _input_tokens = [
                self.vocabs["src"].lookup_index(id)
                for id in start_ids
                if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
            ]
            input_tokens.append(_input_tokens)
        if opt.model_task == ModelTask.LANGUAGE_MODEL:
            translated_batch = self.translator.generate_batch(
                start_tokens=input_tokens,
                batch_type=("examples" if opt.batch_type == "sents" else "tokens"),
                max_batch_size=opt.batch_size,
                beam_size=opt.beam_size,
                num_hypotheses=opt.n_best,
                max_length=opt.max_length,
                return_scores=True,
                include_prompt_in_result=False,
                sampling_topk=opt.random_sampling_topk,
                sampling_topp=opt.random_sampling_topp,
                sampling_temperature=opt.random_sampling_temp,
            )
            preds = [
                [self.transform.apply_reverse(tokens) for tokens in out.sequences]
                for out in translated_batch
            ]
            scores = [out.scores for out in translated_batch]
        elif opt.model_task == ModelTask.SEQ2SEQ:
            translated_batch = self.translator.translate_batch(
                input_tokens,
                batch_type=("examples" if opt.batch_type == "sents" else "tokens"),
                max_batch_size=opt.batch_size,
                beam_size=opt.beam_size,
                num_hypotheses=opt.n_best,
                max_decoding_length=opt.max_length,
                return_scores=True,
                sampling_topk=opt.random_sampling_topk,
                sampling_topp=opt.random_sampling_topp,
                sampling_temperature=opt.random_sampling_temp,
            )
            preds = [
                [self.transform.apply_reverse(tokens) for tokens in out.hypotheses]
                for out in translated_batch
            ]
            scores = [out.scores for out in translated_batch]

        return scores, preds

    def _translate(self, infer_iter):
        translated_bucket = {}
        for batch, bucket_idx in infer_iter:
            batch_inds_in_bucket = batch["ind_in_bucket"].cpu().tolist()
            batch_scores, batch_preds = self.translate_batch(batch, self.opt)
            for i, _ in enumerate(batch["src"]):
                ind_in_bucket = batch_inds_in_bucket[i]
                translated_bucket[ind_in_bucket] = {
                    "scores": batch_scores[i],
                    "preds": batch_preds[i],
                }
        return translated_bucket

    def score(self, infer_iter):
        scored_bucket = {}
        for batch, bucket_idx in infer_iter:
            batch_inds_in_bucket = batch["ind_in_bucket"].cpu().tolist()
            input_tokens = []
            for i, _ in enumerate(batch["src"]):
                src_tok_ids = batch["src"][i, :, 0].cpu().numpy().tolist()
                tgt_tok_ids = batch["tgt"][i, :, 0].cpu().numpy().tolist()
                tgt_tokens = [
                    self.vocabs["src"].lookup_index(id)
                    for id in tgt_tok_ids
                    if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                ]
                src_tokens = [
                    self.vocabs["src"].lookup_index(id)
                    for id in src_tok_ids
                    if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                ]
                tgt_tokens = [
                    self.vocabs["src"].lookup_index(id)
                    for id in tgt_tok_ids
                    if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                ]
                input_tokens.append(src_tokens + tgt_tokens)
            batch_outputs = self.translator.score_batch(input_tokens)
            for j, out in enumerate(batch_outputs):
                score = sum(out.log_probs)
                ppl = np.exp(-np.mean(out.log_probs))
                ind_in_bucket = batch_inds_in_bucket[j]
                scored_bucket[ind_in_bucket] = {
                    "ppl": ppl,
                    "score": score,
                }
        return scored_bucket
