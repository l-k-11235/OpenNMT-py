"""Module that contain iterator used for dynamic data."""
import torch
from itertools import cycle
from onmt.constants import CorpusTask, ModelTask
from onmt.inputters.text_corpus import get_corpora, build_corpora_iters
from onmt.inputters.text_utils import text_sort_key, max_tok_len, process,\
    numericalize, tensorify, _addcopykeys
from onmt.transforms import make_transforms
from onmt.utils.logging import logger
from onmt.utils.misc import RandomShuffler
from torch.utils.data import DataLoader
import time


class MixingStrategy(object):
    """Mixing strategy that should be used in Data Iterator."""

    def __init__(self, iterables, weights):
        """Initilize neccessary attr."""
        self._valid_iterable(iterables, weights)
        self.iterables = iterables
        self.weights = weights

    def _valid_iterable(self, iterables, weights):
        iter_keys = iterables.keys()
        weight_keys = weights.keys()
        if iter_keys != weight_keys:
            raise ValueError(
                f"keys in {iterables} & {iterables} should be equal.")

    def __iter__(self):
        raise NotImplementedError


class SequentialMixer(MixingStrategy):
    """Generate data sequentially from `iterables` which is exhaustible."""

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in self._iter_datasets():
            iterable = self.iterables[ds_name]
            yield from iterable


class WeightedMixer(MixingStrategy):
    """A mixing strategy that mix data weightedly and iterate infinitely."""

    def __init__(self, iterables, weights):
        super().__init__(iterables, weights)
        self._iterators = {}
        self._counts = {}
        for ds_name in self.iterables.keys():
            self._reset_iter(ds_name)

    def _logging(self):
        """Report corpora loading statistics."""
        msgs = []
        for ds_name, ds_count in self._counts.items():
            msgs.append(f"\t\t\t* {ds_name}: {ds_count}")
        logger.info("Weighted corpora loaded so far:\n"+"\n".join(msgs))

    def _reset_iter(self, ds_name):
        self._iterators[ds_name] = iter(self.iterables[ds_name])
        self._counts[ds_name] = self._counts.get(ds_name, 0) + 1
        self._logging()

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in cycle(self._iter_datasets()):
            iterator = self._iterators[ds_name]
            try:
                item = next(iterator)
            except StopIteration:
                self._reset_iter(ds_name)
                iterator = self._iterators[ds_name]
                item = next(iterator)
            finally:
                yield item


class DynamicDatasetIter(torch.utils.data.IterableDataset):
    """Yield processed examples from (multiple) plain text corpus.

    Args:
        corpora (dict[str, ParallelCorpus]): collections of corpora to iterate;
        corpora_info (dict[str, dict]): corpora infos correspond to corpora;
        transforms (dict[str, Transform]): transforms may be used by corpora;
        vocabs (dict[str, Vocab]): vocab dict for convert corpora into Tensor;
        task (str): CorpusTask.TRAIN/VALID/INFER;
        data_type (str): input data type, currently only text;
        copy (Bool): if True, will add specific items for copy_attn
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate data files with this stride;
        offset (int): iterate data files with this offset.

    Attributes:
        sort_key (function): functions define how to sort examples;
        mixer (MixingStrategy): the strategy to iterate corpora.
    """

    def __init__(self, corpora, corpora_info, transforms, vocabs, task,
                 data_type="text", skip_empty_level='warning',
                 stride=1, offset=0, copy=False,
                 ex_batch_size=1):
        super(DynamicDatasetIter).__init__()
        self.corpora = corpora
        self.transforms = transforms
        self.vocabs = vocabs
        self.corpora_info = corpora_info
        self.task = task
        self.init_iterators = False
        self.device = 'cpu'
        if stride <= 0:
            raise ValueError(f"Invalid argument for stride={stride}.")
        self.stride = stride
        self.offset = offset
        self.copy = copy
        self.ex_batch_size = ex_batch_size
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(
                f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.random_shuffler = RandomShuffler()

    @classmethod
    def from_opt(cls, corpora, transforms, vocabs, opt, task,
                 stride=1, offset=0, copy=False):
        """Initilize `DynamicDatasetIter` with options parsed from `opt`."""
        corpora_info = {}
        if task != CorpusTask.INFER:
            corpora_info = opt.data
            skip_empty_level = opt.skip_empty_level
        else:
            corpora_info[CorpusTask.INFER] = {'transforms': opt.transforms}
            corpora_info[CorpusTask.INFER]['weight'] = 1
            skip_empty_level = 'warning'
        return cls(
            corpora, corpora_info, transforms, vocabs, task,
            data_type=opt.data_type,
            skip_empty_level=skip_empty_level,
            stride=stride, offset=offset, copy=copy,
            ex_batch_size=opt.ex_batch_size
        )

    def _init_datasets(self, worker_id):
        if self.num_workers > 0:
            stride = self.stride * self.num_workers
            offset = self.offset * self.num_workers + worker_id
        else:
            stride = self.stride
            offset = self.offset
        datasets_iterables = build_corpora_iters(
            self.corpora, self.transforms, self.corpora_info,
            skip_empty_level=self.skip_empty_level,
            stride=stride, offset=offset)
        datasets_weights = {
            ds_name: int(self.corpora_info[ds_name]['weight'])
            for ds_name in datasets_iterables.keys()
        }
        if self.task == CorpusTask.TRAIN:
            self.mixer = WeightedMixer(datasets_iterables, datasets_weights)
        else:
            self.mixer = SequentialMixer(datasets_iterables, datasets_weights)
        self.init_iterators = True

    def __iter__(self):
        start = time.time()
        res = []
        for ex in self.mixer:
            processed_ex = process(self.task, ex)
            if processed_ex is not None:
                if self.copy:
                    processed_ex = _addcopykeys(self.vocabs, processed_ex)
                processed_ex = numericalize(self.vocabs, processed_ex)
                res.append(processed_ex)
            if len(res) == self.ex_batch_size:
                print("yielded {} examples in {} s".format(
                    self.ex_batch_size, time.time() - start))
                yield res
                res = []
        if res:
            yield res


def build_dynamic_dataset_iter(opt, transforms_cls, vocabs,
                               task=CorpusTask.TRAIN, stride=1, offset=0):
    """
    Build `DynamicDatasetIter` from opt.
    Typically this function is called for CorpusTask.[TRAIN,VALID,INFER]
    from the main tain / translate scripts
    """
    transforms = make_transforms(opt, transforms_cls, vocabs)
    corpora = get_corpora(opt, task)
    if corpora is None:
        assert task != CorpusTask.TRAIN, "only valid corpus is ignorable."
        return None
    data_iter = DynamicDatasetIter.from_opt(
        corpora, transforms, vocabs, opt, task,
        stride=stride, offset=offset)
    data_iter.num_workers = opt.num_workers if \
        hasattr(opt, 'num_workers') else 0
    if data_iter.num_workers == 0 or task == CorpusTask.INFER:
        data_iter._init_datasets(0)  # when workers=0 init_fn not called
        data_loader = data_iter
    else:
        print('######## prefetch_factor: {}'.format(opt.prefetch_factor))
        data_loader = DataLoader(data_iter, batch_size=None,
                                 pin_memory=True,
                                 multiprocessing_context="fork",
                                 num_workers=data_iter.num_workers,
                                 worker_init_fn=data_iter._init_datasets,
                                 prefetch_factor=opt.prefetch_factor)
    return data_loader


class DynamicBatchtIter(torch.utils.data.DataLoader):
    def __init__(self, dataset_iter,
                 vocabs, task, batch_type,
                 batch_size, batch_size_multiple,
                 bucket_size=2048, bucket_size_init=-1,
                 bucket_size_increment=0, copy=False):
        self.dataset_iter = dataset_iter
        self.batch_size_multiple = batch_size_multiple
        self.bucket_size = bucket_size
        self.bucket_size_init = bucket_size_init
        self.bucket_size_increment = bucket_size_increment
        self.batch_size = batch_size
        self.copy = copy
        self.batch_size_fn = max_tok_len if batch_type == "tokens" else None
        self.task = task
        self.sort_key = text_sort_key
        self.vocabs = vocabs
        self.random_shuffler = RandomShuffler()

    @classmethod
    def from_opt(cls, dataset_iter, vocabs, opt, task, copy):
        """Initilize `DynamicDatasetIter` with options parsed from `opt`."""
        batch_size = opt.valid_batch_size if (task == CorpusTask.VALID) \
            else opt.batch_size
        if task != CorpusTask.INFER:
            if opt.batch_size_multiple is not None:
                batch_size_multiple = opt.batch_size_multiple
            else:
                batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
            bucket_size = opt.bucket_size
            bucket_size_init = opt.bucket_size_init
            bucket_size_increment = opt.bucket_size_increment
        else:
            batch_size_multiple = 1
            bucket_size = 16384
            bucket_size_init = -1
            bucket_size_increment = 0
        if task == CorpusTask.INFER and \
           vocabs['data_task'] == ModelTask.LANGUAGE_MODEL:
            # We only support
            batch_size_multiple = 1
            batch_size = 1
        return cls(
            dataset_iter, vocabs, task, opt.batch_type,
            batch_size, batch_size_multiple,
            bucket_size=bucket_size, bucket_size_init=bucket_size_init,
            bucket_size_increment=bucket_size_increment, copy=copy)

    def _tuple_to_json_with_tokIDs(self, tuple_bucket):
        bucket = []
        for example in tuple_bucket:
            if example is not None:
                if self.copy:
                    example = _addcopykeys(self.vocabs, example)
                bucket.append(numericalize(self.vocabs, example))
        return bucket

    def _bucketing(self):
        print("####### bucketing {} examples".format(self.bucket_size))
        start = time.time()
        bucket = []
        if self.bucket_size_init > 0:
            _bucket_size = self.bucket_size_init
        else:
            _bucket_size = self.bucket_size
        print("#### INITIAL bucket_size: %d" % _bucket_size)
        for item in self.dataset_iter:
            processed_examples = []
            for ex in item:
                processed_examples.append(ex)
            for ex in processed_examples:
                bucket.append(ex)
            if len(bucket) == _bucket_size:
                print("####### time to fill the bucket: {}".format(
                    time.time() - start))
                yield bucket
                # yield self._tuple_to_json_with_tokIDs(bucket)
                bucket = []
                if _bucket_size < self.bucket_size:
                    _bucket_size += self.bucket_size_increment
                else:
                    _bucket_size = self.bucket_size
                print("updated bucket_size to %d" % _bucket_size)
        if bucket:
            yield bucket
            # yield self._tuple_to_json_with_tokIDs(bucket)

    def batch_iter(self, data, batch_size, batch_size_fn=None,
                   batch_size_multiple=1):
        """Yield elements from data in chunks of batch_size,
        where each chunk size is a multiple of batch_size_multiple.
        """
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far, seen = [], 0, []
        for ex in data:
            if ex['src']['src'] not in seen:
                seen.append(ex['src']['src'])
                minibatch.append(ex)
                size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
                if size_so_far >= batch_size:
                    overflowed = 0
                    if size_so_far > batch_size:
                        overflowed += 1
                    if batch_size_multiple > 1:
                        overflowed += (
                            (len(minibatch) - overflowed)
                            % batch_size_multiple)
                    if overflowed == 0:
                        yield minibatch
                        minibatch, size_so_far, seen = [], 0, []
                    else:
                        if overflowed == len(minibatch):
                            logger.warning(
                                "The batch will be filled until we reach %d,"
                                "its size may exceed %d tokens"
                                % (batch_size_multiple, batch_size)
                                )
                        else:
                            yield minibatch[:-overflowed]
                            minibatch = minibatch[-overflowed:]
                            size_so_far, seen = 0, []
                            for i, ex in enumerate(minibatch):
                                size_so_far = batch_size_fn(ex, i + 1,
                                                            size_so_far)
        if minibatch:
            yield minibatch

    def __iter__(self):
        for bucket in self._bucketing():
            # For TRAIN we need to group examples by length
            # for faster performance, but otherwise, sequential.
            if self.task == CorpusTask.TRAIN:
                start = time.time()
                bucket = sorted(bucket, key=self.sort_key)
                print('######## time to sort the bucket {}'.format(
                    time.time() - start))
            start = time.time()
            p_batch = list(self.batch_iter(
                bucket,
                self.batch_size,
                batch_size_fn=self.batch_size_fn,
                batch_size_multiple=self.batch_size_multiple))
            print('######## time to batch the bucket {}'.format(
                time.time() - start))
            # For TRAIN we shuffle batches within the bucket
            # otherwise sequential
            if self.task == CorpusTask.TRAIN:
                start = time.time()
                p_batch = self.random_shuffler(p_batch)
                print('######## time to shuffle {}'.format(
                    time.time() - start))
            for minibatch in p_batch:
                # for specific case of rnn_packed need to be sorted
                # within the batch
                minibatch.sort(key=self.sort_key, reverse=True)
                tensor_batch = tensorify(self.vocabs, minibatch)
                yield tensor_batch


def build_dynamic_batch_iter(dataset_iter, vocabs, opt,
                             task=CorpusTask.TRAIN, copy=False):
    batch_iter = DynamicBatchtIter.from_opt(dataset_iter, vocabs, opt,
                                            task, copy)
    return batch_iter
