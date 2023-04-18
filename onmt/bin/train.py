#!/usr/bin/env python
"""Train models with dynamic data."""
import torch
from functools import partial
from onmt.utils.distributed import ErrorHandler, consumer
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts, translate_opts
from onmt.train_single import main as single_main


# Set sharing strategy manually instead of default based on the OS.
# torch.multiprocessing.set_sharing_strategy('file_system')


# def train(train_opt, translate_opt):
def train(train_opt):


    init_logger(train_opt.log_file)

    ArgumentParser.validate_train_opts(train_opt)
    ArgumentParser.update_model_opts(train_opt)
    ArgumentParser.validate_model_opts(train_opt)

    # ArgumentParser.validate_translate_opts(translate_opt)
    # ArgumentParser._get_all_transform_translate(translate_opt)
    # ArgumentParser._validate_transforms_opts(translate_opt)
    # ArgumentParser.validate_translate_opts_dynamic(translate_opt)

    set_random_seed(train_opt.seed, False)

    train_process = partial(
        single_main)

    nb_gpu = len(train_opt.gpu_ranks)

    if train_opt.world_size > 1:
        mp = torch.multiprocessing.get_context('spawn')
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            procs.append(mp.Process(target=consumer, args=(
                train_process, train_opt, #, translate_opt,
                device_id, error_queue),
                daemon=False))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        train_process(train_opt, device_id=0)
        # train_process(train_opt, translate_opt, device_id=0)
    else:   # case only CPU
        # train_process(train_opt, translate_opt, device_id=-1)
        train_process(train_opt, device_id=0)


def _get_train_parsers():
    import sys
    prec_argv = sys.argv
    sys.argv = sys.argv[:1]
    train_parser = ArgumentParser(description='train options in train.py')
    train_opts(train_parser)
    sys.argv = prec_argv
    translate_parser = ArgumentParser(description='translate options in train.py')
    translate_opts(translate_parser, is_train=False)
    # translate_opts(train_parser,  is_train=True)
    return train_parser, translate_parser
    #return train_parser


# def _get_translate_parser():
#     parser = ArgumentParser(
#         description='Parse translate options for dynamic scoring')
#     translate_opts(parser, dynamic=True)
#     return parser


def main():
    train_parser, translate_parser = _get_train_parsers()
    train_opt, _ = train_parser.parse_known_args()
    translate_opt, _ =  translate_parser.parse_known_args()
    print("#### train_opt", train_opt)
    train(train_opt)
    print("#### tanslate_opt", translate_opt)
    train(train_opt)
    import sys
    sys.exit()
    train(train_opt, translate_opt)


if __name__ == "__main__":
    main()
