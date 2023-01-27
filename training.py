"""
Training functions
"""

from typing import Union, List
from pathlib import Path
from time import time
from copy import deepcopy
import logging

from torch import Tensor, FloatTensor, load, save
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtoolkit.train import train, log_model_parameters, log_cuda_info, select_device
from torchtoolkit.utils import seed_everything
from torchtoolkit.data import create_subsets
from torchtoolkit.metrics import calculate_accuracy
from tqdm import tqdm

from dataset import MIDIDatasetGen
from model import ClassifierTransformer
from classes import Experiment, TrainingConfig
from constants import TIME_DIVISION


def save_tokens(prompt: Tensor, continuation: Tensor, tokenizer, out_dir: Path,
                file_name: Union[int, str]):
    r"""Saves generated tokens, as json and MIDi files.
    :param prompt: original sample (prompt) used for the generation.
    :param continuation: generated sequence, with original_sample at its beginning.
    :param tokenizer: tokenizer object.
    :param out_dir: output directory.
    :param file_name: file name, with no extension (.json and .mid will be added).
    """
    generated = continuation[len(prompt):]
    tokens = [generated, prompt, continuation]  # list compr. as seqs of dif. lengths
    tokens = [seq.tolist() for seq in tokens]
    midi = tokenizer.tokens_to_midi(deepcopy(tokens), time_division=TIME_DIVISION)  # copy as inplace decompose bpe op
    midi.instruments[0].name = f'Continuation of original sample ({len(generated)} tokens)'
    midi.instruments[1].name = f'Original sample ({len(prompt)} tokens)'
    midi.instruments[2].name = f'Original sample and continuation'
    midi.dump(out_dir / f'{file_name}.mid')
    tokenizer.save_tokens(tokens, out_dir / f'{file_name}.json')


def acc_generator(res: Union[Tensor, List[Tensor]], exp: Tensor) -> float:
    r"""Computes the accuracy of a result, in top-k mode.
    :param res: result tensor of shape (T,N,C) or a list of Z (T,N,C)
    :param exp: expected result of shape (N,T) or (N,T,Z)
    :return: the mean accuracy (token / time step wise)
    """
    if isinstance(res, list):
        return sum([acc_generator(res[i], exp[..., i]) for i in range(len(res))]) / len(res)
    return calculate_accuracy(res, exp, mode='top_k', top_kp=5)


def acc_classifier(res: Union[FloatTensor, List[FloatTensor]], exp: FloatTensor) -> float:
    r"""Accuracy method for the classifier
    :param res: result tensor of shape (T,N,C) or a list of Z (T,N,C)
    :param exp: expected result of shape (N,T) or (N,T,Z)
    :return: the mean accuracy (token / time step wise)
    """
    if isinstance(res, list):
        return sum([acc_classifier(res[i], exp[..., i]) for i in range(len(res))]) / len(res)
    return calculate_accuracy(res, exp, mode='softmax')


def train_all_gen(exp: Experiment):
    r"""Trains the generative models (baselines) for a given experiment

    :param exp: experiment associated
    :return: discriminator model trained
    """
    for baseline in exp.baselines:
        if (baseline.run_path / 'checkpoint.pt.tar').exists() and baseline.gen_data_path.exists():
            continue  # already trained and results already generated
        seed_everything(baseline.seed)
        baseline.run_path.mkdir(parents=True, exist_ok=True)
        device = select_device(exp.gen_train_conf.use_cuda)
        logger = logging.getLogger(exp.name)
        (fh := logging.FileHandler(baseline.run_path / 'training.log')).setLevel(logging.DEBUG)
        (sh := logging.StreamHandler()).setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.DEBUG)

        # Loads data
        baseline.load_tokenizer()
        dataset = MIDIDatasetGen(baseline.data_path, exp.gen_train_conf.min_seq_len, exp.gen_train_conf.max_seq_len,
                                 baseline.pad_token, baseline.sos_token)
        subset_train, subset_valid = create_subsets(dataset, [exp.gen_train_conf.valid_split])
        dataloader_train = DataLoader(subset_train, exp.gen_train_conf.batch_size, collate_fn=dataset.collate_fn)
        dataloader_valid = DataLoader(subset_valid, exp.gen_train_conf.batch_size, collate_fn=dataset.collate_fn)

        # Creates the model
        model = exp.create_gen(baseline).to(device)

        # Training if not already done
        if not (baseline.run_path / 'checkpoint.pt.tar').exists():
            log_model_parameters(model, logger)
            if device.type == 'cuda':
                log_cuda_info(logger)
            logger.info(f'Training: {len(subset_train)} samples - {len(dataloader_train)} batches')
            logger.info(f'Validation: {len(subset_valid)} samples - {len(dataloader_valid)} batches')
            logger.info(f'1 epoch is equivalent to {len(dataloader_train) / exp.gen_train_conf.grad_acc_steps} steps')
            logger.info(f'Training ({exp.gen_train_conf.training_steps} steps) is equivalent to '
                        f'{exp.gen_train_conf.training_steps / len(dataloader_train):.2f} epochs')
            tt0 = time()

            tensorboard = SummaryWriter(str(baseline.run_path))
            criterion = CrossEntropyLoss(label_smoothing=exp.gen_train_conf.label_smoothing,
                                         ignore_index=baseline.pad_token)
            optimizer = Adam(params=model.parameters(), lr=exp.gen_train_conf.learning_rate,
                             weight_decay=exp.gen_train_conf.weight_decay)
            lr_scheduler = OneCycleLR(optimizer, exp.gen_train_conf.learning_rate, exp.gen_train_conf.training_steps,
                                      pct_start=exp.gen_train_conf.warmup_ratio)
            train(model, criterion, optimizer, dataloader_train, dataloader_valid, exp.gen_train_conf.training_steps,
                  exp.gen_train_conf.valid_intvl, exp.gen_train_conf.nb_valid_steps, tensorboard,
                  f'TRAINING GEN {baseline.run_path.name}', logger, exp.gen_train_conf.log_intvl, acc_generator,
                  iterator_kwargs=exp.gen_train_conf.iterator_kwargs, lr_scheduler=lr_scheduler, device=device,
                  use_amp=exp.gen_train_conf.use_amp, gradient_clip_norm=exp.gen_train_conf.gradient_clip_norm,
                  saving_dir=baseline.run_path)
            logger.info(f'TRAINING FINISHED - {time() - tt0:.2f}')
            save({'model_state_dict': model.state_dict()}, baseline.run_path / 'last_checkpoint.pt.tar')
            tensorboard.close()
        logger.handlers = []

        # Generation
        seed_everything(baseline.seed)
        model.load_state_dict(load(baseline.run_path / 'checkpoint.pt.tar', map_location=device)['model_state_dict'])
        model.eval()
        baseline.gen_data_path.mkdir(parents=True, exist_ok=True)
        dataloader_test = DataLoader(subset_valid, exp.gen_test_conf.batch_size,
                                     collate_fn=dataset.collate_fn_infer)
        count = 0
        for batch in tqdm(dataloader_test, desc='Testing model'):  # (N,T) or (N,T,Z)
            # Attention mask (handling padding), sampling are handled in generate method
            if isinstance(baseline.tokenizer.vocab, list):  # (N,T,Z)
                res = model.generate(batch.to(device), exp.gen_test_conf.nb_inferences_test,
                                     exp.gen_train_conf.max_seq_len)
            else:  # using hugging face default method, with beam search and nucleus sampling
                res = model.generate(batch.to(device), do_sample=True, num_beams=exp.gen_test_conf.num_beams,  # (N,T)
                                     top_p=exp.gen_test_conf.top_p, max_new_tokens=exp.gen_test_conf.nb_inferences_test)
            for prompt, continuation in zip(batch, res):
                save_tokens(prompt, continuation, baseline.tokenizer, baseline.gen_data_path, count)
                count += 1


def train_classifier(conf: TrainingConfig, model: ClassifierTransformer, dataloader_train: DataLoader,
                     dataloader_valid: DataLoader, seed: int, saving_dir: Path, pbar_desc: str = None,
                     pad_token: int = None, binary: bool = False):
    r"""Trains a classifier model.

    :param conf: training configuration
    :param model: the model
    :param dataloader_train:
    :param dataloader_valid:
    :param seed: seed for reproducibility, random operations: dropout and sampling
    :param saving_dir: path to save the models parameters
    :param pbar_desc: progress bar description
    :param pad_token: padding token
    :param binary: if binary classifier, then BCELoss will be used (default: False)
    :return: discriminator model trained
    """
    if (saving_dir / 'checkpoint.pt.tar').is_file():
        return
    seed_everything(seed)
    saving_dir.mkdir(exist_ok=True, parents=True)
    device = select_device(conf.use_cuda)
    pbar_desc = f'PRE-TRAINING DISC {saving_dir.name}' if pbar_desc is None else pbar_desc
    logger = logging.getLogger()
    (fh := logging.FileHandler(saving_dir / 'training.log')).setLevel(logging.DEBUG)
    (sh := logging.StreamHandler()).setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    acc_fn = acc_classifier

    # Log info
    log_model_parameters(model, logger)
    if device.type == 'cuda':
        log_cuda_info(logger)
    logger.info(f'Training: {len(dataloader_train.dataset)} samples - {len(dataloader_train)} batches')
    logger.info(f'Validation: {len(dataloader_valid.dataset)} samples - {len(dataloader_valid)} batches')
    logger.info(f'1 epoch is equivalent to {len(dataloader_train) / conf.grad_acc_steps} steps')
    logger.info(f'Training ({conf.training_steps} steps) is equivalent to '
                f'{conf.training_steps / len(dataloader_train):.2f} epochs')
    tt0 = time()

    # Training
    tensorboard = SummaryWriter(str(saving_dir))
    criterion = BCEWithLogitsLoss() if binary else CrossEntropyLoss(ignore_index=pad_token)
    optimizer = Adam(params=model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    lr_scheduler = OneCycleLR(optimizer, conf.learning_rate, conf.training_steps, pct_start=conf.warmup_ratio)
    train(model, criterion, optimizer, dataloader_train, dataloader_valid, conf.training_steps,
          conf.valid_intvl, conf.nb_valid_steps, tensorboard, pbar_desc, logger, conf.log_intvl, acc_fn,
          lr_scheduler=lr_scheduler, device=device, use_amp=conf.use_amp, gradient_clip_norm=conf.gradient_clip_norm,
          saving_dir=saving_dir, iterator_kwargs=conf.iterator_kwargs)
    logger.info(f'PRE-TRAINING FINISHED - {time() - tt0:.2f}')
    save({'model_state_dict': model.state_dict()}, saving_dir / 'last_checkpoint.pt.tar')
    tensorboard.close()
    logger.handlers = []
