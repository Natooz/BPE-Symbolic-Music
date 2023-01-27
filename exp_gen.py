#!/usr/bin/python3 python

"""
Lists the Experiment baselines and training
"""

from classes import Experiment, Baseline, ModelConfig, TrainingConfig, TestingConfig
from constants import *


datasets = ['POP909-merged', 'GiantMIDI']
tokenizations = ['TSD', 'REMI']
bpe_factors = [0, 4, 10, 20, 50, 100]
gen_model_conf = ModelConfig()
cla_model_conf = ModelConfig(dim=CLA_DIM, nb_heads=CLA_NB_HEADS, d_ffwd=CLA_D_FFWD, nb_layers=CLA_NB_LAYERS,
                             embed_sizes=[CLA_DIM] * 5)
cla_model_conf_large = ModelConfig(dim=CLA_LARGE_DIM, nb_heads=CLA_LARGE_NB_HEADS, d_ffwd=CLA_LARGE_D_FFWD,
                                   nb_layers=CLA_LARGE_NB_LAYERS, embed_sizes=[CLA_LARGE_DIM] * 5)
gen_train_conf = TrainingConfig()
cla_pre_train_conf = TrainingConfig(training_steps=CLA_PRE_TRAINING_STEPS, learning_rate=CLA_PT_LEARNING_RATE,
                                    batch_size=CLA_BATCH_SIZE, iterator_kwargs={'early_stop_steps': CLA_EARLY_STOP})
cla_fine_tune_conf = TrainingConfig(training_steps=CLA_TRAINING_STEPS, learning_rate=CLA_FT_LEARNING_RATE,
                                    batch_size=CLA_BATCH_SIZE, iterator_kwargs={'early_stop_steps': CLA_EARLY_STOP})
gen_test_conf = TestingConfig()

experiments = []
for dataset in datasets:
    for tokenization in tokenizations:
        exp_name = f'{dataset}_{tokenization}'
        baselines = [Baseline('noBPE' if bf == 0 else f'bpe{bf}', exp_name, dataset, bi, tokenization, bf,
                              gen_model_conf) for bi, bf in enumerate(bpe_factors)]
        for token_combination in ['PVm', 'PVDm']:
            baselines += [Baseline(token_combination, exp_name, dataset, len(baselines),
                                   f'{tokenization}{token_combination}', 0, gen_model_conf)]
        if tokenization == 'REMI':  # adds CPWord and Octuple for comparison
            for tok, embed_sizes in [('CPWord', EMBED_SIZES_CP), ('OctupleMono', EMBED_SIZES_OCTUPLE)]:
                baselines += [Baseline(tok, exp_name, f'{dataset}-short' if tok == 'OctupleMono' else dataset,
                                       len(baselines), tok, 0, ModelConfig(embed_sizes=embed_sizes))]
        experiments.append(Experiment(baselines, dataset, tokenization, SEED, cla_model_conf, gen_train_conf,
                                      cla_pre_train_conf, cla_fine_tune_conf, gen_test_conf))


if __name__ == '__main__':
    from training import train_all_gen

    for exp_ in experiments:
        for baseline in exp_.baselines:
            baseline.load_tokenizer()

    for exp_ in experiments:
        train_all_gen(exp_)
