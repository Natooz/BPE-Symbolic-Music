# Byte Pair Encoding for Symbolic Music (EMNLP 2023)

[[Paper](https://arxiv.org/abs/2301.11975)]
[[Companion website](https://Natooz.github.io/BPE-Symbolic-Music/)]

Byte Pair Encoding (BPE) is a compression technique that allows to reduce the sequence length of a corpus by iteratively replacing the most recurrent byte successions by newly created symbols. It is widely used in NLP, as it allows to automatically create vocabularies made of words or parts of words.

In this paper, we show that it can address two main concerns about how symbolic music was previously tokenized:

1. The fairly long sequence length resulting by using one token per note attribute (e.g. pitch, duration) and time events. Long sequences is problematic as the time and space complexity of Transformer models grows quadratically with the input sequence.
2. The poor usage of the model's embedding space. Language models first project tokens into a learned embedding space, in which the embeddings (continuous representations of the tokens) are learnt to represent their semantic information. This is an essential feature of such models, as it allows them to capture the meaning of the tokens and data. In symbolic music, the tokens usually only represent note attribute values or time values, which do not carry much information other than their absolute value. And vocabularies range often between 200 and 500 tokens, which are then represented on 512 to 1024 dimensions. In such conditions, the embedding space is misused and the potential of the model is poorly exploited.

When applied on symbolic music, BPE will allow to drastically reduce the sequence length, while creating new tokens that can represent whole notes, and sequences of notes. The model's efficiency is then greatly improved, while bringing more information per tokens. It greatly improves the quality of generation, while improving up to three times the inference speed.

BPE is fully implemented within [MidiTok](https://github.com/Natooz/MidiTok), allowing you to easily benefit from this method on top of most of the existing tokenizations.

We invite you to read the paper, and check our [companion website](https://Natooz.github.io/bpe-symbolic-music/) to listen generated results!

## Steps to reproduce

1. `pip install -r requirements` to install requirements
2. Download the [Maestro](https://magenta.tensorflow.org/datasets/maestro) and [MMD](https://zenodo.org/record/5142664#.YQN3c5NKgWo) datasets and put them in `data/`
3. `python scripts/preprocess_maestro.py` and `python scripts/preprocess_for_octuple.py`
4. `python scripts/tokenize_datasets.py` to tokenize data and learn BPE
5. `python exp_generation.py` to train generative models and generate results
6. `python exp_pretrain.py` to pretrain classification models
7. `python exp_cla.py` to train classification models and test them

[Scripts](./scripts) can be run to get reproduce the analysis.

## Citation

(ACL url/doi/pages will be added once the proceeding will be published)
```bibtex
@inproceedings{bpe-symbolic-music,
    title = "Byte Pair Encoding for Symbolic Music",
    author = "Fradet, Nathan  and
      Gutowski, Nicolas  and
      Chhel, Fabien  and
      Briot, Jean-Pierre",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2301.11975",
}
```