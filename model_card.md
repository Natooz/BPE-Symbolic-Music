---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{}
---

# Model card

<!-- Provide a quick summary of what the model is/does. -->

This is a generative model from the paper "*Byte Pair Encoding for Symbolic Music*" (EMNLP 2023). The model has been trained with Byte Pair Encoding (BPE) on the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro) to generate classical piano music with the REMI tokenizer.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

It has a vocabulary of 20k tokens learned with [Byte Pair Encoding (BPE)](https://arxiv.org/abs/2301.11975) using [MidiTok](https://github.com/Natooz/MidiTok).

- **Developed and shared by:** [Nathan Fradet](https://twitter.com/NathanFradet)
- **Affiliations**: [Sorbonne University (LIP6 lab)](https://www.sorbonne-universite.fr/en) and [Aubay](https://aubay.com/en/)
- **Model type:** causal autoregressive Transformer
- **Backbone model:** [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- **Music genres:** Classical piano 🎹
- **License:** Apache 2.0

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/Natooz/BPE-Symbolic-Music
- **Paper:** https://arxiv.org/abs/2301.11975

## Uses

The model is designed for autoregressive music generation. It generates the continuation of a music prompt.

## How to Get Started with the Model

Use the code below to get started with the model.
You will need the `miditok`, `transformers` and `torch` packages to make it run, that can be installed with pip.

You will also need to manually download the `tokenizer.conf` file from the [repo files](https://huggingface.co/Natooz/Maestro-REMI-bpe20k/tree/main).

```Python
import torch
from transformers import AutoModelForCausalLM
from miditok import REMI
from miditoolkit import MidiFile

torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained("Natooz/Maestro-REMI-bpe20k", trust_remote_code=True, torch_dtype="auto")
tokenizer = REMI(params="tokenizer.conf")
input_midi = MidiFile("path/to/file.mid")
input_tokens = tokenizer(input_midi)

generated_token_ids = model.generate(input_tokens.ids, max_length=200)
generated_midi = tokenizer(generated_token_ids)
generated_midi.dump("path/to/continued.mid")
```

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model has been trained on the [Maestro](https://magenta.tensorflow.org/datasets/maestro) dataset. The dataset contains about 200 hours of classical piano music. The tokenizer is trained with Byte Pair Encoding (BPE) to build a vocabulary of 20k tokens.

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

- **Training regime:** fp16 mixed precision on V100 PCIE 32GB GPUs
- **Compute Region:** France

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 64
- eval_batch_size: 96
- seed: 444
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine_with_restarts
- lr_scheduler_warmup_ratio: 0.3
- training_steps: 100000

### Environmental impact

We cannot estimate reliably the amount of CO2eq emitted, as we lack data on the exact power source used during training. However, we can highlight that the cluster used is mostly powered by nuclear energy, which is a low carbon energy source ensuring a reduced direct environmental impact.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

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

