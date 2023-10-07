---
title                 : ""
disableTitleSeparator : true
toc : True
---

## Introduction

<!-- Feel free to check the [posts](/posts) to see some of the theme features. -->

Welcome to the demo website of the paper "Byte Pair Encoding for Symbolic Music".

### BPE

[Byte Pair Encoding (BPE)](https://www.derczynski.com/papers/archive/BPE_Gage.pdf) is a data compression technique. It converts the most recurrent successive bytes\footnote{For symbolic music in our case, the basic "tokens" (e.g. describing note and time attributes) can be seen as the base characters (bytes) that will be merged.} in a corpus into newly created ones. For instance, in the character sequence ``aabaabaacaa``, the sub-sequence ``aa`` occurs three times and is the most recurrent one. Learning and applying BPE on this sequence would replace ``aa`` with a new symbol, e.g., ``d``, resulting in a compressed sequence ``dbdbdcd``. The latter can be reduced again by replacing the ``db`` subsequence, giving ``eedcd``. In practice BPE is learned on a corpus until the vocabulary reaches a target size.

BPE is nowadays largely used in the NLP field as it allows to [encode rare words and segmenting unknown or composed words as sequences of sub-word units](https://aclanthology.org/P16-1162/). Other token aggregation, or vocabulary building techniques exist. The two other most commonly used are [Unigram](https://aclanthology.org/P18-1007/) or [WordPiece](https://arxiv.org/abs/1609.08144), which operations share similarities with BPE.

For natural language, bytes are the distinct characters composing the text. Its application to symbolic music has however not yet been studied.
For symbolic music, the "bytes" are considered as the distinct note and time attributes in this paper. In this context, BPE can allow to represent a note, or even a succession of notes, that is very recurrent in the dataset, as a single token. For instance, a note that would be tokenized as the succession of tokens ``Pitch_D3``, ``Velocity_60``, ``Duration_2.0`` could be replaced by a single new one.
Rare note (and attributes) can still be tokenized as non-BPE tokens. The same logic applies to time tokens, that can also be associated to note tokens.

## Main results

We recap the main results from the paper.

### Model weights

Models weights are not currently shared during the reviewing process, as we cannot share them in a way that would not break anonymity. We will made them publicly available upon acceptance, along with the training files. We also plan to publish the weights of the best performing models on the Hugging Face hub.

### Generation

BPE allows to improve the quality of the generation, while increasing the inference speed.
Generated examples are shown in the last sections.

![Generation quality](generation.png "Metrics of generated results. TSE results are all scaled at e−3 for better readability. Hum stand for human, - for non-concerned (i.e. 0).")

![Inference speed](inference_speed.png "Inference speeds on a V100 GPU and proportion of vocabulary sampled during generation. For tok/sec, the results account for basic tokens of note attributes and time. Tok/sec for Octuple is not showed as the equivalent number of base tokens is not clearly calculable.")

### Classification

![Classification accuracy](classification.png "Average accuracy of classification models.")

### Embedding space

One of the main limitation of the previous work on symbolic music modeling is the suboptimal usage of the embedding space of LMs. Most of them use models with embeddings represented from 512 to 1024 dimensions, for vocabularies of less than 500 tokens. One can easily see here the suboptimal situation of using as much dimensions to represent so few points. The same models, when used with natural language, use to learn up to 50k embeddings on a range of 512 to 1024 dimensions.

BPE allows to make a better use of the embedding space, using more space while having better distributed embeddings (isotropy).

![Isotropy](isotropy.png "Isocore, and intrinsic dimension (ID) estimations. Gen. corresponds to the causal generative models, Pt. to the pretrained bidirectional models.")

We display below the 2D and 3D UMAP projections of the embedding matrices of all the models.
For each window, the seven representations correspond to: no BPE, BPE 1k, BPE 5k, BPE 10k, BPE 20k, PVm and PVDm.

#### Generator (GPT2) + TSD

{{< embed-pdf url="umap/umap_2d_gen_Maestro_TSD.pdf" >}}
{{< embed-pdf url="umap/umap_3d_gen_Maestro_TSD.pdf" >}}

#### Generator (GPT2) + REMI

{{< embed-pdf url="umap/umap_2d_gen_Maestro_REMI.pdf" >}}
{{< embed-pdf url="umap/umap_3d_gen_Maestro_REMI.pdf" >}}

#### Pretrained (BERT) + TSD

{{< embed-pdf url="umap/umap_2d_cla_genre_MMD_TSD.pdf" >}}
{{< embed-pdf url="umap/umap_3d_cla_genre_MMD_TSD.pdf" >}}

#### Pretrained (BERT) + REMI

{{< embed-pdf url="umap/umap_2d_cla_genre_MMD_REMI.pdf" >}}
{{< embed-pdf url="umap/umap_3d_cla_genre_MMD_REMI.pdf" >}}

#### Singular values

The pages correspond to *Pretrained + TSD*, *Pretrained + REMI*, *Generator + TSD* and *Generator + REMI*.

{{< embed-pdf url="umap/umap_2d_gen_Maestro_TSD.pdf" >}}

## Generated examples

We made listenable generated samples below, that have been cherry-picked so that all tracks show enough diversity.
You can
{{< a href="bpe_for_symbolic_music_human_eval_samples.zip" download="bpe_for_symbolic_music_human_eval_samples" >}}
download all samples
{{< /a >}}
used for the human evaluations as MIDI files. On these files, the tracks are not shuffled and **corresponds by order to: no BPE, BPE 1k, BPE 5k, BPE 10k, BPE 20k, PVm, PVDm, CP Word and Octuple**. (the two latter are only present for REMI files) **The same goes for the audio samples below.**

### TSD

##### File 27

{{< audio src="audio_tsd/21_1.mp3">}}
{{< audio src="audio_tsd/21_2.mp3">}}
{{< audio src="audio_tsd/21_3.mp3">}}
{{< audio src="audio_tsd/21_4.mp3">}}
{{< audio src="audio_tsd/21_5.mp3">}}
{{< audio src="audio_tsd/21_6.mp3">}}
{{< audio src="audio_tsd/21_7.mp3">}}

##### File 41

{{< audio src="audio_tsd/41_1.mp3">}}
{{< audio src="audio_tsd/41_2.mp3">}}
{{< audio src="audio_tsd/41_3.mp3">}}
{{< audio src="audio_tsd/41_4.mp3">}}
{{< audio src="audio_tsd/41_5.mp3">}}
{{< audio src="audio_tsd/41_6.mp3">}}
{{< audio src="audio_tsd/41_7.mp3">}}

##### File 52

{{< audio src="audio_tsd/52_1.mp3">}}
{{< audio src="audio_tsd/52_2.mp3">}}
{{< audio src="audio_tsd/52_3.mp3">}}
{{< audio src="audio_tsd/52_4.mp3">}}
{{< audio src="audio_tsd/52_5.mp3">}}
{{< audio src="audio_tsd/52_6.mp3">}}
{{< audio src="audio_tsd/52_7.mp3">}}

##### File 55

{{< audio src="audio_tsd/55_1.mp3">}}
{{< audio src="audio_tsd/55_2.mp3">}}
{{< audio src="audio_tsd/55_3.mp3">}}
{{< audio src="audio_tsd/55_4.mp3">}}
{{< audio src="audio_tsd/55_5.mp3">}}
{{< audio src="audio_tsd/55_6.mp3">}}
{{< audio src="audio_tsd/55_7.mp3">}}

### REMI

Recall: **tracks corresponds by order to: no BPE, BPE 1k, BPE 5k, BPE 10k, BPE 20k, PVm, PVDm, CP Word and Octuple**

##### File 12

{{< audio src="audio_remi/12_1.mp3">}}
{{< audio src="audio_remi/12_2.mp3">}}
{{< audio src="audio_remi/12_3.mp3">}}
{{< audio src="audio_remi/12_4.mp3">}}
{{< audio src="audio_remi/12_5.mp3">}}
{{< audio src="audio_remi/12_6.mp3">}}
{{< audio src="audio_remi/12_7.mp3">}}
{{< audio src="audio_remi/12_8.mp3">}}
{{< audio src="audio_remi/12_9.mp3">}}

##### File 33

{{< audio src="audio_remi/33_1.mp3">}}
{{< audio src="audio_remi/33_2.mp3">}}
{{< audio src="audio_remi/33_3.mp3">}}
{{< audio src="audio_remi/33_4.mp3">}}
{{< audio src="audio_remi/33_5.mp3">}}
{{< audio src="audio_remi/33_6.mp3">}}
{{< audio src="audio_remi/33_7.mp3">}}
{{< audio src="audio_remi/33_8.mp3">}}
{{< audio src="audio_remi/33_9.mp3">}}

##### File 47

{{< audio src="audio_remi/47_1.mp3">}}
{{< audio src="audio_remi/47_2.mp3">}}
{{< audio src="audio_remi/47_3.mp3">}}
{{< audio src="audio_remi/47_4.mp3">}}
{{< audio src="audio_remi/47_5.mp3">}}
{{< audio src="audio_remi/47_6.mp3">}}
{{< audio src="audio_remi/47_7.mp3">}}
{{< audio src="audio_remi/47_8.mp3">}}
{{< audio src="audio_remi/47_9.mp3">}}

##### File 54

{{< audio src="audio_remi/54_1.mp3">}}
{{< audio src="audio_remi/54_2.mp3">}}
{{< audio src="audio_remi/54_3.mp3">}}
{{< audio src="audio_remi/54_4.mp3">}}
{{< audio src="audio_remi/54_5.mp3">}}
{{< audio src="audio_remi/54_6.mp3">}}
{{< audio src="audio_remi/54_7.mp3">}}
{{< audio src="audio_remi/54_8.mp3">}}
{{< audio src="audio_remi/54_9.mp3">}}
