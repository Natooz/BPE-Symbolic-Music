"""Model

"""

from typing import List, Tuple, Union, Callable, Optional
from functools import partial

from torch import LongTensor, FloatTensor, Tensor, cat, stack, no_grad, arange
from torch.nn import Module, ModuleList, Linear, Embedding, Dropout, LayerNorm
from torch.nn.modules import BCEWithLogitsLoss
from torchtoolkit.sampling import nucleus
from transformers import GPT2LMHeadModel, GPT2Config, BertPreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.gpt2.modeling_gpt2 import CausalLMOutputWithCrossAttentions

from constants import TOP_P


class GenTransformer(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer.wpe.padding_idx = config.pad_token_id  # updates the padding idx
        self.transformer.wte.padding_idx = config.pad_token_id

    def forward_train(self, x: LongTensor, target: LongTensor, criterion: Module):
        y = self.forward(x).logits  # (N,T,C)
        loss = criterion(y.transpose(2, 1), target)
        return y, loss, None  # no need for sampled

    @no_grad()
    def generate_(self, x: LongTensor, nb_steps: int, max_seq_len: int, sampling_func: Callable = None) -> LongTensor:
        r"""UNUSED
        Generate (extend) from the generator
        :param x: input tensor to extend, shape (N,T) or (T)
        :param nb_steps: number of steps (inferences) to run
        :param max_seq_len: maximum sequence length during inference
        :param sampling_func: sampling function (default: top_k with k=15)
        :return: the generated tensor
        """
        assert max_seq_len <= (nb_pos := self.transformer.wpe.weight.shape[0]), \
            'The maximum sequence length must be <= to the nb of positions the model can handle'
        sampling_func = partial(nucleus, p=0.9) if sampling_func is None else sampling_func
        y = x.clone()
        if y.dim() == 1:
            y = y.unsqueeze(0)  # (T) --> (N,T) with N=1
        past_key_val, pos_ids = None, None  # (NLY,2,N,NH,T,DH) & (T'), T' for the non-past-kv part (often 1)
        offset = 0
        tokens = y.clone()  # (N,T)
        for _ in range(nb_steps):
            # Adds the prediction to the target sequence, updates past key values and y sequence
            logits = self.forward(tokens, past_key_val, position_ids=pos_ids)
            logits, past_key_val = logits.logits, logits.past_key_values  # (N,T,C)
            tokens = sampling_func(logits[:, -1]).unsqueeze(1).to(x.device)  # (N,1)
            y = cat([y, tokens], dim=1)  # (N,T+1)

            # Reset past_kv and offset to not exceed pos enc
            if past_key_val[0][0].shape[-2] + offset >= nb_pos:
                past_key_val, pos_ids, offset = None, None, 0
                tokens = y[..., -x.shape[-1]:].clone()  # starting back with len of x for prompt

            # Reduces past_kv if the max len is reached
            if past_key_val is not None and past_key_val[0][0].shape[-2] >= max_seq_len:
                offset += 1
                past_key_val = convert_past_key_values_to_tensor(past_key_val)[..., -max_seq_len:, :]
                pos_ids = LongTensor([past_key_val.shape[-2] + offset]).to(x.device)

        return y[0] if x.dim() == 1 else y  # (T) or (N,T)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[Embedding, Tuple[Embedding]]:
        pass


class GenTransformerPooling(GPT2LMHeadModel):

    def __init__(self, config: GPT2Config, num_classes: List[int], embed_sizes: List[int]):
        super().__init__(config)
        self.transformer.wte = MultiEmbeddings(num_classes, embed_sizes, config.n_embd, padding_idx=config.pad_token_id)
        self.lm_head = MultiOutput(num_classes=num_classes, d_model=config.n_embd)

        self.register_buffer('padding_token', LongTensor([config.pad_token_id]))

    def forward_train(self, x: LongTensor, target: LongTensor, criterion: Module):
        y = self.forward(x).logits  # list of (N,T,C)
        loss = sum([criterion(yi.transpose(2, 1), target[..., i]) for i, yi in enumerate(y)])
        return y, loss, None  # no need for sampled

    def forward(self, input_ids: LongTensor = None, past_key_values: Tuple[Tuple[Tensor]] = None,
                attention_mask: FloatTensor = None, token_type_ids: LongTensor = None, position_ids: LongTensor = None,
                head_mask: FloatTensor = None, inputs_embeds: FloatTensor = None, encoder_hidden_states: Tensor = None,
                encoder_attention_mask: FloatTensor = None, labels: LongTensor = None, use_cache: bool = None,
                output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None, **kwargs) \
            -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""We need to override the function as it would get the wrong batch size with a 3D tensor
        """
        inputs_embeds = self.transformer.wte(input_ids)  # multi input module
        input_ids = None  # we instead directly pass the embeddings to the transformer / GPT2 model
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @no_grad()
    def generate(self, x: LongTensor, nb_steps: int, max_seq_len: int, sampling_func: Callable = None) -> LongTensor:
        r"""Generate (extend) from the generator
        :param x: input tensor to extend, shape (N,T,Z) or (T,Z), padded to the left
        :param nb_steps: number of steps (inferences) to run
        :param max_seq_len: maximum sequence length during inference
        :param sampling_func: sampling function (default: top_k with k=15)
        :return: the generated tensor
        """
        assert max_seq_len <= (nb_pos := self.transformer.wpe.weight.shape[0]), \
            'The maximum sequence length must be <= to the nb of positions the model can handle'
        sampling_func = partial(nucleus, p=TOP_P) if sampling_func is None else sampling_func
        y = x.clone()
        if y.dim() == 2:
            y = y.unsqueeze(0)  # (T,Z) --> (N,T,Z) with N=1
        past_key_val, pos_ids = None, None  # (NLY,2,N,NH,T,DH) & (T'), T' for the non-past-kv part (often 1)
        offset = 0
        tokens = y.clone()  # (N,T,Z)
        for _ in range(nb_steps):
            # Adds the prediction to the target sequence, updates past key values and y sequence
            logits = self.forward(tokens, past_key_val, position_ids=pos_ids)
            logits, past_key_val = logits.logits, logits.past_key_values  # [Z: (N,T,C)]
            tokens = [sampling_func(dist[:, -1]).unsqueeze(1).to(x.device) for dist in logits]  # [Z: (N,1)]
            tokens = stack(tokens).permute(1, 2, 0)  # (Z,N,1) --> (N,1,Z)
            y = cat([y, tokens], dim=1)  # (N,T+1,Z)

            # Reset past_kv and offset to not exceed pos enc
            if past_key_val[0][0].shape[-2] + offset >= nb_pos:
                past_key_val, pos_ids, offset = None, None, 0
                tokens = y[:, -x.shape[1]:].clone()  # starting back with len of x for prompt

            # Reduces past_kv if the max len is reached
            if past_key_val is not None and past_key_val[0][0].shape[-2] >= max_seq_len:
                offset += 1
                past_key_val = convert_past_key_values_to_tensor(past_key_val)[..., -max_seq_len:, :]
                pos_ids = LongTensor([past_key_val.shape[-2] + offset]).to(x.device)

        return y[0] if x.dim() == 1 else y  # (T,Z) or (N,T,Z)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[Embedding, Tuple[Embedding]]:
        pass


class ClassifierTransformer(BertPreTrainedModel):

    def __init__(self, config: BertConfig, pre_train: bool = False):
        super().__init__(config)
        self.pre_train = pre_train

        self.bert = BertModel(config)

        # For pre-training
        self.cls = BertOnlyMLMHead(config)

        # Classifier head
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids: Optional[LongTensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> FloatTensor:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (N,T,E) & (N,E)

        if self.pre_train:
            vocab_logits = self.cls(sequence_output)
            return vocab_logits  # (N,T,V)

        else:  # we dont use the pooled_output from hf module
            pooled_output = sequence_output[:, 0]
            pooled_output = self.dropout(pooled_output)
            labels_logits = self.classifier(pooled_output)
            return labels_logits  # (N,C)

    def forward_train(self, x: LongTensor, target: LongTensor, criterion: Module):
        attention_mask = (x != self.config.pad_token_id).float()  # 1 for non-masked positions
        y = self.forward(x, attention_mask=attention_mask)  # (N,T,C)

        if self.pre_train:  # with cross entropy, target is (N,T)
            loss = criterion(y.transpose(2, 1), target)
        else:  # classification, target is [N]
            if isinstance(criterion, BCEWithLogitsLoss):  # y is [N,C], is the pooled position
                y = y.squeeze(-1)  # (N) & (N) for BCELoss
            loss = criterion(y, target)
        return y, loss, None  # no need for sampled

    @no_grad()
    def infer(self, x: LongTensor) -> Tensor:
        """Infer from the classifier

        :param x: input sequences, of shape (N,T). Have to be padded to the left.
        :return: results for real samples, and fake samples
        """
        attention_mask = (x != self.config.pad_token_id).float()  # 1 for non-masked positions
        return self.forward(x, attention_mask=attention_mask)  # (N,T,C)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[Embedding, Tuple[Embedding]]:
        pass

    def _reorder_cache(self, past, beam_idx):
        pass


class ClassifierTransformerPooling(BertPreTrainedModel):

    def __init__(self, config: BertConfig, num_classes: List[int], embed_sizes: List[int], pre_train: bool = False):
        super().__init__(config)
        self.pre_train = pre_train

        self.bert = BertModel(config)
        self.bert.embeddings = BertMultiEmbeddings(config, num_classes, embed_sizes)

        # For pre-training
        self.cls = MultiOutput(num_classes, config.hidden_size)

        # Classifier head
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(config.hidden_size, config.num_labels)

        # Weight tying for multi_embed
        for i in range(len(num_classes)):
            self._tie_or_clone_weights(self.cls.output_layers[i],
                                       self.bert.embeddings.word_embeddings.embedding_layers[i])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids: Optional[LongTensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> FloatTensor:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (N,T,E) & (N,E)

        if self.pre_train:
            vocab_logits = self.cls(sequence_output)
            return vocab_logits  # [Z (N,T,V)]

        else:
            pooled_output = sequence_output[:, 0]
            pooled_output = self.dropout(pooled_output)
            labels_logits = self.classifier(pooled_output)
            return labels_logits  # (N,C)

    def forward_train(self, x: LongTensor, target: LongTensor, criterion: Module):
        attention_mask = (x == self.config.pad_token_id)[..., 0]  # reduce of 1 dim to have good shape
        y = self.forward(x, attention_mask=attention_mask)  # [Z (N,T,C)] or (N,T,E)

        if self.pre_train:
            loss = sum([criterion(yi.transpose(2, 1), target[..., i]) for i, yi in enumerate(y)])
        else:
            if isinstance(criterion, BCEWithLogitsLoss):  # y is [N,C], is the pooled position
                y = y.squeeze(-1)  # (N) & (N) for BCELoss
            loss = criterion(y, target)
        return y, loss, None  # no need for sampled

    @no_grad()
    def infer(self, x: LongTensor) -> Tensor:
        """Infer from the classifier

        :param x: input sequences, of shape (N,T). Have to be padded to the left.
        :return: results for real samples, and fake samples
        """
        attention_mask = (x == self.config.pad_token_id)[..., 0]  # reduce of 1 dim to have good shape
        return self.forward(x, attention_mask=attention_mask)  # (N,T,E)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[Embedding, Tuple[Embedding]]:
        pass

    def _reorder_cache(self, past, beam_idx):
        pass


class MultiEmbeddings(Module):
    """Multi-input module, taking several tokens as input, converting them to embeddings and
    concatenate them to make a single 'merged' embedding
    :param num_classes: number of classes for each token type
    :param embedding_sizes: sizes of each embedding type
    :param d_model: size of the final embedding, i.e. dimension of the transformer
    :param padding_idx: padding index, must be the same for each token type
    """
    def __init__(self, num_classes: List[int], embedding_sizes: List[int], d_model: int, padding_idx: int = 0):
        assert len(num_classes) == len(embedding_sizes), \
            f'The number of classes and embedding sizes must be the same ({len(num_classes)} and ' \
            f'{len(embedding_sizes)} were given)'
        super().__init__()
        self.embedding_layers = ModuleList([Embedding(num_classes[i], embedding_sizes[i], padding_idx)
                                            for i in range(len(num_classes))])
        self.proj = Linear(sum(embedding_sizes), d_model)

    def forward(self, x: LongTensor) -> FloatTensor:
        """
        :param x: Tokens sequences, shape: (N,T,Z)
        :return: Embeddings, as a tensor with a shape (N,T,E)
        """
        embeds = []
        for i, mod in enumerate(self.embedding_layers):
            embeds.append(mod(x[..., i]))
        x = cat(embeds, dim=-1)  # (N,T,sum(embedding_sizes))
        return self.proj(x)  # (N,T,E)


class BertMultiEmbeddings(Module):
    """OVERRIDDEN TO REMOVE TOKEN TYPE EMBEDDING AS INCOMPATIBLE
    Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig, num_classes: List[int], embed_sizes: List[int]):
        super().__init__()
        self.word_embeddings = MultiEmbeddings(num_classes, embed_sizes, config.hidden_size, config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        token_type_ids=None,
        inputs_embeds: Optional[FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            inputs_embeds += position_embeddings
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiOutput(Module):
    """Multi-output module.
    :param num_classes: number of classes for each token type
    :param d_model: size of the final embedding, i.e. dimension of the transformer
    """
    def __init__(self, num_classes: List[int], d_model: int):
        super().__init__()
        self.output_layers = ModuleList([Linear(d_model, num) for num in num_classes])

    def forward(self, x: List[FloatTensor]) -> List[FloatTensor]:
        """
        :param x: Tokens sequences, shape: (L, N, E)
        :return: List of tensors of shape (L, N, *)
        """
        return [out(x) for out in self.output_layers]  # (L, N, *)


def convert_past_key_values_to_tensor(past_kv: Tuple) -> Tensor:
    """Convert past_key_values returned by HF model from tuple(tuple(Tensor)) to a Tensor.
    :param past_kv: tuple of past_key_val, shape (NLY,2,N,NH,T,DH) with first two dims as tuple
    :return: Tensor of shape (NLY,2,N,NH,T,DH)
    """
    return stack([stack([kv for kv in layer]) for layer in past_kv])  # (NLY,2,N,NH,T,DH)
