"""Model

"""

from typing import List, Tuple, Union, Optional, Callable

from torch import LongTensor, FloatTensor, Tensor, cat, stack, no_grad, nn, multinomial, softmax
from torch.nn import Module, ModuleList, Linear, Embedding, CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Config, BertForPreTraining, BertForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput
from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutputWithCrossAttentions
from transformers.generation.utils import top_k_top_p_filtering, GenerationConfig, StoppingCriteriaList, \
    LogitsProcessorList


class GPT2LMHeadModelEmbedPooling(GPT2LMHeadModel):
    """
    We override this class as we need to alter the way the loss is computed with embedding pooling.
    """
    def __init__(self, config: GPT2Config, num_classes: List[int], embed_sizes: List[int]):
        super().__init__(config)

        self.transformer.wte = MultiEmbeddings(num_classes, embed_sizes, config.n_embd, config.pad_token_id)
        self.lm_head = MultiOutput(num_classes=num_classes, d_model=config.n_embd)

    def forward(
            self,
            input_ids: Optional[LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[FloatTensor] = None,
            token_type_ids: Optional[LongTensor] = None,
            position_ids: Optional[LongTensor] = None,
            head_mask: Optional[FloatTensor] = None,
            inputs_embeds: Optional[FloatTensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[FloatTensor] = None,
            labels: Optional[LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pad_on_left: Optional[bool] = None,  # Subclassing to change signature for data collator
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Compute embeds here so that the input shape is kept for method below
        inputs_embeds = self.transformer.wte(input_ids)
        transformer_outputs = self.transformer(
            input_ids=None,
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

        lm_logits = self.lm_head(hidden_states)  # (N,T,C) or [Z (N,T,C*)]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = sum([loss_fct(pr[..., :-1, :].contiguous().view(-1, pr.size(-1)),
                                 labels[..., 1:, i].contiguous().view(-1))
                        for i, pr in enumerate(lm_logits)])  # [Z (N,T,C)] & (N,T,Z)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @no_grad()
    def generate(
            self,
            inputs: Optional[Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = False,
            **kwargs,
    ) -> LongTensor:

        if generation_config is None:
            generation_config = self.generation_config
        assert generation_config.max_new_tokens <= self.transformer.wpe.weight.shape[0], \
            'The maximum sequence length must be <= to the nb of positions the model can handle'
        inputs = kwargs.get("input_ids", inputs)
        y = inputs.clone()
        if y.dim() == 2:
            y = y.unsqueeze(0)  # (T,Z) --> (N,T,Z) with N=1
        past_key_val = None  # (NLY,2,N,NH,T,DH)
        tokens = y.clone()  # (N,T,Z)

        for _ in range(generation_config.max_new_tokens):
            # Adds the prediction to the target sequence, updates past key values and y sequence
            logits = self.forward(tokens, past_key_val)
            logits, past_key_val = logits.logits, logits.past_key_values  # [Z: (N,T,C)]
            tokens = []
            for token_type in logits:
                logit = top_k_top_p_filtering(token_type[:, -1].cpu(), generation_config.top_k, generation_config.top_p)
                logit = softmax(logit / generation_config.temperature, -1)  # (N,C)
                logit = multinomial(logit, 1)  # (N,1)
                tokens.append(logit)
            tokens = stack(tokens).permute(1, 2, 0).to(self.device)  # (Z,N,1) --> (N,1,Z)
            y = cat([y, tokens], dim=1)  # (N,T+1,Z)

        return y[0] if inputs.dim() == 2 else y  # (T,Z) or (N,T,Z)

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.wte

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[Embedding, Tuple[Embedding]]:
        pass


class BertForPreTrainingEmbedPooling(BertForPreTraining):
    """
    We override this class as we need to alter the way the loss is computed with embedding pooling.
    """
    def __init__(self, config: BertConfig, num_classes: List[int], embed_sizes: List[int]):
        super().__init__(config)

        self.bert.embeddings.word_embeddings = MultiEmbeddings(num_classes, embed_sizes, config.hidden_size,
                                                               padding_idx=config.pad_token_id)
        self.cls.predictions = MultiOutput(num_classes, config.hidden_size)
        """# Weight tying
        for i in range(len(num_classes)):
            self._tie_or_clone_weights(self.cls.predictions.output_layers[i],
                                       self.bert.embeddings.word_embeddings.embedding_layers[i])"""

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            next_sentence_label: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], BertForPreTrainingOutput, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Compute embeds here so that the input shape is kept for method below
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = sum([loss_fct(pr.view(-1, pr.size(-1)), labels[..., i].contiguous().view(-1))
                                  for i, pr in enumerate(prediction_scores)])  # [Z (N,T,C)] & (N,T,Z)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.bert.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self):
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

    def _reorder_cache(self, past, beam_idx):
        pass


class BertForSequenceClassificationEmbeddingPooling(BertForSequenceClassification):
    """
    We override the class to first compute pooled embeddings before giving it to the backbone BERT model.
    """
    def __init__(self, config: BertConfig, num_classes: List[int], embed_sizes: List[int]):
        super().__init__(config)

        self.bert.embeddings.word_embeddings = MultiEmbeddings(num_classes, embed_sizes, config.hidden_size,
                                                               padding_idx=config.pad_token_id)

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], BertForPreTrainingOutput, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Compute embeds here so that the input shape is kept for method below
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


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
        self.weight = None  # to mock weight tying in hf from_pretrained

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


class MultiOutput(Module):
    """Multi-output module.
    :param num_classes: number of classes for each token type
    :param d_model: size of the final embedding, i.e. dimension of the transformer
    """
    def __init__(self, num_classes: List[int], d_model: int):
        super().__init__()
        self.output_layers = ModuleList([Linear(d_model, num) for num in num_classes])
        self.weight = None  # to mock weight tying in hf from_pretrained

    def forward(self, x: List[FloatTensor]) -> List[FloatTensor]:
        """
        :param x: Tokens sequences, shape: (L, N, E)
        :return: List of tensors of shape (L, N, *)
        """
        return [out(x) for out in self.output_layers]  # Z (L, N, *)

    """ This does not work as in and out modules does not have the same dimensions
    @property
    def weight(self):
        return [mod.weight for mod in self.output_layers]

    @weight.setter
    def weight(self, value):
        for mod, arg_weight in zip(self.output_layers, value):
            mod.weight = arg_weight"""
