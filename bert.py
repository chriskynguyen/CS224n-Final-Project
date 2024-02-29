# Retrofitted code for PALs referring to 
# https://github.com/AsaCooperStickland/Bert-n-Pals for implementation of PALs
# particularly BertPals, parts of BertLayer, and the encoder section in BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *

HIDDEN_SIZE_AUG = 204
NUM_TASKS = 3


class BertSelfAttention(nn.Module):
  def __init__(self, config, multi_params=None):
    super().__init__()
    if multi_params is not None: # used for pals
      self.num_attention_heads = multi_params
      self.attention_head_size = int(HIDDEN_SIZE_AUG / self.num_attention_heads)
      self.all_head_size = self.num_attention_heads * self.attention_head_size
      hidden_size = HIDDEN_SIZE_AUG
    else:
      self.num_attention_heads = config.num_attention_heads
      self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
      self.all_head_size = self.num_attention_heads * self.attention_head_size
      hidden_size = config.hidden_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(hidden_size, self.all_head_size)
    self.key = nn.Linear(hidden_size, self.all_head_size)
    self.value = nn.Linear(hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, use the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    # Make sure to:
    # - Normalize the scores with softmax.
    # - Multiply the attention scores with the value to get back weighted values.
    # - Before returning, concatenate multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

    ### TODO: FINISHED
    # Compute attention scores. Score matrix S (unnormalized)
    S = (query @ key.transpose(-2, -1)) / (self.attention_head_size**0.5) # [bs, num_attention_heads, seq_len, seq_len]
    # Mask out padding token scores 
    S = S + attention_mask # masked_fill already done through get_extended_attention_mask so only adding score to mask is necessary 
    # Normalize score
    att = F.softmax(S, dim=-1)
    att = self.dropout(att) # following original implementation of transformer
    attn_value = att @ value # [bs, num_attention_heads, seq_len, attention_head_size]
    # Concatenate heads to make shape: [bs, seq_len, hidden_size]
    attn_value = attn_value.transpose(1, 2).contiguous().view(query.size(0), query.size(-2), self.all_head_size) # self.all_head_size == hidden_size
    return attn_value

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertPals(nn.Module):
  """
  Class implementing task specific PALs layer (TS). Reference
  Equation: TS(h) = V_D (SA(V_E(h))) 
  V_E is the encoder layer and V_D is the decoder layer
  """
  def __init__(self, config):
    super().__init__()
    # Encoder and decoder matrices project down to the smaller dimension
    self.encoder_layer = nn.Linear(config.hidden_size, HIDDEN_SIZE_AUG)
    self.decoder_layer = nn.Linear(HIDDEN_SIZE_AUG, config.hidden_size)
    # Attention without the final matrix multiply.
    self.attn = BertSelfAttention(config, 12)
    self.hidden_act_fn = F.gelu

  def forward(self, hidden_states, attention_mask=None):
    hidden_states_aug = self.encoder_layer(hidden_states) # hidden_states_aug = V_E(h)
    hidden_states_aug = self.attn(hidden_states_aug, attention_mask)  # hidden_states_aug = SA(hidden_states_aug)
    hidden_states = self.decoder_layer(hidden_states_aug) # hidden_states = V_D(hidden_states_aug)
    hidden_states = self.hidden_act_fn(hidden_states) 
    return hidden_states


class BertLayer(nn.Module):
  def __init__(self, config, mult=False):
    super().__init__()
    # Multi-head attention.
    self.self_attention = BertSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
    if mult:
      multi = BertPals(config)
      self.multi_layers = nn.ModuleList([copy.deepcopy(multi) for _ in range(NUM_TASKS)])
    self.mult = mult

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
    # before it is added to the sub-layer input and normalized with a layer norm.
    ### TODO: FINISHED
    #Apply dropout to output of each sub-layer
    drop_output = dropout(dense_layer(output))
    # Added to sub-layer input
    residual = input + drop_output
    # Normalize with a layer norm
    norm = ln_layer(residual)
    return norm


  def forward(self, hidden_states, attention_mask, i=0):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    ### TODO: FINISHED
    # Multi-head attention layer
    att_output = self.self_attention(hidden_states, attention_mask)
    # Add-norm for multi-head attention layer
    att_norm = self.add_norm(hidden_states, att_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    # Feed forward layer
    feed_output = self.interm_af(self.interm_dense(att_norm))
    # Add PALs to final output
    if self.mult:
      extra_pals = self.multi_layers[i](hidden_states, attention_mask)
      layer_output = self.add_norm(att_norm + extra_pals, feed_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    else:
      # Add-norm for feed forward layer
      layer_output = self.add_norm(att_norm, feed_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    return layer_output


class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder.
    self.multis = [True if i < 999 else False for i in range(config.num_hidden_layers)]
    self.bert_layers = nn.ModuleList([BertLayer(config, mult=mult) for mult in self.multis])

    dense_encoder = nn.Linear(config.hidden_size, HIDDEN_SIZE_AUG)
    # Shared encoder and decoder across layers
    self.mult_encoder_layer = nn.ModuleList([copy.deepcopy(dense_encoder) for _ in range(NUM_TASKS)])
    dense_decoder = nn.Linear(HIDDEN_SIZE_AUG, config.hidden_size)
    self.mult_decoder_layer = nn.ModuleList([copy.deepcopy(dense_decoder) for _ in range(NUM_TASKS)])
    for l, layer in enumerate(self.bert_layers):
      if self.multis[l]:
        for i, lay in enumerate(layer.multi_layers):
          lay.encoder_layer = self.mult_encoder_layer[i]
          lay.decoder_layer = self.mult_decoder_layer[i]

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = None
    ### TODO: FINISHED
    inputs_embeds = self.word_embedding(input_ids)

    # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = None
    ### TODO: FINISHED
    pos_embeds = self.pos_embedding(pos_ids)

    # Get token type ids. Since we are not considering token type, this embedding is
    # just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    ### TODO: FINISHED
    combined_embeds = inputs_embeds + pos_embeds + tk_type_embeds
    embed_norm = self.embed_layer_norm(combined_embeds)
    return self.embed_dropout(embed_norm)


  def encode(self, hidden_states, attention_mask, i=0):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for j, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask, i)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of BertLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # Get cls token hidden state.
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}