"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    vocab = {}
    for idx, token in enumerate(specials):
        vocab[token] = idx

    current_id = len(specials)

    for sentence in sentences:
        tokens = sentence.split()

        for token in tokens:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1
                
    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    id_to_token = {}

    for token, idx in token_to_id.items():
        id_to_token[idx] = token

    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    unk_id = token_to_id[unk_token]
    ids = []
    if not sentence:
        return ids
    tokens = sentence.split()

    for token in tokens:
        if token in token_to_id:
            ids.append(token_to_id[token])
        else:
            ids.append(unk_id)
    return ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    tokens = []
    
    for idx in ids:
        token = id_to_token[idx]
        tokens.append(token)
        
    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    if len(ids) > max_len:
        return ids[:max_len]
        
    padded_ids = list(ids)
    while len(padded_ids) < max_len:
        padded_ids.append(pad_id)
        
    return padded_ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    batch_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    
    return batch_tensor

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Rescales a token embedding tensor by the square root of d_model."""
    scaling_factor = math.sqrt(d_model)
    scaled_embeddings = embeddings * scaling_factor
    return scaled_embeddings

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    """Computes the 1D frequency divisor tensor of length d_model // 2."""
    half_dim = d_model // 2
    
    div_terms = []
    
    for i in range(half_dim):
        exponent = -(2 * i) / d_model * torch.log(torch.tensor(10000.0))
        val = torch.exp(exponent)
        div_terms.append(val.item())
        
    return torch.tensor(div_terms, dtype=torch.float32)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    indices = torch.arange(max_len, dtype=torch.float32)
    column_vector = indices.unsqueeze(1)
    return column_vector

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch


def fill_odd_indices_with_cos(pe, position, div_term):
    # Fill odd-indexed columns of pe with the cosine of position times div_term
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    # Assemble the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)
    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # Add the first L rows of positional encoding to the embedded batch with broadcasting
    return embedded_batch + positional_encoding[: embedded_batch.size(1)]

# Step 14 - build_padding_mask
import torch


def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # Create the base boolean mask (B, L) where real tokens are True and padding is False
    mask = token_ids != pad_id
    # Unsqueeze dimensions twice to shape it as (B, 1, 1, L) for multi-head attention broadcasting
    return mask.unsqueeze(1).unsqueeze(1)

# Step 15 - build_causal_mask
import torch


def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # Create a lower-triangular boolean matrix of shape (seq_len, seq_len)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    # Unsqueeze dimensions twice to shape it as (1, 1, seq_len, seq_len) for attention broadcasting
    return mask.unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(q, k):
    return torch.matmul(q, k.transpose(-1, -2))

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return scores.masked_fill(mask == 0, float('-inf'))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    # Handle rows that are entirely -inf to prevent NaNs
    all_inf_mask = torch.isinf(masked_scores) & (masked_scores < 0)
    # If a row is all -inf across the last dimension, temporarily replace with 0 so softmax yields uniform/zero, then zero them out
    # A cleaner approach in PyTorch is using torch.where or handling explicitly:
    weights = F.softmax(masked_scores, dim=-1)
    weights = weights.masked_fill(all_inf_mask, 0.0)
    return weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(weights, v):
    return torch.matmul(weights, v)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    # 1. Compute raw attention scores
    scores = torch.matmul(q, k.transpose(-1, -2))
    
    # 2. Scale by the square root of the key dimension (d_k)
    d_k = q.size(-1)
    scores = scores / (d_k ** 0.5)
    
    # 3. Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
    # 4. Compute attention weights via softmax
    weights = torch.softmax(scores, dim=-1)
    
    # Handle rows that are entirely -inf to prevent NaNs
    all_inf_mask = torch.isinf(scores) & (scores < 0)
    weights = weights.masked_fill(all_inf_mask, 0.0)
    
    # 5. Apply weights to values
    return torch.matmul(weights, v), weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    batch_size, seq_len, d_model = tensor.shape
    d_k = d_model // num_heads
    return tensor.view(batch_size, seq_len, num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    batch_size, num_heads, seq_len, d_k = multi_head_tensor.shape
    # Transpose back from (B, H, L, d_k) to (B, L, H, d_k) and reshape to (B, L, H * d_k)
    return multi_head_tensor.transpose(1, 2).reshape(batch_size, seq_len, num_heads * d_k).contiguous()

# Step 26 - apply_linear_projection
import torch

def apply_linear_projection(x, weight, bias):
    # Perform right multiplication by weight transpose: x @ weight.T
    out = torch.matmul(x, weight.t())
    if bias is not None:
        out = out + bias
    return out

# Step 27 - project_to_query_key_value
import torch

def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)
    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # q_h, k_h, v_h shape: (B, num_heads, L, d_k)
    # scaled_dot_product_attention can operate directly on these tensors because 
    # PyTorch batch matrix multiplication handles any number of batch dimensions smoothly.
    return scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
import torch

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # 1. Project Q, K, V
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)
    
    # 2. Split into heads
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    
    # 3. Run scaled dot-product attention
    context, _ = scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)
    
    # 4. Merge heads and project output
    return merge_heads_and_project_output(context, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
import torch

def apply_ffn_first_linear_and_relu(x, w_1, b_1):
    # apply_linear_projection does x @ w.t(), but here w1 has shape (d_model, d_ff).
    # To match x @ w1 + b1, we can do manual matmul and bias add:
    return torch.relu(torch.matmul(x, w_1) + b_1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    return torch.matmul(hidden, w2) + b2

# Step 34 - position_wise_feed_forward_network
import torch

def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, variance = compute_layer_norm_mean_and_variance(x)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return normalized * gamma + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    combined = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
import torch

def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    if keep_prob == 0.0:
        return torch.zeros_like(x)
    return x * keep_mask.float() / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
import torch

def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, mask=None):
    attn_out = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, mask=mask)
    return apply_residual_add_and_norm(x, attn_out, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
import torch

def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    ffn_out = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, ffn_out, gamma, beta)

# Step 41 - assemble_encoder_layer
import torch

def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # 1. Self-attention sublayer
    x = encoder_layer_self_attention_sublayer(
        x, 
        layer_params['w_q'], layer_params['w_k'], layer_params['w_v'], layer_params['w_o'], 
        layer_params['attn_gamma'], layer_params['attn_beta'], 
        num_heads, src_mask
    )
    # 2. Feed-forward sublayer
    x = encoder_layer_feed_forward_sublayer(
        x, 
        layer_params['w1'], layer_params['b1'], 
        layer_params['w2'], layer_params['b2'], 
        layer_params['ffn_gamma'], layer_params['ffn_beta']
    )
    return x

# Step 42 - stack_encoder_layers
import torch

def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    attn_out = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, mask=tgt_mask)
    return apply_residual_add_and_norm(y, attn_out, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    if src_mask is not None and src_mask.ndim == 2:
        src_mask = src_mask.unsqueeze(1).unsqueeze(1)
    
    attn_out = assemble_multi_head_attention_forward(
        query=y, 
        key=encoder_output, 
        value=encoder_output, 
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, 
        num_heads=num_heads, 
        mask=src_mask
    )
    return apply_residual_add_and_norm(y, attn_out, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta, eps=1e-5):
    ffn_out = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, ffn_out, gamma, beta, eps)

# Step 46 - assemble_decoder_layer
import torch

def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    # 1. Masked self-attention sublayer
    y = decoder_layer_masked_self_attention_sublayer(
        y, 
        layer_params['w_q_self'], layer_params['w_k_self'], layer_params['w_v_self'], layer_params['w_o_self'], 
        layer_params['self_gamma'], layer_params['self_beta'], 
        num_heads, tgt_mask
    )
    # 2. Cross-attention sublayer
    y = decoder_layer_cross_attention_sublayer(
        y, 
        encoder_output, 
        layer_params['w_q_cross'], layer_params['w_k_cross'], layer_params['w_v_cross'], layer_params['w_o_cross'], 
        layer_params['cross_gamma'], layer_params['cross_beta'], 
        num_heads, src_mask
    )
    # 3. Feed-forward sublayer
    y = decoder_layer_feed_forward_sublayer(
        y, 
        layer_params['w1'], layer_params['b1'], 
        layer_params['w2'], layer_params['b2'], 
        layer_params['ffn_gamma'], layer_params['ffn_beta']
    )
    return y

# Step 47 - stack_decoder_layers
import torch

def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    for layer_params in decoder_layer_params_list:
        y = assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask)
    return y

# Step 48 - apply_final_output_projection
import torch

def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    return token_embedding_weight.t()

# Step 50 - apply_log_softmax_over_vocab
import torch

def apply_log_softmax_over_vocab(logits):
    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
import torch

def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    d_model = model_params['src_embedding'].shape[1]
    
    # 1. Embed and scale source and target
    src_emb = model_params['src_embedding'][src_ids]
    src_emb = scale_embeddings_by_sqrt_d_model(src_emb, d_model)
    
    tgt_emb = model_params['tgt_embedding'][tgt_ids]
    tgt_emb = scale_embeddings_by_sqrt_d_model(tgt_emb, d_model)
    
    # 2. Add positional encodings
    src_pe = build_sinusoidal_positional_encoding(src_ids.shape[1], d_model)
    tgt_pe = build_sinusoidal_positional_encoding(tgt_ids.shape[1], d_model)
    
    # Ensure PE matches device and dtype if needed
    src_pe = src_pe.to(src_emb.device, dtype=src_emb.dtype)
    tgt_pe = tgt_pe.to(tgt_emb.device, dtype=tgt_emb.dtype)
    
    x = add_positional_encoding_to_embeddings(src_emb, src_pe)
    y = add_positional_encoding_to_embeddings(tgt_emb, tgt_pe)
    
    # 3. Build masks
    src_padding_mask = build_padding_mask(src_ids, pad_id)
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    tgt_causal_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, tgt_causal_mask)
    
    # 4. Run encoder stack
    encoder_output = stack_encoder_layers(x, model_params['encoder_layers'], num_heads, src_padding_mask)
    
    # 5. Run decoder stack
    decoder_output = stack_decoder_layers(y, encoder_output, model_params['decoder_layers'], num_heads, src_padding_mask, tgt_mask)
    
    # 6. Project to vocabulary logits and apply log softmax
    logits = apply_final_output_projection(decoder_output, model_params['output_projection'])
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    std = 1.0 / math.sqrt(d_model)
    
    w_q = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_k = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_v = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_o = torch.randn(d_model, d_model, dtype=torch.float32) * std
    
    w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * (1.0 / math.sqrt(d_model))
    b1 = torch.zeros(d_ff, dtype=torch.float32)
    
    w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * (1.0 / math.sqrt(d_ff))
    b2 = torch.zeros(d_model, dtype=torch.float32)
    
    attn_gamma = torch.ones(d_model, dtype=torch.float32)
    attn_beta = torch.zeros(d_model, dtype=torch.float32)
    
    ffn_gamma = torch.ones(d_model, dtype=torch.float32)
    ffn_beta = torch.zeros(d_model, dtype=torch.float32)
    
    parameters = {
        'w_q': w_q.requires_grad_(True),
        'w_k': w_k.requires_grad_(True),
        'w_v': w_v.requires_grad_(True),
        'w_o': w_o.requires_grad_(True),
        'w1': w1.requires_grad_(True),
        'b1': b1.requires_grad_(True),
        'w2': w2.requires_grad_(True),
        'b2': b2.requires_grad_(True),
        'attn_gamma': attn_gamma.requires_grad_(True),
        'attn_beta': attn_beta.requires_grad_(True),
        'ffn_gamma': ffn_gamma.requires_grad_(True),
        'ffn_beta': ffn_beta.requires_grad_(True),
    }
    
    return parameters

# Step 53 - init_decoder_layer_parameters
import torch
import math

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one decoder layer."""
    std = 1.0 / math.sqrt(d_model)
    
    # Self-attention projection matrices (no bias)
    w_q_self = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_k_self = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_v_self = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_o_self = torch.randn(d_model, d_model, dtype=torch.float32) * std
    
    # Cross-attention projection matrices (no bias)
    w_q_cross = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_k_cross = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_v_cross = torch.randn(d_model, d_model, dtype=torch.float32) * std
    w_o_cross = torch.randn(d_model, d_model, dtype=torch.float32) * std
    
    # Feed-forward weights and biases
    w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * (1.0 / math.sqrt(d_model))
    b1 = torch.zeros(d_ff, dtype=torch.float32)
    
    w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * (1.0 / math.sqrt(d_ff))
    b2 = torch.zeros(d_model, dtype=torch.float32)
    
    # LayerNorm parameters for self-attention, cross-attention, and feed-forward
    self_gamma = torch.ones(d_model, dtype=torch.float32)
    self_beta = torch.zeros(d_model, dtype=torch.float32)
    
    cross_gamma = torch.ones(d_model, dtype=torch.float32)
    cross_beta = torch.zeros(d_model, dtype=torch.float32)
    
    ffn_gamma = torch.ones(d_model, dtype=torch.float32)
    ffn_beta = torch.zeros(d_model, dtype=torch.float32)
    
    parameters = {
        'w_q_self': w_q_self.requires_grad_(True),
        'w_k_self': w_k_self.requires_grad_(True),
        'w_v_self': w_v_self.requires_grad_(True),
        'w_o_self': w_o_self.requires_grad_(True),
        'self_gamma': self_gamma.requires_grad_(True),
        'self_beta': self_beta.requires_grad_(True),
        'w_q_cross': w_q_cross.requires_grad_(True),
        'w_k_cross': w_k_cross.requires_grad_(True),
        'w_v_cross': w_v_cross.requires_grad_(True),
        'w_o_cross': w_o_cross.requires_grad_(True),
        'cross_gamma': cross_gamma.requires_grad_(True),
        'cross_beta': cross_beta.requires_grad_(True),
        'w1': w1.requires_grad_(True),
        'b1': b1.requires_grad_(True),
        'w2': w2.requires_grad_(True),
        'b2': b2.requires_grad_(True),
        'ffn_gamma': ffn_gamma.requires_grad_(True),
        'ffn_beta': ffn_beta.requires_grad_(True),
    }
    
    return parameters

# Step 54 - init_embedding_and_projection_parameters
import torch
import math

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    std = 1.0 / math.sqrt(d_model)
    
    src_embedding = torch.randn(vocab_size, d_model, dtype=torch.float32) * std
    tgt_embedding = torch.randn(vocab_size, d_model, dtype=torch.float32) * std
    
    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = torch.randn(vocab_size, d_model, dtype=torch.float32) * std
        
    return {
        'src_embedding': src_embedding.requires_grad_(True),
        'tgt_embedding': tgt_embedding.requires_grad_(True),
        'output_projection': output_projection.requires_grad_(True),
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    params_list = []
    seen_ids = set()
    
    def add_tensor(tensor):
        if id(tensor) not in seen_ids:
            seen_ids.add(id(tensor))
            params_list.append(tensor)
            
    # 1. Encoder layers
    for layer in encoder_layer_params:
        for val in layer.values():
            add_tensor(val)
            
    # 2. Decoder layers
    for layer in decoder_layer_params:
        for val in layer.values():
            add_tensor(val)
            
    # 3. Embedding and output projection parameters
    for val in embedding_params.values():
        add_tensor(val)
        
    return params_list

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 68 - compute_adam_parameter_update (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

