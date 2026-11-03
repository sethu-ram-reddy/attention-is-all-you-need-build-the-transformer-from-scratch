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

# Step 56 - shift_targets_right_with_start_token
import torch

def shift_targets_right_with_start_token(target_ids, start_token_id):
    batch_size, seq_len = target_ids.shape
    device = target_ids.device
    dtype = target_ids.dtype
    
    # Create start token column of shape (batch_size, 1) with the same dtype and device
    start_col = torch.full((batch_size, 1), start_token_id, dtype=dtype, device=device)
    
    # If seq_len is 1, the shifted target is just the start column
    if seq_len <= 1:
        return start_col
    
    # Otherwise, prepend start_col and drop the last column of target_ids
    return torch.cat([start_col, target_ids[:, :-1]], dim=1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    return (d_model ** (-0.5)) * min(step ** (-0.5), step * (warmup_steps ** (-1.5)))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    fill_value = epsilon / (vocab_size - 2) if (vocab_size - 2) > 0 else 0.0
    return torch.full(shape, fill_value, dtype=torch.float32)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    out = smoothed_distribution.clone()
    batch_indices = torch.arange(smoothed_distribution.size(0)).unsqueeze(1).expand_as(gold_token_ids)
    time_indices = torch.arange(smoothed_distribution.size(1)).unsqueeze(0).expand_as(gold_token_ids)
    
    out[batch_indices, time_indices, gold_token_ids] = confidence
    return out

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    out = smoothed_distribution.clone()
    
    # 1. Zero out the entire pad column across the vocabulary axis
    out[..., pad_id] = 0.0
    
    # 2. Zero out full rows where the gold token id equals pad_id
    pad_mask = (gold_token_ids == pad_id)
    out[pad_mask] = 0.0
    
    return out

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed label smoothing cross-entropy loss over all entries."""
    loss = -torch.sum(smoothed_distribution * log_probabilities)
    return torch.where(loss == 0, torch.zeros_like(loss), loss)

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # Count the number of non-pad tokens
    non_pad_count = (gold_token_ids != pad_id).sum().item()
    
    # If every target position is padding, return total_loss as a fallback
    if non_pad_count == 0:
        return total_loss
        
    return total_loss / non_pad_count

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    """Compute token prediction accuracy ignoring padding positions."""
    # Take argmax over the vocabulary axis (last dimension)
    predictions = torch.argmax(log_probabilities, dim=-1)
    
    # Create mask for non-pad positions
    non_pad_mask = (gold_token_ids != pad_id)
    total_non_pad = non_pad_mask.sum().item()
    
    # If every token is padding, return 0.0
    if total_non_pad == 0:
        return torch.tensor(0.0, dtype=torch.float32)
        
    # Count correct predictions where mask is True
    correct = ((predictions == gold_token_ids) & non_pad_mask).sum().item()
    
    return torch.tensor(correct / total_non_pad, dtype=torch.float32)

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    m = []
    v = []
    for p in parameter_list:
        m.append(torch.zeros_like(p, dtype=p.dtype, device=p.device).detach())
        v.append(torch.zeros_like(p, dtype=p.dtype, device=p.device).detach())
        
    return {
        'm': m,
        'v': v,
        't': 0
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m, grad, beta1):
    """Apply one exponential moving average update for Adam's first moment."""
    return (beta1 * m + (1.0 - beta1) * grad).detach()

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    return (beta2 * v_prev + (1.0 - beta2) * (grad ** 2)).detach()

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    m_hat = m_t / (1.0 - (beta1 ** step))
    v_hat = v_t / (1.0 - (beta2 ** step))
    return m_hat, v_hat

# Step 68 - compute_adam_parameter_update
import torch

def compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon):
    """Return delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon); the caller subtracts it."""
    delta = learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
    return delta.detach()

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    """Run one full Adam update over every parameter in the list."""
    # 1. Increment step counter
    optimizer_state['t'] += 1
    t = optimizer_state['t']
    
    m_list = optimizer_state['m']
    v_list = optimizer_state['v']
    
    for i, p in enumerate(parameter_list):
        if p.grad is None:
            continue
            
        grad = p.grad.data
        
        # 2. Update first and second moments
        m_list[i] = update_adam_first_moment(m_list[i], grad, beta1)
        v_list[i] = update_adam_second_moment(v_list[i], grad, beta2)
        
        # 3. Apply bias correction
        m_hat, v_hat = apply_adam_bias_correction(m_list[i], v_list[i], beta1, beta2, t)
        
        # 4. Compute parameter update delta
        delta = compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon)
        
        # 5. Subtract delta from parameter data in place
        p.data.sub_(delta)
        
    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the accumulated gradient on every parameter tensor."""
    for p in parameter_list:
        if p.grad is not None:
            p.grad = None

# Step 71 - compute_batch_training_loss
import torch

def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    """One teacher-forced forward pass -> label-smoothed KL loss averaged over non-pad tokens."""
    pad_id = config['pad_id']
    start_id = config['start_id']
    vocab_size = config['vocab_size']
    smoothing = config['smoothing']
    num_heads = config['num_heads']
    
    # 1. Shift target sequence right with the start token for decoder input
    tgt_shifted = shift_targets_right_with_start_token(tgt_batch, start_id)
    
    # 2. Run the full Transformer forward pass to get log probabilities over the vocabulary
    log_probs = run_transformer_forward(src_batch, tgt_shifted, model_params, num_heads, pad_id)
    
    # 3. Build uniform smoothing distribution based on log_probs shape
    dist = build_uniform_smoothing_distribution(log_probs.shape, vocab_size, smoothing)
    
    # 4. Set confidence on gold tokens (using unshifted target sequence tgt_batch)
    dist = set_confidence_on_gold_tokens(dist, tgt_batch, 1.0 - smoothing)
    
    # 5. Zero out pad column and pad token rows
    dist = zero_pad_column_and_pad_token_rows(dist, tgt_batch, pad_id)
    
    # 6. Compute label smoothed KL loss per position (summed)
    total_loss = compute_label_smoothed_kl_loss(log_probs, dist)
    
    # 7. Average the loss over non-pad tokens
    avg_loss = average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)
    
    return avg_loss

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.
    Returns the scalar loss value for the step as a Python float.
    """
    # 1. Clear stale gradients
    zero_all_parameter_gradients(parameter_list)
    
    # 2. Run forward pass to compute batch loss
    loss = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)
    
    # 3. Backpropagate gradients
    loss.backward()
    
    # 4. Look up Noam learning rate for step_number
    d_model = config['d_model']
    warmup_steps = config['warmup_steps']
    lr = compute_noam_learning_rate(step_number, d_model, warmup_steps)
    
    # 5. Apply one Adam update across parameter_list
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.98)
    epsilon = config.get('epsilon', 1e-9)
    
    apply_adam_step_to_all_parameters(parameter_list, optimizer_state, lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
    
    # Return scalar loss value as a Python float
    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    losses = []
    num_batches = len(batches)
    
    for step in range(1, num_steps + 1):
        # Cycle through batches using modulo
        batch_idx = (step - 1) % num_batches
        src_batch, tgt_batch = batches[batch_idx]
        
        # Run training step with backprop
        loss = run_training_step_with_backprop(
            src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step, config
        )
        losses.append(loss)
        
    return losses

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.
    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    return torch.argmax(final_step_logits, dim=-1).to(torch.int64)

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

