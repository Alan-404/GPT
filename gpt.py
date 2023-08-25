import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable
import numpy as np

class GPT(nn.Module):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.embed = TextAndEmbed(token_size, d_model)
        self.decoder = Decoder(n, d_model, heads, d_ff, activation, dropout_rate, eps)
        self.norm_layer = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.classifier = nn.Linear(in_features=d_model, out_features=token_size)

    def forward(self, x: torch.Tensor):
        if self.training:
            mask = generate_look_ahead_mask(x)
        else:
            mask = None
        x = self.embed(x)
        x = self.decoder(x, mask)
        x = self.norm_layer(x)
        x = self.classifier(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, activation, dropout_rate, eps) for _ in range(n)])
        self.dropout_rate = dropout_rate
    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class TextAndEmbed(nn.Module):
    def __init__(self, token_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.positional_encoder = PositionalEncoding()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        x = self.positional_encoder(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        assert self.d_model % self.heads == 0

        self.head_samples = self.d_model//self.heads

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

        self.dropout_rate = dropout_rate

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        dk = torch.tensor(k.size(-1))
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores/torch.sqrt(dk)

        if mask is not None:
            attention_scores += mask*(-1e15)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_context = torch.matmul(attention_weights, v)

        return attention_context
    
    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ctx, embedding_dim = x.size()

        assert embedding_dim == self.d_model

        x = x.reshape((batch_size, n_ctx, self.heads, self.head_samples))
        x = x.permute((0, 2, 1, 3))

        return x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        batch_size, n_ctx, _ = q.size()

        qw = F.dropout(self.linear_q(q), p=self.dropout_rate, training=self.training)
        kw = F.dropout(self.linear_k(k), p=self.dropout_rate, training=self.training)
        vw = F.dropout(self.linear_v(v), p=self.dropout_rate, training=self.training)

        q_heads = self.split_head(qw)
        k_heads = self.split_head(kw)
        v_heads = self.split_head(vw)

        attention_context = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention_context = attention_context.permute((0, 2, 1, 3))
        attention_context = attention_context.reshape((batch_size, n_ctx, self.d_model))

        attention_context = self.linear_output(attention_context)
        return attention_context

class PositonWiseFeedForwardNetworks(nn.Module):
    def __init__(self, d_ff: int, d_model: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float = 0.1) -> None:
        super().__init__()      
        self.hidden_layer = nn.Linear(in_features=d_model, out_features=d_ff)
        self.activation = activation
        self.output_layer = nn.Linear(in_features=d_ff, out_features=d_model)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.attention_layer = MultiHeadAttention(heads, d_model, dropout_rate)
        self.ffn = PositonWiseFeedForwardNetworks(d_ff, d_model, activation, dropout_rate)

        self.norm_1 = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.norm_2 = nn.LayerNorm(normalized_shape=d_model, eps=eps)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        # sublayer 1
        norm_x = self.norm_1(x)
        attention_output = self.attention_layer(norm_x, norm_x, norm_x, mask)
        attention_output = F.dropout(attention_output, p=self.dropout_rate, training=self.training) + x

        # sublayer 2
        norm_attention = self.norm_2(attention_output)
        ffn_output = self.ffn(norm_attention)
        ffn_output = F.dropout(ffn_output, p=self.dropout_rate, training=self.training) +  attention_output
        
        return ffn_output

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def __encode_ctx(self, n_ctx: int) -> torch.Tensor:
        pos = torch.arange(n_ctx)
        pos = pos.unsqueeze(-1)
        return pos.type(torch.float32)
    
    def __encode_embedding(self, embedding_dim: int) -> torch.Tensor:
        angles = torch.arange(embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1/(torch.pow(10000, angles/embedding_dim))
        angles = angles.unsqueeze(0)
        return angles
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.__encode_ctx(x.size(1))
        angles = self.__encode_embedding(x.size(2))
        
        pos_angles = torch.matmul(pos, angles)
        pos_angles[0::2] = torch.sin(pos_angles[0::2])
        pos_angles[1::2] = torch.cos(pos_angles[1::2])

        pos_angles = pos_angles.unsqueeze(0)
        x += pos_angles.to(x.device)
        return x
    
def generate_padding_mask(tensor: torch.Tensor)-> torch.Tensor:
    return torch.Tensor(tensor == 0).type(torch.int64)[:, np.newaxis, np.newaxis, :]

def __generate_look_ahead_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.ones((length, length)), diagonal=1)

def generate_look_ahead_mask(tensor: torch.Tensor) -> torch.Tensor:
    padding_mask = generate_padding_mask(tensor)

    look_ahead_mask = __generate_look_ahead_mask(tensor.size(1)).to(tensor.device)

    look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask)

    return look_ahead_mask