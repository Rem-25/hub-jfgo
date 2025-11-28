import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = hidden_size // num_attention_heads

        # self-attention
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.Linear(hidden_size, hidden_size)

        #层归一化
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)

        #feed_forward
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)

        #初始化权重
        self._init_weights()


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x,attention_mask=None):
        attention_output = self.self_attention(x, attention_mask)
        # 残差连接和归一化
        x = self.attention_layer_norm(x + attention_output)

        # 前馈网络
        feed_forward_output = self.feed_forward(x)

        # 再次残差连接和归一化
        x = self.ff_layer_norm(x + feed_forward_output)

        return  x

    def self_attention(self, x, attention_mask= None):
        batch_size, seq_len, hidden_size = x.shape()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        attention_scores = torch.matmul

        return attention_scores, q, k, v, attention_mask
