import math

import torch

from torch import nn
from torch.nn.functional import dropout


def count_parameters(model):
    """
    计算模型中可训练参数的总数。

    Args:
        model (torch.nn.Module): 要计算参数的模型。

    Returns:
        int: 模型中可训练参数的总数。
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型的参数总数：", total_params)
    return total_params

class BiRNN_LSTM_Emotion(nn.Module):
    """
    双向LSTM情感分类模型。

    Args:
        vocab_size (int): 词汇表大小。
        embed_size (int): 词向量维度。
        num_hiddens (int): LSTM隐藏层单元数。
        num_layers (int): LSTM层数。
        num_outputs (int): 输出类别数。
        dropout (float, optional): dropout概率。 Defaults to 0.0.

    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, num_outputs, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(4 * num_hiddens, num_outputs)
        self.num_hiddens = num_hiddens

    def forward(self, inputs):
        """
        模型前向传播。

        Args:
            inputs (torch.Tensor): 输入的文本序列，形状为 (batch_size, seq_len)。

        Returns:
            torch.Tensor: 模型输出，形状为 (batch_size, num_outputs)。
        """
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0, :, :], outputs[-1, :, :]), dim=1)
        encoding  = self.dropout(encoding)
        outs = self.decoder(encoding)
        return outs

def sequence_mask(X, valid_len, value=0):
    """
    使用掩码屏蔽序列中的填充标记。

    Args:
        X (torch.Tensor): 输入的序列，形状为 (batch_size, seq_len)。
        valid_len (torch.Tensor): 每个序列的有效长度，形状为 (batch_size,)。
        value (int, optional): 用于填充的值。 Defaults to 0.
    Returns:
        torch.Tensor: 屏蔽后的序列，形状为 (batch_size, seq_len)。
    """
    maxlen = X.size(1)  # 获取序列的最大长度
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]  # 生成掩码，True表示有效部分，False表示填充部分
    X[~mask] = value  # 将填充部分替换为指定的值
    return X


def masked_softmax(logits, valid_lens):
    """
    对最后一个轴执行softmax操作，同时屏蔽无效位置。

    Args:
        logits (torch.Tensor): 形状为 (batch_size, num_steps, num_outputs) 的logits张量。
        valid_lens (torch.Tensor): 形状为 (batch_size,) 的张量，表示每个序列的有效长度。

    Returns:
        torch.Tensor: softmax后的张量，形状与logits相同。
    """
    if valid_lens is None:
        return nn.functional.softmax(logits, dim=-1)
    else:
        shape = logits.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        logits = sequence_mask(logits.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(logits.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """
    点积注意力机制。

    Args:
        dropout (float): dropout概率。
        **kwargs: 其他关键字参数。
    """
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        前向传播。

        Args:
            queries (torch.Tensor): 查询向量，形状为 (batch_size, num_queries, d)。
            keys (torch.Tensor): 键向量，形状为 (batch_size, num_key_value_pairs, d)。
            values (torch.Tensor): 值向量，形状为 (batch_size, num_key_value_pairs, v)。
            valid_lens (torch.Tensor, optional): 每个序列的有效长度，形状为 (batch_size,) 或 (batch_size, num_queries)。 Defaults to None.

        Returns:
            torch.Tensor: 上下文向量，形状为 (batch_size, num_queries, v)。
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制。

    Args:
        key_size (int): 键向量的维度。
        query_size (int): 查询向量的维度。
        value_size (int): 值向量的维度。
        num_hiddens (int): 隐藏层单元数，也是每个注意力头的输出维度乘以头数。
        num_heads (int): 注意力头的数量。
        dropout (float): dropout概率。
        bias (bool, optional): 是否使用偏置。 Defaults to False.
        **kwargs: 其他关键字参数。
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        前向传播。

        Args:
            queries (torch.Tensor): 查询向量，形状为 (batch_size, num_queries, num_hiddens)。
            keys (torch.Tensor): 键向量，形状为 (batch_size, num_key_value_pairs, num_hiddens)。
            values (torch.Tensor): 值向量，形状为 (batch_size, num_key_value_pairs, num_hiddens)。
            valid_lens (torch.Tensor, optional): 每个序列的有效长度，形状为 (batch_size,) 或 (batch_size, num_queries)。 Defaults to None.

        Returns:
            torch.Tensor: 上下文向量，形状为 (batch_size, num_queries, num_hiddens)。
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0上，将第一项（标量或者矢量）复制num_heads次，
            # 然后复制第二项等等。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """
    将输入张量X的形状进行转换，以适应多头注意力的计算。

    Args:
        X (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, num_hiddens)。
        num_heads (int): 注意力头的数量。

    Returns:
        torch.Tensor: 转换后的张量，形状为 (batch_size * num_heads, seq_len, num_hiddens / num_heads)。
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """
    将多头注意力的输出张量X的形状进行转换，使其能够与后续层连接。

    Args:
        X (torch.Tensor): 输入张量，形状为 (batch_size * num_heads, seq_len, num_hiddens / num_heads)。
        num_heads (int): 注意力头的数量。

    Returns:
        torch.Tensor: 转换后的张量，形状为 (batch_size, seq_len, num_hiddens)。
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):
    """
    位置编码。

    Args:
        num_hiddens (int): 隐藏层单元数。
        dropout (float): dropout概率。
        max_len (int, optional): 序列的最大长度。 Defaults to 1000.
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个形状为(1, max_len, num_hiddens)的位置编码矩阵
        self.P = torch.zeros((1, max_len, num_hiddens))
        # 创建一个形状为(max_len, 1)的张量，表示位置索引
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        # 计算位置编码的缩放因子
        div_term = torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 计算位置编码的正弦和余弦值
        self.P[:, :, 0::2] = torch.sin(X / div_term)
        self.P[:, :, 1::2] = torch.cos(X / div_term)

    def forward(self, X):
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, num_hiddens)。

        Returns:
            torch.Tensor: 加上位置编码后的张量，形状为 (batch_size, seq_len, num_hiddens)。
        """
        X = X + self.P[:, :X.shape[1], :].to(X.device)

        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    """
    位置相关的全连接前馈网络。

    Args:
        ffn_num_input (int): 输入特征维度。
        ffn_num_hiddens (int): 隐藏层维度。
        ffn_num_outputs (int): 输出特征维度。
        **kwargs: 其他关键字参数。
    """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, ffn_num_input)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, ffn_num_outputs)。
        """
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """
    残差连接后进行层规范化。

    Args:
        normalized_shape (int or list or torch.Size): LayerNorm的输入尺寸。
        dropout (float): dropout概率。
        **kwargs: 其他关键字参数。
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, num_hiddens)。
            Y (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, num_hiddens)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, num_hiddens)。
        """
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    """
    Transformer编码器块。

    Args:
        key_size (int): 键向量的维度。
        query_size (int): 查询向量的维度。
        value_size (int): 值向量的维度。
        num_hiddens (int): 隐藏层单元数。
        norm_shape (int or list or torch.Size): LayerNorm的输入尺寸。
        ffn_num_input (int): 前馈网络的输入维度。
        ffn_num_hiddens (int): 前馈网络的隐藏层维度。
        num_heads (int): 注意力头的数量。
        dropout (float): dropout概率。
        use_bias (bool, optional): 是否使用偏置。 Defaults to False.
        **kwargs: 其他关键字参数。
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, num_hiddens)。
            valid_lens (torch.Tensor, optional): 每个序列的有效长度，形状为 (batch_size,) 或 (batch_size, num_queries)。 Defaults to None.

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, num_hiddens)。
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class Encoder(nn.Module):
    """
    编码器基类。用于定义编码器的基本接口。

    所有编码器都应继承此类并实现 `forward` 方法。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X, *args):
        """
        编码器前向传播方法。

        Args:
            X (torch.Tensor): 输入张量。
            *args: 其他参数。

        Raises:
            NotImplementedError: 如果子类未实现此方法，则抛出异常。
        """
        raise NotImplementedError

class TransformerEncoder(Encoder):
    """
    Transformer编码器。

    Args:
        vocab_size (int): 词汇表大小。
        key_size (int): 键向量的维度。
        query_size (int): 查询向量的维度。
        value_size (int): 值向量的维度。
        num_hiddens (int): 隐藏层单元数。
        norm_shape (int or list or torch.Size): LayerNorm的输入尺寸。
        ffn_num_input (int): 前馈网络的输入维度。
        ffn_num_hiddens (int): 前馈网络的隐藏层维度。
        num_heads (int): 注意力头的数量。
        num_layers (int): 编码器层数。
        dropout (float): dropout概率。
        use_bias (bool, optional): 是否使用偏置。 Defaults to False.
    """
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(
                "block"+str(i),
                TransformerEncoderBlock(
                    key_size, query_size, value_size, num_hiddens,
                    norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, dropout, use_bias))
        self.norm = nn.LayerNorm(norm_shape)

    def forward(self, X, valid_lens, *args):
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, seq_len)。
            valid_lens (torch.Tensor, optional): 每个序列的有效长度，形状为 (batch_size,) 或 (batch_size, num_queries)。 Defaults to None.

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, num_hiddens)。
        """
        # 因为位置编码值在-1和1之间，
        # 因此在嵌入之后乘以嵌入维度的平方根可以调整数值范围，
        # 从而产生更稳定的梯度。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embedding.embedding_dim))
        for block in self.blocks:
            X = block(X, valid_lens)
        return self.norm(X)

class TransformerDecoderBlock(nn.Module):
    """
    Transformer解码器块。

    Args:
        key_size (int): 键向量的维度。
        query_size (int): 查询向量的维度。
        value_size (int): 值向量的维度。
        num_hiddens (int): 隐藏层单元数。
        norm_shape (int or list or torch.Size): LayerNorm的输入尺寸。
        ffn_num_input (int): 前馈网络的输入维度。
        ffn_num_hiddens (int): 前馈网络的隐藏层维度。
        num_heads (int): 注意力头的数量。
        dropout (float): dropout概率。
        i (int): 当前解码器块的索引。
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, num_hiddens)。
            state (list): 包含编码器输出、编码器有效长度和历史键值对的列表。

        Returns:
            tuple: 包含输出张量（形状为 (batch_size, seq_len, num_hiddens)）和更新后的状态列表。
        """
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i, X]), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device
            ).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(X)), state

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def init_state(self, enc_outputs, *args):
        return NotImplementedError

    def forward(self, X, state):
        return NotImplementedError

class AttentionDecoder(Decoder):
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError

class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block{i}", TransformerDecoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, dropout, i
            ))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            return self.dense(X), state

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

if __name__ == '__main__':
    valid_lens = torch.tensor([3, 2])
    encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()

    encoder = TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape

    decoder_blk = TransformerDecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    X = torch.ones((2, 100, 24))
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    print(decoder_blk(X, state)[0].shape)




















