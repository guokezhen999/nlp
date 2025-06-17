import os
import torch
import collections
from torch.utils import data

def tokenize(lines, token='word'):
    """ 词元化 """
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def count_corups(tokens):
    """ 统计词的出现频率 """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    """ 按照出现次数降序生成词典 """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按照出现频率排序
        counter = count_corups(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词的索引为0
        self.idx_to_token = ['unk'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            # 去除低于指定次数的token
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 查询的是单个token
        if not isinstance(tokens, (list, tuple)):
            # 键不存在，返回unk
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def get_freqs(self, token):
        for pair in self._token_freqs:
            if pair[0] == token:
                return pair

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

class Labels:
    """ 将标签由字符串转化为数字 """
    def __init__(self, labels):
        counter = count_corups(labels)
        self._label_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_label = []
        self.label_to_idx = {}
        for label, freq in self._label_freqs:
            if label not in self.label_to_idx:
                self.idx_to_label.append(label)
                self.label_to_idx[label] = len(self.idx_to_label) - 1

    def __len__(self):
        return len(self.idx_to_label)

    def __getitem__(self, labels):
        if not isinstance(labels, (list, tuple)):
            return self.label_to_idx.get(labels, self.unk)
        return [self.__getitem__(label) for label in labels]

    def to_labels(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_label[indices]
        return [self.idx_to_label[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return -1

    @property
    def label_freqs(self):
        return self._label_freqs

class TokenEmbedding:
    """ 预处理嵌入模型加载 """
    def __init__(self, embedding_name):
        self.unknown_idx = 0
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.token_to_idx = {token : idx
                             for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = os.path.join(os.path.dirname(__file__), 'data', embedding_name)
        with open(os.path.join(data_dir, 'vec.txt')) as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

def load_array(data_arrays, batch_size, is_train=True):
    """ 构造torch数据迭代器 """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def truncate_pad(line, num_steps, padding_token):
    """ 截断或填充文本序列至制定长度 """
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

