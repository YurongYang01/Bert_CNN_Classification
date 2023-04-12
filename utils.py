# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
#数据预处理
#构建数据集
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):    #遍历每一行
                lin = line.strip()  #去掉首位空白
                if not lin:         #遇到空白Skip
                    continue
                content, label = lin.split('\t')#数据集格式 Text'\t'Label
                token = config.tokenizer.tokenize(content)#分词
                token = [CLS] + token                           #头部加入[CLS]token
                seq_len = len(token)                            #填充或截断前实际长度
                mask = []                                       #区分填充、非填充部分
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:#长截短填
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))                #用0填充
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]                            #截断
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    #依次对train、dev、test进行处理
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test

#自定义数据迭代器
#训练时，不是把全部数据都加载到内存或显存中，而是用到哪一部分数据(某个batch)，就用数据生成器生成该部分数据，只把这部分数据加载到内存或显存中，避免溢出。
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size#得到batch数量
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:#不能整除
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        #转化为tensor并to（device）
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)#输入序列
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)#输入标签

        # seq_len为文本实际长度「转换为tensor并to（device）」
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        #MASK遮罩
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:#当数据集大小不整除batch_size
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1 #不整除 batch+1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, *value, end="\n"):

        current = time.strftime("[%Y-%m-%d %H:%M:%S]")
        s = current
        for v in value:
            s += " " + str(v)
        s += end
        print(s, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)
