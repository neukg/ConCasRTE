#! -*- coding:utf-8 -*-
import numpy as np
import random
from copy import deepcopy
import os
import pickle

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def mat_padding(inputs,dim=0, length=None, padding=0):
    """Numpy函数，将序列的dim维padding到同一长度
    """
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[dim])
        pad_width[1] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def tuple_mat_padding(inputs,dim=1, length=None, padding=0):
    """Numpy函数，将序列的dim维padding到同一长度
    """
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[1] = (0, length - x.shape[dim])
        pad_width[2] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def sequence_padding(inputs,dim=0, length=None, padding=0):
    """Numpy函数，将序列的dim维padding到同一长度
    """
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)


def data_augmentation(example,ex2):
    '''数据增强，返回新的样例'''
    same_example=deepcopy(example)
    try:
        a=random.randint(0,6)
        if a==0:#多个句子随机拼接
            text1,text2=example["text"],ex2["text"]
            tokens1=text1.split()
            tokens2=text2.split()
            loc=random.randint(0,len(tokens1))
            tokens=tokens1[:loc]+tokens2+tokens2[loc:]
            spo_list=[]
            text=" ".join(tokens)
            for s,p,o in example['triple_list']+ex2["triple_list"]:
                if s in text and o in text:
                    spo_list.append([s,p,o])
            res={"text":text,"triple_list":spo_list}
        elif a==1:#随机插入单词
            text=example["text"]
            all_tokens=text.split()
            num_token=len(all_tokens)//10+1 #每10个单词随机插入一个词语
            for i in range(num_token):
                token=random.choice(all_tokens) #将要插入的单词
                loc = random.randint(0, len(all_tokens)) #将要插入的位置
                all_tokens.insert(loc,token)
            text=" ".join(all_tokens)
            spo_list=[]
            for s,p,o in example["triple_list"]:
                if s in text and o in text:
                    spo_list.append([s,p,o])
                else:
                    return same_example #保证原本的三元组不变
            res={"text":text,"triple_list":spo_list}
        elif a==2:#随机减少单词
            text=example["text"]
            all_tokens=text.split()
            num_token=len(text)//10+1 #每10个单词随机减少一个词语
            for i in range(num_token):
                loc = random.randint(0, len(all_tokens)-1) #将要删除的位置
                all_tokens.pop(loc)
            text=" ".join(all_tokens)
            spo_list=[]
            for s,p,o in example["triple_list"]:
                if s in text and o in text:
                    spo_list.append([s,p,o])
                else:
                    return same_example #保证原本的三元组不变
            res={"text":text,"triple_list":spo_list}
        else: #不进行数据增强
            res=example
        if len(res["triple_list"])==0: #防止没有标注的情况
            res=same_example
        return res
    except:
        return same_example

def judge(ex):
    '''判断样本是否正确'''
    for s,p,o in ex["triple_list"]:
        if s=='' or o=='' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True


class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random: #乱序
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i] #返回样本编号

            data = generator()
        else: #正序
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        for d in self.__iter__(True):
            yield d


class Vocab(object):
    def __init__(self, filename, load=False, word_counter=None, threshold=0):
        if load:
            assert os.path.exists(filename), "Vocab file does not exist at " + filename
            # load from file and ignore all other params
            self.id2word, self.word2id = self.load(filename)
            self.size = len(self.id2word)
            print("Vocab size {} loaded from file".format(self.size))
        else:
            print("Creating vocab from scratch...")
            assert word_counter is not None, "word_counter is not provided for vocab creation."
            self.word_counter = word_counter
            if threshold > 1:
                # remove words that occur less than thres
                self.word_counter = dict([(k, v) for k, v in self.word_counter.items() if v >= threshold])
            self.id2word = sorted(self.word_counter, key=lambda k: self.word_counter[k], reverse=True)
            # add special tokens to the beginning
            self.id2word = ['**PAD**', '**UNK**'] + self.id2word
            self.word2id = dict([(self.id2word[idx], idx) for idx in range(len(self.id2word))])
            self.size = len(self.id2word)
            self.save(filename)
            print("Vocab size {} saved to file {}".format(self.size, filename))

    def load(self, filename):
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])
        return id2word, word2id

    def save(self, filename):
        # assert not os.path.exists(filename), "Cannot save vocab: file exists at " + filename
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(self.id2word, outfile)
        return

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.word2id[w] if w in self.word2id else constant.VOCAB_UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]

    def get_embeddings(self, word_vectors=None, dim=100):
        # self.embeddings = 2 * constant.EMB_INIT_RANGE * np.random.rand(self.size, dim) - constant.EMB_INIT_RANGE
        self.embeddings = np.zeros((self.size, dim))
        if word_vectors is not None:
            assert len(list(word_vectors.values())[0]) == dim, \
                "Word vectors does not have required dimension {}.".format(dim)
            for w, idx in self.word2id.items():
                if w in word_vectors:
                    self.embeddings[idx] = np.asarray(word_vectors[w])
        return self.embeddings
