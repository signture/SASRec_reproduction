import copy
from math import ceil, floor
import numpy as np


class RandSeqPool(object):
    def __init__(self, aug_num, item_num):
        # 这个其实就传进来的参数暂时就不利用了，就直接使用测试的时候各个增强方式最好的参数
        self.pool = []
        self.pool.append(CropSeq(0.2, item_num))
        self.pool.append(MaskSeq(0.4, item_num))
        self.pool.append(ReplaceSeq(0.4, item_num))
        
    def __call__(self, orig_seq):
        aug = np.random.choice(self.pool)
        return aug(orig_seq)

class CropSeq(object):
    def __init__(self, prop, item_num):
        self.prop = prop
    
    def __call__(self, orig_seq):
        seq = copy.deepcopy(np.array(orig_seq))
        seq_len = len(seq) * self.prop
        assert seq_len > 0, 'CropSeq: seq_len must be greater than 0'
        begin = np.random.randint(0, ceil(len(seq) - seq_len)) # 随机选择一个片段的开始位置
        idx = np.arange(begin, ceil(begin + seq_len))
        return seq[idx].tolist()
    
    
class MaskSeq(object):
    def __init__(self, prob, item_num, hard=False):
        self.prob = prob
        self.hard = hard
        
    def __call__(self, orig_seq):
        if self.hard:
            return self.hard_mask(orig_seq)
        else:
            return self.soft_mask(orig_seq)
    
    def soft_mask(self, ori_seq):
        seq = copy.deepcopy(np.array(ori_seq))
        mask = ((np.random.rand(seq.size)) > self.prob)  # 对每个位置进行随机mask
        while mask.sum() < 1:
            mask = ((np.random.rand(seq.size)) > self.prob)
        seq[mask] = 0
        return seq.tolist()

    def hard_mask(self, ori_seq):
        seq = copy.deepcopy(np.array(ori_seq))
        seq_idx = np.random.choice(np.arange(0, len(seq)), size=floor(len(seq) * self.prob), replace=False)  # 固定采样mask长度
        seq[seq_idx] = 0
        return seq.tolist()
            
            
class ReplaceSeq(object):
    def __init__(self, prob, item_num):
        self.prob = prob
        self.item_list = np.arange(1, item_num + 1)
    
    def __call__(self, ori_seq):
        # 这里是软替换
        seq = copy.deepcopy(np.array(ori_seq))
        replace_mask = np.random.rand(len(seq)) < self.prob
        seq[replace_mask] = np.random.choice(self.item_list, size=replace_mask.sum())
        return seq.tolist()
    

        