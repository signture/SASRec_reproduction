import torch
import torch.nn
from torch.utils import data
import numpy as np
from multiprocessing import Process, Queue
from data.augmentation import *
from utils.utils import *

augmentations = {
    'mask': MaskSeq,
    'crop': CropSeq,
    'replace': ReplaceSeq,
    'random': RandSeqPool,
    'none': None
}

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, augmentation=None, genre_dict=None, timestamp=False):
    all_item_ids = set(range(1, itemnum + 1))
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        if timestamp:
            item_ids = [i[1] for i in user_train[uid]]
            time_seq = scaleTime([i[0] for i in user_train[uid]])
        else:    
            item_ids = user_train[uid]
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        aug1 = np.zeros([maxlen], dtype=np.int32)
        aug2 = np.zeros([maxlen], dtype=np.int32)
        # nxt = user_train[uid][-1]
        # idx = maxlen - 1
        valid_length = min(len(item_ids) - 1, maxlen)
        
        # 赋值seq
        seq[-valid_length:] = np.array(item_ids[:-1])[-valid_length:]
        pos[-valid_length:] = np.array(item_ids[1:])[-valid_length:]
        if timestamp:
            tseq = np.zeros([maxlen], dtype=np.float32)
            tseq[-valid_length:] = np.array(time_seq[:-1])[-valid_length:]

        ts = set(user_train[uid])
        non_interacted_items = np.array(list(all_item_ids - ts))  # 未交互物品的 ID 数组
        if len(non_interacted_items) > 0:
            neg_samples = np.random.choice(non_interacted_items, size=valid_length, replace=True)
            neg[-valid_length:] = neg_samples
        # for i in reversed(user_train[uid][:-1]):
        #     seq[idx] = i
        #     pos[idx] = nxt
        #     if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
        #     nxt = i
        #     idx -= 1
        #     if idx == -1: break
        return_tuple = (uid, seq, pos, neg)
        # 这里暂时只考虑原先的情况进行添加
        if timestamp:
            return_tuple = (uid, seq, pos, neg, tseq)
        if genre_dict is not None:
            pos_genres = np.array([genre_dict.get(i, np.zeros_like(genre_dict[1])) for i in pos])
            neg_genres = np.array([genre_dict.get(i, np.zeros_like(genre_dict[1])) for i in neg])
            seq_genres = np.array([genre_dict.get(i, np.zeros_like(genre_dict[1])) for i in seq])
            return_tuple = (uid, (seq, seq_genres), (pos, pos_genres), (neg, neg_genres))
            
        if augmentation is not None:
            aug1 = np.zeros([maxlen], dtype=np.int32)
            aug2 = np.zeros([maxlen], dtype=np.int32)
        
            aug_seq1 = augmentation(item_ids[:-1])
            aug_seq2 = augmentation(item_ids[:-1])
            valid_length = min(len(aug_seq1), maxlen)
            aug1[-valid_length:] = np.array(aug_seq1)[-valid_length:]
            valid_length = min(len(aug_seq2), maxlen)
            aug2[-valid_length:] = np.array(aug_seq2)[-valid_length:]
            
            if genre_dict is not None:
                aug1_genres = np.array([genre_dict.get(i, np.zeros_like(genre_dict[1])) for i in aug1])
                aug2_genres = np.array([genre_dict.get(i, np.zeros_like(genre_dict[1])) for i in aug2])
                return_tuple = (uid, (seq, seq_genres), (pos, pos_genres), (neg, neg_genres), (aug1, aug1_genres), (aug2, aug2_genres))
                
            else:    
                return_tuple = (uid, seq, pos, neg, aug1, aug2) 

        return return_tuple
    
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, augmentation=[None, 1], genre_dict=None, timestamp=False):
        assert augmentation[0] in ['mask', 'crop', 'replace', 'random', 'none'], 'Invalid augmentation type'
        assert augmentation[1] >= 0 and augmentation[1] <= 1, 'Invalid augmentation probability'
        if augmentations[augmentation[0]] is not None:
            self.augmentation = augmentations[augmentation[0]](augmentation[1], itemnum) 
        else:
            self.augmentation = None
        self.genre_dict = genre_dict
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      self.augmentation, 
                                                      self.genre_dict,
                                                      timestamp
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


class SeqItemDataset(data.Dataset):
    def __init__(self, u2i:dict, user_num:int, item_num:int, max_len:int=200):
        super(SeqItemDataset, self).__init__()
        self.u2i = u2i
        self.user_num = user_num
        self.item_num = item_num
        self.max_len = max_len
        
    def __getitem__(self, index):
        user_id = index + 1  # 这里的index是从0开始的，所以要加1
        item_ids = self.u2i[user_id]
    
        # 构造训练样本（seq, pos, neg）
        seq = np.zeros(self.max_len, dtype=np.int32)
        pos = np.zeros(self.max_len, dtype=np.int32)
        neg = np.zeros(self.max_len, dtype=np.int32)
        
        nxt = item_ids[-1]
        valid_idx = self.max_len - 1
        
        reversed_items = list(reversed(item_ids[:-1]))
        seq_len = min(len(reversed_items), self.max_len)
        
        seq[-seq_len:] = reversed_items[:seq_len]
        pos[-seq_len:] = [nxt] + [reversed_items[i] for i in range(seq_len-1)]
            
        if hasattr(self, 'neg_pools'):
            neg_samples = np.random.choice(self.neg_pools[user_id], size=seq_len, replace=True)
        else:
            seen = set(item_ids)
            neg_samples = [self._fast_neg_sample(seen) for _ in range(seq_len)]
            
        neg[-seq_len:] = neg_samples
        
        return torch.Tensor([user_id]), torch.Tensor(seq), torch.Tensor(pos), torch.Tensor(neg)

    def _fast_neg_sample(self, seen):
        t = np.random.randint(1, self.item_num + 1)  # 这里numpy和random的生成不一样，numpy是左闭右开的，random是左闭右闭的
        while t in seen:
            t = np.random.randint(1, self.item_num + 1)
        return t
        
    def __len__(self):
        return len(self.u2i)
    

if __name__ == "__main__":
    u2i = {1: [1, 2, 3, 4, 5, 8, 10], 2: [4, 5, 6, 7]}  # userid: [itemid1, itemid2, ...]  # 这里的itemid是物品id
    user_num = 2  # 用户数量
    item_num = 12  # 物品数量
    # max_len = 10  # 序列最大长度
    # dataset = SeqItemDataset(u2i, user_num, item_num, max_len)  # 这里的u2i是用户id到物品id的映射，user_num是用户数量，item_num是物品数量，max_len是序列最大长度        
    # dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)  # 这里的batch_size是批次大小，shuffle是是否打乱顺序
    # for user_id, seq, pos, neg in dataloader:  # 这里的user_id是用户id，seq是序列，pos是正样本，neg是负样本
    #     print(user_id)  # [batch_size]  # 这里的user_id是用户id
    #     print(seq)  # [batch_size, max_len]  # 这里的seq是序列
    #     print(pos)  # [batch_size, max_len]  # 这里的pos是正样本
    #     print(neg)  # [batch_size, max_len]  # 这里的neg是负样本
    #     break
    genre_dict = {1 : [1, 3], 2 : [3, 5], 3: [1, 3], 4: [2, 5], 5: [1, 2], 6: [3, 5], 7: [1, 2], 8: [3, 5], 9: [1, 2], 10: [3, 5]}
    init_vector = np.zeros(18, dtype=np.float32)
    for item in genre_dict:
        vector = init_vector.copy()
        np.put(vector, [int(x) - 1 for x in genre_dict[item]], 1.0)
        genre_dict[item] = vector
    sampler = WarpSampler(u2i, user_num, item_num, batch_size=2, maxlen=10, n_workers=1, augmentation=['crop', 0.5], genre_dict=genre_dict)  # 这里的batch_size是批次大小，maxlen是序列最大长度，n_workers是进程数量
    user_id, seq, pos, neg, aug1, aug2 = sampler.next_batch() # 这里的user_id是用户id，seq是序列，pos是正样本，neg是负样本
    print(user_id)  # [batch_size]  # 这里的user_id是用户id
    print(seq)  # [batch_size, max_len]  # 这里的seq是序列
    print(pos)  # [batch_size, max_len]  # 这里的pos是正样本
    print(neg)  # [batch_size, max_len]  # 这里的neg是负样本
    print(aug1)
    print(aug2)