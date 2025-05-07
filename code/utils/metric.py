import os
import sys
sys.path.append('./')
import copy
import torch
import random
import numpy as np
from model.SASRec import SASRec

def evaluate(model, dataset, valid=True, maxlen=200, genre_dict=None, timestamp=False):
    model.eval()
    [train, valid, test, user_num, item_num] = copy.deepcopy(dataset)
    if timestamp:
        train_time_seq = {k: [i[0] for i in v] for k, v in train.items()}
        valid_time_seq = {k: [i[0] for i in v] for k, v in valid.items()}
        test_time_seq = {k: [i[0] for i in v] for k, v in test.items()}
        train = {k: [i[1] for i in v] for k, v in train.items()}
        valid = {k: [i[1] for i in v] for k, v in valid.items()}
        test = {k: [i[1] for i in v] for k, v in test.items()}
    target_data = valid if valid else test
    
    users = range(1, user_num + 1)
    valid_users = 0
    ht = 0.0
    ndcg = 0.0
    
    with torch.no_grad():
        for u in users:
            
            if len(train[u]) < 1 or len(target_data[u]) < 1:
                continue  # 排除没有训练数据或测试数据的用户
            
            # # 构造输入序列
            # seq = np.zeros((1, maxlen), dtype=np.int32)
            # # 新增：构造时间序列
            time_seq = np.zeros((1, maxlen), dtype=np.float32) if timestamp else None

            # train_seq = np.array(train[u][-maxlen + (1 if not valid else 0):])[::-1]
            # start_idx = maxlen - len(train_seq) - (1 if not valid else 0)

            # if not valid:
            #     seq[0, -1] = valid[u][0]
            #     if timestamp:
            #         time_seq[0, -1] = valid_time_seq[u][0]

            # seq[0, start_idx:-1 if not valid else None] = train_seq
            # if timestamp:
            #     train_time = np.array(train_time_seq[u][-maxlen + (1 if not valid else 0):])[::-1]
            #     time_seq[0, start_idx:-1 if not valid else None] = (train_time - train_time[0]) / (train_time[-1] - train_time[0] + 1e-6)  # 归一化时间序列
        
            # 构造输入序列
            seq = np.zeros((1, maxlen), dtype=np.int32)
            idx = maxlen - 1  # 从序列的末尾开始填充
            if not valid:  # 测试的话就要将在验证里的倒数第二个填充进去
                seq[0][idx] = valid[u][0]  # 填充最后一个物品
                if timestamp:
                    time_seq[0][idx] = valid_time_seq[u][0]
                idx -= 1
            for i in reversed(range(len(train[u]))):
                seq[0][idx] = train[u][i]
                if timestamp:
                    time_seq[0][idx] = train_time_seq[u][i]
                idx -= 1
                if idx == -1:
                    break  # 如果序列已经满了，就不再填充
            
            # 构建物品候选
            rated = set(train[u])
            rated.add(0)  # 0是填充项，需要加入候选集
            item_idx = [target_data[u][0]]
            for _ in range(100):
                t = np.random.randint(1, item_num + 1)  # 这里numpy和random的生成不一样，numpy是左闭右开的，random是左闭右闭的
                while t in rated:
                    t = np.random.randint(1, item_num + 1)        
                item_idx.append(t)  # 这里是将还未交互的物品加入候选集
                
            if genre_dict is not None:  # 如果有genre_dict，就将genre加入到输入序列中
                seq_genres = np.array([genre_dict.get(i, np.zeros_like(genre_dict[1])) for i in seq[0]])
                item_genres = np.array([genre_dict.get(i, np.zeros_like(genre_dict[1])) for i in item_idx])
                seq = list(zip(seq, seq_genres))
                item_idx = list(zip(item_idx, item_genres))
            
            if timestamp:
                time_seq[0][idx+1:] = (time_seq[0][idx+1:] - time_seq[0][idx+1]) / (time_seq[0][-1] - time_seq[0][idx+1] + 1e-6)  # 归一化时间序列
                predictions = -model.predict((seq), (item_idx), time_seq=time_seq)
            else:
                predictions = -model.predict((seq), (item_idx))
            predictions = predictions[0]
        
            rank = predictions.argsort().argsort()[0].item()  # 计算排名(降序排序)

            valid_users += 1

            if rank < 10:  # 前10名
                ht += 1
                ndcg += 1 / np.log2(rank + 2)  # 计算ndcg

    return ht / valid_users, ndcg / valid_users


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRec(10, 10, 2, 2, device)
    model.to(device)
    train = {
        1: [1, 2, 3, 4, 5],
        2: [1, 2],
        3: [1, 2, 3],
    }
    valid = {
        1: [6],
        2: [3],
        3: [4], 
    }
    test = {
        1: [7],
        2: [4],
        3: [5], 
    }
    user_num = 3
    item_num = 10
    ht, ndcg = evaluate(model, [train, valid, test], user_num, item_num, valid=True)
    print(ht, ndcg)