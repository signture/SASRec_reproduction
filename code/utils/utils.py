import torch
import random
import numpy as np
from collections import defaultdict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    
def load_data(data_path:str, timestamp:bool) -> list:
    data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if timestamp:
                data.append([int(line[0]), int(line[1]), int(line[2])])  # userid, itemid, timestamp
            else:
                data.append([int(line[0]), int(line[1])])  # userid, itemid
    return data

def load_genre_data(genre_path:str, genre_num:int=18) -> dict:
    genre = {}
    init_vector = np.zeros(genre_num, dtype=np.float32)
    with open(genre_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            vector = init_vector.copy()
            np.put(vector, [int(x) - 1 for x in line[1:]], 1.0)
            genre[int(line[0])] = vector  # itemid, genreid
    return genre
    
def data_split(data_path: str, timestamp:bool=False):
    data = load_data(data_path, timestamp)
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    user_num = max([row[0] for row in data])
    item_num = max([row[1] for row in data])
    
    
    if timestamp:
        for user, item, timestamp in data: 
            User[user].append((timestamp, item))
        # 对每个用户的时间戳进行放缩处理
    else:
        for user, item in data:
            User[user].append(item)
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:  # 这里是为了保证每个用户至少有3个交互记录,但是其实这个数据集不用担心这个问题
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []  
        else:
            user_train[user] = User[user][:-2]  # 前n-2个作为训练集
            user_valid[user] = []  # 这里是为了保证每个用户至少有1个交互记录
            user_valid[user].append(User[user][-2])  # 倒数第二个作为验证集
            user_test[user] = []  # 这里是为了保证每个用户至少有1个交互记录
            user_test[user].append(User[user][-1])  # 倒数第一个作为测试集
    return [user_train, user_valid, user_test, user_num, item_num]


def getRelativePos(time_seq, time_span):
    time_matrix = np.abs(time_seq[:, np.newaxis] - time_seq[np.newaxis, :])
    # 这里需要防止全是0
    min_time_span = np.min(time_matrix[time_matrix > 0]) if np.any(time_matrix > 0) else 1.0
    # 对矩阵进行放缩处理(处于集合中的最小间隔并向下取整)
    time_matrix = np.floor(time_matrix / min_time_span)
    # 裁剪到[0, time_span]范围内
    time_matrix = np.clip(time_matrix, 0, time_span)
    return time_matrix.astype(np.int32)


def getRelations(user_set:dict, max_len, time_span):
    user_relations = {}
    for user in user_set:
        time_seq = np.array([item[0] for item in user_set[user]]) - user_set[user][0][0]
        if len(time_seq) < max_len:
            # 如果时间序列长度小于max_len，则进行填充
            time_seq = np.pad(time_seq, (max_len - len(time_seq), 0), 'constant', constant_values=0)
        else:
            # 如果时间序列长度大于max_len，则进行截断
            time_seq = time_seq[-max_len:]
        user_relations[user] = getRelativePos(time_seq, time_span)
    return user_relations

# 简单统计一下时间间隔集里面的一些信息
def getTimeSet(user_set:dict, max_len, time_span):
    time_span_set = set()
    for user in user_set:
        time_seq = np.array([item[0] for item in user_set[user]]) - user_set[user][0][0]
        diff_matrix = np.abs(time_seq[:, np.newaxis] - time_seq[np.newaxis, :])
        min_time_span = np.min(diff_matrix[diff_matrix > 0]) if np.any(diff_matrix > 0) else 1.0
        # 对矩阵进行放缩处理(处于集合中的最小间隔并向下取整)
        diff_matrix = np.floor(diff_matrix / min_time_span)
        diff_set = set(diff_matrix[np.triu_indices(len(time_seq), k=1)])
        time_span_set.update(diff_set)
    time_span_set = sorted(list(time_span_set))
    time_span_set = [int(x) for x in time_span_set]
    import matplotlib.pyplot as plt
    plt.hist(time_span_set, bins=100)
    plt.show()
    print(min(time_span_set), max(time_span_set), len(time_span_set))
    return time_span_set

def scaleTime(time_seq:list):
    time_seq = (np.array(time_seq) - time_seq[0]) / (time_seq[-1] - time_seq[0])
    return time_seq
    

if __name__ == "__main__":
    data_path = '../data/ml-1m/ratings_process_timestamp.txt'
    user_train, user_valid, user_test, user_num, item_num = data_split(data_path, timestamp=True)
    # selected_keys = [1, 2, 3, 4]
    # sub_dict = {key: user_train[key] for key in selected_keys if key in user_train}
    # user_relations = getRelations(sub_dict, 10, 10)
    time_span_set = scaleTime(user_train)