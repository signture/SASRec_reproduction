import torch
import random
from collections import defaultdict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    
def load_data(data_path:str) -> list:
    data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            data.append([int(line[0]), int(line[1])])  # userid, itemid
    return data
    
def data_split(data_path: str):
    data = load_data(data_path)
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    user_num = max([row[0] for row in data])
    item_num = max([row[1] for row in data])
    
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