import gzip
from collections import defaultdict
from datetime import datetime 


def load_raw_data(data_path:str) -> list:
    # 这个是加载的原始数据，不是处理后的数据
    data = []
    with open(data_path, 'rb') as f:
        data = f.read().split(b'\n')[:-1]
    return data     


def process_movie_data(data_path:str, mapping_path):
    # 电影data的字段之间使用::分割，分别是MovieID::Title::Genres
    # 这里的话其实主要使用MovieID字段和Genres字段就好，也是将信息重新保存一下
    # 预计保存为id g1 g2 g3 g4的形式，其中id是重映射过的id，g1也是重映射的电影类型
    # 读取映射关系
    movies_map = {}
    with open(mapping_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            movies_map[line[0]] = int(line[1])
    # 第一步读取数据
    data = []
    genres_map = {}
    with open(data_path, 'rb') as f:
        for line in f.readlines():
            genres = []
            line = line.strip().split(b'::')
            for genre in line[2].split(b'|'):  # 这里是将电影类型进行分割
                mtype = str(genre, encoding='utf-8')
                if mtype not in genres_map:
                    genres_map[mtype] = len(genres_map) + 1
                genres.append(genres_map[mtype])
            map_id = str(line[0], encoding='utf-8')
            if map_id in movies_map:
                data.append([movies_map[str(line[0], encoding='utf-8')], genres])
    with open('movies_process.txt', "w") as f:
        for line in data:
            f.write(str(line[0]) + " " + " ".join([str(x) for x in line[1]]) + "\n")
    print(len(genres_map))
    with open('genres_map.txt', "w") as f:
        for key, value in genres_map.items():
            f.write(key + " " + str(value) + "\n")
    print(len(data))
    print(len(genres_map))
    return data

def process_data(raw_data:list, save_path:str, timestamp:bool=False):
    # rating数据的格式是UserID::MovieID::Rating::Timestamp
    # 项目里面似乎没有对rating进行处理，需要查一下文献看一下方案，但是个人理解来说rating就是一个交互反馈
    # 大致处理一般是首先将用户交互比较少的和产品交互比较少的给筛出
    # 找到了对这个数据集的处理方法，来源于仓库作者pull的issue：
    # As you can see, I borrowed the pre-processed datasets from the paper authors' repo, 
    # you can generate your own datasets just by groupby users and then sort by timestamps, 
    # lastly drop other columns except these two columns for generating (user_id, item_id) pairs used in model training/validation/testing.
    # FYI to those who care about data pre-processing, 
    # I just noticed paper authors claimed to remove users and items with less than 3 interactions, 
    # so for ml1m dataset, the pre-processed dataset has bit less rows than original ml1m rating file. 
    # It's reasonable to remove these users/items, otherwise we can not generate training/validation(2nd last interacted item for the user)/testing(last interacted item for the user) data.
    data = []
    new_data = []
    usermap = dict()
    itemmap = dict()
    user_count = 0
    item_count = 0
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    User = dict()
    # 首先解析成字典形式
    for line in raw_data:
        line = line.split(b'::')
        data.append([int(line[0]), int(line[1]), int(line[3])])  # userid, movieid, timestamp
        countU[int(line[0])] += 1  # 统计每个用户的交互次数
        countP[int(line[1])] += 1  # 统计每个产品的交互次数
       
    # 对时间戳进行排序
    data.sort(key=lambda x: x[2])    
        
    # 筛选出交互次数大于等于3的用户和产品
    for line in data:
        userid = line[0]
        movieid = line[1]
        timestamp = line[2]
        if countU[userid] < 5 or countP[movieid] < 5:  # 实际上似乎并不用筛用户，因为数据里用户都有20个点评以上 
            continue
        if userid in usermap:
            uid = usermap[userid]
        else:  # 筛选交互高的用户，并进行用户重排序
            user_count += 1
            usermap[userid] = user_count  # 建立数据id和重新排序的id之间的映射关系
            uid = user_count
            User[uid] = []  # 这个就是重排序的id的用户的交互记录
        if movieid in itemmap:
            itemid = itemmap[movieid]
        else:  # 筛选交互高的产品，并进行产品重排序
            item_count += 1
            itemid = item_count
            itemmap[movieid] = itemid  # 这里是产品id和重新排序的id之间的映射关系
        User[uid].append((timestamp, itemid))  # 存储用户的交互记录，包括时间戳和产品ID
        new_data.append([uid, itemid])
    # 对每个用户的交互记录按照时间戳进行排序
    for userid in User:
        User[userid].sort(key=lambda x: x[0])  # 按照时间戳排序
        
    # 记录逆映射方式
    with open(save_path + 'user2new.txt', 'w') as f:
        for userid in usermap:
            f.write('%d %d\n' % (userid, usermap[userid]))  # 左边是userid，右边是新映射的id

    with open(save_path + 'item2new.txt', 'w') as f:
        for movieid in itemmap:
            f.write('%d %d\n' % (movieid, itemmap[movieid]))  # 左边是movieid，右边是新映射的id
        
    # 记录
    save_path = save_path + 'ratings_process.txt'
    if timestamp:
        save_path = save_path.replace('.txt', '_timestamp.txt')  # 如果有时间戳的话，文件名后面加上_timestamp
    with open(save_path, 'w') as f:
        for user in User:
            for i in User[user]:
                if timestamp:
                    f.write('%d %d %d\n' % (user, i[1], i[0]))  # 写入用户ID、产品ID和时间戳，每行一个交互记录
                else:
                    f.write('%d %d\n' % (user, i[1]))  # 写入用户ID和产品ID，每行一个交互记录
    print('Data preprocess finished!')
    return new_data
        
        
if __name__ == "__main__":
    data_path = '../data/ml-1m/ratings.dat'
    check_data = '../data/ml-1m/ml-1m.txt'
    save_path = '../data/ml-1m/'
    real_data = []
    with open(check_data, 'r') as f:
        real_data = f.read().split('\n')[:-1]
        real_data = [row.split(' ') for row in real_data]
        real_data = [[int(row[0]), int(row[1])] for row in real_data]
    print(len(real_data))
    print(max([row[0] for row in real_data]))
    print(max([row[1] for row in real_data]))
    raw_data = load_raw_data(data_path)
    new_data = process_data(raw_data, save_path, timestamp=True)
    print(len(new_data))
    print(max([row[0] for row in new_data]))
    print(max([row[1] for row in new_data]))
    print(1)
    # data_path = '../data/ml-1m/movies.dat'
    # mapping_path = '../data/ml-1m/item2new.txt'
    # movie_data = process_movie_data(data_path, mapping_path)