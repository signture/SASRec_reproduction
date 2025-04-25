# this file is use to check if the preprocess.py is correct
import hashlib
from collections import defaultdict

def read_sequences(file_path):
    # 读取文件并按用户分组，保留物品顺序
    sequences = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            user_id, item_id = line.strip().split()
            sequences[int(user_id)].append(item_id)
    return sequences

if __name__ == "__main__":
    check_data_path = '../data/ml-1m/ml-1m.txt'
    process_data_path = '../data/ml-1m/ratings_process.txt'
    # 读取原始数据和预处理后的数据
    official_sequences = read_sequences(check_data_path)
    processed_sequences = read_sequences(process_data_path)
    
    processed_item_counts = defaultdict(int)
    official_item_counts = defaultdict(int)
    
    # 计算物品出现次数
    for seq in processed_sequences.values():
        for item in seq:
            processed_item_counts[item] += 1
    for seq in official_sequences.values():
        for item in seq:
            official_item_counts[item] += 1
            
    # 处理后的数据
    processed_users = len(processed_sequences)
    processed_items = len(processed_item_counts)
    processed_interactions = sum(len(seq) for seq in processed_sequences.values())

    # 官方数据
    official_users = len(official_sequences)
    official_items = len(official_item_counts)
    official_interactions = sum(len(seq) for seq in official_sequences.values())

    assert (processed_users, processed_items, processed_interactions) == \
           (official_users, official_items, official_interactions), "基础统计量不一致"
    
    # 提取用户交互次数并排序
    processed_user_counts = sorted([len(seq) for seq in processed_sequences.values()])
    official_user_counts = sorted([len(seq) for seq in official_sequences.values()])

    assert processed_user_counts == official_user_counts, "用户交互次数分布不一致"
    
    # 提取物品出现次数并排序
    processed_item_counts = sorted(processed_item_counts.values())
    official_item_counts = sorted(official_item_counts.values())

    assert processed_item_counts == official_item_counts, "物品流行度分布不一致"
    
    import networkx as nx

    def build_bipartite_graph(sequences):
        G = nx.Graph()
        for user_id, items in sequences.items():
            for item in items:
                G.add_edge(f"u_{user_id}", f"i_{item}")
        return G

    # 构建图并对比度数分布
    G_processed = build_bipartite_graph(processed_sequences)
    G_official = build_bipartite_graph(official_sequences)

    # 对比用户和物品的度数分布
    def get_degree_distribution(graph):
        degrees = [d for _, d in graph.degree()]
        return sorted(degrees, reverse=True)

    assert get_degree_distribution(G_processed) == get_degree_distribution(G_official), "二部图结构不一致"
    
    print("所有断言通过，数据预处理暂无发现异常。")

    
    