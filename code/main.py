import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import logging
import argparse
from data.dataset import SeqItemDataset, WarpSampler
from model.BSARec import BSARec
from model.SASRec import SASRec
from utils.metric import evaluate
from utils.utils import data_split, set_seed, load_genre_data
from utils.train import train_epoch, EarlyStopping


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='none', choices=['mask', 'crop', 'replace', 'random', 'none'])
    parser.add_argument('--prob', type=float, default=0.8)
    parser.add_argument('--weight', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--view_type', type=str, default='mean', choices=['flatten', 'mean'])
    parser.add_argument('--timestamp', action="store_true", help='whether to use timestamp information')
    parser.add_argument('--genre', action="store_true", help='whether to use genre information')
    parser.add_argument('--model', type=str, default='SASRec', choices=['SASRec', 'BSARec'])
    parser.add_argument('--c', type=float, default=9)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    # 训练超参数
    max_seq_len = 200
    batch_size = 128
    learning_rate = 0.001 # 0.001
    epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_only = False
    patience = 10
    num_workers = 4
    seed = 14531
    
    # 可改动的超参数
    args = get_args()
    contrastive_type = args.type
    contrastive_prob = args.prob
    contrastive_weight = args.weight
    sim_temp = args.temp
    view_type = args.view_type
    timestamp = args.timestamp
    genre = args.genre
    model_name = args.model
    c = args.c
    # contrastive_type = 'mask'
    # contrastive_prob = 0.8
    # contrastive_weight = 0.1
    # sim_temp = 1.0  # 0.1
    
    # 模型超参数
    hidden_dim = 50
    num_heads = 1
    dropout_rate = 0.2
    num_blocks = 2
    l2_emb = 0.0
    
    # 相关路径设置
    if timestamp:
        data_path = '../data/ml-1m/ratings_process_timestamp.txt'
    else:
        data_path = '../data/ml-1m/ratings_process.txt'
    state_dict_path = '../model/SASRec.pth'
    genre_path = '../data/ml-1m/movies_process.txt'
    if os.path.exists(state_dict_path) == False:  # 如果不存在这个文件夹，则创建这个文件夹
        state_dict_path = None
    save_dir = '../result' + '/' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    ckpt_dir = save_dir + '/checkpoint'
    if os.path.exists(ckpt_dir) == False:  # 如果不存在这个文件夹，则创建这个文件夹
        os.makedirs(ckpt_dir)
        print('create dir: {}'.format(ckpt_dir))
    if os.path.exists(save_dir) == False:  # 如果不存在这个文件夹，则创建这个文件夹
        os.makedirs(save_dir)
        print('create dir: {}'.format(save_dir))
    log = open(save_dir + '/log.txt', 'w')  # 日志文件
    setting = open(save_dir + '/setting.txt', 'w')  # 日志文件
    
    # 写入模型超参数
    setting.write('hidden_dim: {}\n'.format(hidden_dim))
    setting.write('num_heads: {}\n'.format(num_heads))
    setting.write('dropout_rate: {}\n'.format(dropout_rate))
    setting.write('num_blocks: {}\n'.format(num_blocks))
    setting.write('l2_emb: {}\n'.format(l2_emb))
    setting.write('learning_rate: {}\n'.format(learning_rate))
    setting.write('batch_size: {}\n'.format(batch_size))
    setting.write('epochs: {}\n'.format(epochs))
    setting.write('patience: {}\n'.format(patience))
    setting.write('num_workers: {}\n'.format(num_workers))
    setting.write('seed: {}\n'.format(seed))
    setting.write('device: {}\n'.format(device))   
    setting.write('contrastive_type: {}\n'.format(contrastive_type))
    setting.write('contrastive_prob: {}\n'.format(contrastive_prob))
    setting.write('contrastive_weight: {}\n'.format(contrastive_weight))
    setting.write('contrastive_view_type: {}\n'.format(view_type))
    setting.write('sim_temp: {}\n'.format(sim_temp))
    setting.write('genre: {}\n'.format(genre))
    setting.write('timestamp: {}\n'.format(timestamp))
    setting.write('model_name: {}\n'.format(model_name))
    setting.write('c: {}\n'.format(c))
    
    setting.close()
    
    # 设置随机中种子
    set_seed(seed)
    
    # 加载数据并划分
    [user_train, user_valid, user_test, user_num, item_num] = data_split(data_path, timestamp=timestamp)
    genre_dict = load_genre_data(genre_path) if genre else None  # 加载电影类型数据
    num_batch = (len(user_train) - 1) // batch_size + 1  # 计算batch数量
    print('the number of batch for a epoch is %.2f' % num_batch)
    asl = 0.0  # 计算平均序列长度
    for user in user_train:
        asl += len(user_train[user])
    print('the average sequence length is %.2f' % (asl / len(user_train)))
    sampler = WarpSampler(user_train, user_num, item_num, batch_size=batch_size, maxlen=max_seq_len, \
                          n_workers=num_workers, augmentation=[contrastive_type, contrastive_prob], genre_dict=genre_dict, \
                          timestamp=timestamp)  # 采样器
    # dataset = SeqItemDataset(user_train, user_num, item_num, max_seq_len)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # 模型加载与训练准备
    model = eval(model_name)(item_num, hidden_dim, num_heads, num_blocks, device, l2_emb, dropout_rate, max_seq_len, temp=sim_temp, c=c)  # 模型加载
    # model = SASRec(item_num, hidden_dim, num_heads, num_blocks, device, l2_emb, dropout_rate, max_seq_len, temp=sim_temp)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()  # 交叉熵损失函数
    # early_stopping = EarlyStopping(save_dir + '/checkpoint', log, patience=patience, verbose=True)  # 早停策略
    model.to(device)  # 将模型加载到设备上
    print('the number of parameters is %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))  # 计算模型参数数量
    
    # 将模型参数写入日志文件
    log.write('the number of parameters is %.2fM\n' % (sum(p.numel() for p in model.parameters()) / 1e6))
    log.write('the number of batch for a epoch is %.2f\n' % num_batch)
    log.write('the average sequence length is %.2f\n' % (asl / len(user_train)))
    
    # 加载预训练模型(可选)
    if state_dict_path != None:
        model.load_state_dict(torch.load(state_dict_path))  # 加载预训练模型
        print('load model from {}'.format(state_dict_path))
    
    if inference_only == True:  # 如果不是只进行推理，则进行训练
        t_test = evaluate(model, [user_test, user_valid, user_test, user_num, item_num], False, max_seq_len)  # 计算测试集上的指标
        print('test (HT@10: %.4f, NDCG@10: %.4f)' % (t_test[0], t_test[1]))
    
    best_val_ndcg, best_val_ht = 0.0, 0.0
    best_test_ndcg, best_test_ht = 0.0, 0.0
    
    start_time = time.time()
    counter = 0
    for epoch in range(1, epochs + 1):  # 训练
        print('Epoch: %d' % epoch)
        log.write('Epoch: %d\n' % epoch)
        if genre:
            weight = model.get_weight()
            log.write('weight: %s\n' % str(weight))
        train_epoch(model, optimizer, criterion, num_batch, sampler, device, log, args)  # 训练一个epoch
        if epoch % patience == 0:
            t1 = time.time() - start_time
            print('Evaluate')
            log.write('Evaluate\n')
            t_valid = evaluate(model, [user_train, user_valid, user_test, user_num, item_num], True, max_seq_len, genre_dict=genre_dict, timestamp=timestamp)  # 计算验证集上的指标
            t_test = evaluate(model, [user_train, user_valid, user_test, user_num, item_num], False, max_seq_len, genre_dict=genre_dict, timestamp=timestamp)  # 计算测试集上的指标
            print('epoch:%d, time: %f(s), valid (HT@10: %.4f, NDCG@10: %.4f), test (HT@10: %.4f, NDCG@10: %.4f)'
                    % (epoch, t1, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            log.write('epoch:%d, time: %f(s), valid (HT@10: %.4f, NDCG@10: %.4f), test (HT@10: %.4f, NDCG@10: %.4f)\n'
                    % (epoch, t1, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            
            if t_valid[0] > best_val_ht or t_valid[1] > best_val_ndcg:
                best_val_ht = max(t_valid[0], best_val_ht)
                best_val_ndcg = max(t_valid[1], best_val_ndcg)
                best_test_ht = max(t_test[0], best_test_ht)
                best_test_ndcg = max(t_test[1], best_test_ndcg)
                fname = 'SASRec_best_ndcg_{:.4f}_ht_{:.4f}_epoch_{}.pth'.format(best_test_ndcg, best_test_ht, epoch)  # 保存模型的文件名
                print('save model to {}'.format(fname))
                torch.save(model.state_dict(), os.path.join(ckpt_dir, fname))
                counter = 0
                
            else:
                counter += 1
                if counter > 1:  # 如果连续20个epoch没有提升，则停止训练
                    print('Early stopping')
                    log.write('Early stopping\n')
                    break
                
            log.flush()
            start_time = time.time()
        
    print('best valid (NDCG@10: %.4f, HT@10: %.4f)' % (best_val_ndcg, best_test_ht))  # 打印最好的验证集上的指标
    log.write('best valid (NDCG@10: %.4f, HT@10: %.4f)\n' % (best_val_ndcg, best_test_ht))  # 写入日志文件
    print('best test (NDCG@10: %.4f, HT@10: %.4f)' % (best_test_ndcg, best_test_ht))  # 打印最好的测试集上的指标
    log.write('best test (NDCG@10: %.4f, HT@10: %.4f)\n' % (best_test_ndcg, best_test_ht))  # 写入日志文件
    log.close()  # 关闭日志文件
    # 保存最后一个模型
    fname = 'SASRec_last_ndcg_{:.4f}_ht_{:.4f}_epoch_{}.pth'.format(t_test[1], t_test[0], epoch)  # 保存模型的文件名
    print('save model to {}'.format(fname))
    torch.save(model.state_dict(), os.path.join(ckpt_dir, fname))
    print('Done')
    
        
        
    
        
    
    