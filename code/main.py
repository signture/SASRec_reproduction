import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import logging
from data.dataset import SeqItemDataset, WarpSampler
from model.SASRec import SASRec
from utils.metric import evaluate
from utils.utils import data_split, set_seed
from utils.train import train_epoch, EarlyStopping



if __name__ == "__main__":
    
    # 训练超参数
    max_seq_len = 200
    batch_size = 128
    learning_rate = 0.001
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_only = False
    patience = 10
    num_workers = 4
    seed = 14531
    
    # 模型超参数
    hidden_dim = 50
    num_heads = 1
    dropout_rate = 0.2
    num_blocks = 2
    l2_emb = 0.0001
    
    # 相关路径设置
    data_path = '../data/ml-1m/ratings_process.txt'
    state_dict_path = '../model/SASRec.pth'
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
    
    # 写入模型超参数
    log.write('hidden_dim: {}\n'.format(hidden_dim))
    log.write('num_heads: {}\n'.format(num_heads))
    log.write('dropout_rate: {}\n'.format(dropout_rate))
    log.write('num_blocks: {}\n'.format(num_blocks))
    log.write('l2_emb: {}\n'.format(l2_emb))
    log.write('learning_rate: {}\n'.format(learning_rate))
    log.write('batch_size: {}\n'.format(batch_size))
    log.write('epochs: {}\n'.format(epochs))
    log.write('patience: {}\n'.format(patience))
    log.write('num_workers: {}\n'.format(num_workers))
    log.write('seed: {}\n'.format(seed))
    log.write('device: {}\n'.format(device))   
    
    # 设置随机中种子
    set_seed(seed)
    
    # 加载数据并划分
    [user_train, user_valid, user_test, user_num, item_num] = data_split(data_path)
    num_batch = (len(user_train) - 1) // batch_size + 1  # 计算batch数量
    print('the number of batch for a epoch is %.2f' % num_batch)
    asl = 0.0  # 计算平均序列长度
    for user in user_train:
        asl += len(user_train[user])
    print('the average sequence length is %.2f' % (asl / len(user_train)))
    sampler = WarpSampler(user_train, user_num, item_num, batch_size=batch_size, maxlen=max_seq_len, n_workers=num_workers)
    # dataset = SeqItemDataset(user_train, user_num, item_num, max_seq_len)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # 模型加载与训练准备
    model = SASRec(item_num, hidden_dim, num_heads, num_blocks, device, l2_emb, dropout_rate, max_seq_len)
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
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    
    T = 0.0    
    t0 = time.time()  # 记录开始时间    
    for epoch in range(1, epochs + 1):  # 训练
        print('Epoch: %d' % epoch)
        log.write('Epoch: %d' % epoch)
        train_epoch(model, optimizer, criterion, num_batch, sampler, device, log)  # 训练一个epoch
        if epoch % patience == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluate')
            t_valid = evaluate(model, [user_train, user_valid, user_test, user_num, item_num], True, max_seq_len)  # 计算验证集上的指标
            t_test = evaluate(model, [user_train, user_valid, user_test, user_num, item_num], False, max_seq_len)  # 计算测试集上的指标
            print('epoch:%d, time: %f(s), valid (HT@10: %.4f, NDCG@10: %.4f), test (HT@10: %.4f, NDCG@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                fname = 'SASRec_best_ndcg_{:.4f}_hr_{:.4f}_epoch_{}.pth'.format(best_test_ndcg, best_test_hr, epoch)  # 保存模型的文件名
                print('save model to {}'.format(fname))
                torch.save(model.state_dict(), os.path.join(ckpt_dir, fname))
                
            log.flush()
            t0 = time.time()
        
    print('best valid (NDCG@10: %.4f, HR@10: %.4f)' % (best_val_ndcg, best_val_hr))  # 打印最好的验证集上的指标
    log.write('best valid (NDCG@10: %.4f, HR@10: %.4f)\n' % (best_val_ndcg, best_val_hr))  # 写入日志文件
    print('best test (NDCG@10: %.4f, HR@10: %.4f)' % (best_test_ndcg, best_test_hr))  # 打印最好的测试集上的指标
    log.write('best test (NDCG@10: %.4f, HR@10: %.4f)\n' % (best_test_ndcg, best_test_hr))  # 写入日志文件
    log.close()  # 关闭日志文件
    # 保存最后一个模型
    fname = 'SASRec_last_ndcg_{:.4f}_hr_{:.4f}_epoch_{}.pth'.format(t_test[0], t_test[1], epoch)  # 保存模型的文件名
    print('save model to {}'.format(fname))
    torch.save(model.state_dict(), os.path.join(ckpt_dir, fname))
    print('Done')
    
        
        
    
        
    
    