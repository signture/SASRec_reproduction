import torch
import numpy as np

def train_epoch(model, optimizer, criterion, num_batch, sampler, device, log=None, configs=None):
    model.train()
    total_loss = 0.0
    cl_loss = 0.0
    for step in range(num_batch):
        if configs.type != 'none':  # 如果contrastive不为None，就进行对比学习
            u, seq, pos, neg, aug1, aug2 = sampler.next_batch()  # 从sampler中获取一个batch的数据
            # seq, pos, neg, aug1, aug2 = np.array(seq), np.array(pos), np.array(neg), np.array(aug1), np.array(aug2)  # 将数据转换为numpy数组
            aug_emb1 = model.get_embedding(aug1)
            aug_emb2 = model.get_embedding(aug2)
            batch_size = aug_emb1.shape[0]
            if configs.view_type == 'flatten':
                cl_loss = model.cl_loss(aug_emb1.view(batch_size, -1), aug_emb2.view(batch_size, -1))
            elif configs.view_type == 'mean':
                cl_loss = model.cl_loss(aug_emb1.mean(dim=1), aug_emb2.mean(dim=1))
        else:
            u, seq, pos, neg = sampler.next_batch()
            # seq, pos, neg = np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits, pos = model(seq, pos, neg) 
        pos_labels, neg_labels = torch.ones_like(pos_logits, device=device), torch.zeros_like(neg_logits, device=device)
        optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = criterion(pos_logits[indices], pos_labels[indices]) + criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters():
            loss += model.l2_emb * torch.norm(param)
        if configs.type != 'none':  # 如果contrastive不为None，就将对比学习的损失加入到总损失中
            loss += cl_loss * configs.weight
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if step % 10 == 0:  # 每100个batch打印一次损失
            # print(f"Batch {step}, Loss: {loss.item()}, CL_loss: {cl_loss.item()}")
            if log is not None:  # 如果log不为None，就将损失写入log文件中
                if configs.type!= 'none':  # 如果contrastive不为None，就将对比学习的损失加入到总损失中
                    log.write(f"Batch {step}, Loss: {loss.item()}, CL_loss: {cl_loss.item()}\n")
                else:
                    log.write(f"Batch {step}, Loss: {loss.item()}\n")
            
        
class EarlyStopping:
    def __init__(self, path, log, patience=10, verbose=False, delta=0):
        self.save_path = path
        self.patience = patience  # 耐心值
        self.verbose = verbose  # 是否打印信息
        self.counter = 0  # 计数器
        self.best_scores = None  # 最佳分数
        self.early_stop = False  # 是否停止
        self.delta = delta  # 最小变化量
        self.log = log

    def __call__(self, scores, model, epoch):
        # scores 是一个包含多个指标分数的列表
        # 将scores转换为列表
        scores = list(scores)
        if self.best_scores is None:  # 第一次调用
            self.best_scores = scores.copy()
            self.save_checkpoint(scores, model, self.save_path + '_score_epoch_{}.pth'.format(scores[0], epoch))  # 保存模型
        else:
            any_improvement = False
            for i in range(len(scores)):
                if scores[i] > self.best_scores[i] + self.delta:  # 只要有一个指标提升
                    any_improvement = True
                    break

            if any_improvement:  # 性能提升
                self.best_scores = scores.copy()
                self.save_checkpoint(scores, model, self.save_path + '_score_epoch_{}.pth'.format(scores[0], epoch))  # 保存模型
                if self.log is not None:  # 如果log不为None，就将信息写入log文件中
                    self.log.write(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
                self.counter = 0  # 计数器清零
            else:  # 性能没有提升
                self.counter += 1  # 计数器加1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # 打印信息
                if self.log is not None:  # 如果log不为None，就将信息写入log文件中
                    self.log.write(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
                if self.counter >= self.patience:  # 计数器超过耐心值
                    self.early_stop = True  # 停止

    def save_checkpoint(self, scores, model):
        if self.verbose:  # 是否打印信息
            score_str = " ".join([f"{score:.6f}" for score in scores])
            best_score_str = " ".join([f"{score:.6f}" for score in self.best_scores])
            print(f'Validation scores improved ({best_score_str} --> {score_str}).  Saving model ...')  # 打印信息
            if self.log is not None:  # 如果log不为None，就将信息写入log文件中
                self.log.write(f'Validation scores improved ({best_score_str} --> {score_str}).  Saving model...\n')
        torch.save(model.state_dict(), self.save_path)  # 保存模型
        self.best_scores = scores.copy()  # 更新最佳分数列表
        
    
        