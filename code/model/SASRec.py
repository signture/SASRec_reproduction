# 实现是参照https://github.com/pmixer/SASRec.pytorch/blob/main/python/model.py，但是先自己实现一个基础版的
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    # 其实不应该写在这个文件里面，但是这样写稍微方便一点就这样写了
    def __init__(self, temp, device):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, emb1, emb2):
        # emb1\2: [batch_size, (seq_len), hidden_dim]
        batch_size = emb1.shape[0]
        score11 = torch.matmul(emb1, emb1.transpose(0, 1)) / self.temp
        score22 = torch.matmul(emb2, emb2.transpose(0, 1)) / self.temp
        score12 = torch.matmul(emb1, emb2.transpose(0, 1)) / self.temp
        
        pos_diff = torch.diagonal(score12).view(-1, 1)
        
        # 构造掩码，忽略对角线上的正样本对
        mask = (-torch.eye(batch_size).long() + 1).bool()
        score11 = score11[mask].view(batch_size, -1)
        score22 = score22[mask].view(batch_size, -1)
        score12 = score12[mask].view(batch_size, -1)
        
        score = torch.cat((pos_diff, score11, score22, score12), dim=1).to(self.device)  # [batch_size, batch_size * 3 - 2]
        label = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # [batch_size]
        
        return self.ce_loss(score, label)
    

class Genre_embedding(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(Genre_embedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(18, hidden_dim)  # 18是genre的数量，hidden_dim是embedding的维度
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 对输入进行归一化，防止梯度爆炸
    
    def forward(self, x):
        # x: [batch_size, seq_len, 18]
        assert x.shape[2] == 18, 'expected dim 18 in input, but got {} instead.'.format(x.shape[2])
        x = self.fc(x)  # x: [batch_size, seq_len, hidden_dim]
        x = self.dropout(x)  # x: [batch_size, seq_len, hidden_dim]
        x = self.layer_norm(x)  # x: [batch_size, seq_len, hidden_dim]
        return x  # x: [batch_size, seq_len, hidden_dim


class Point_Wise_Feed_Forward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(Point_Wise_Feed_Forward, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 对输入进行归一化，防止梯度爆炸
        # 官方实现采用一维卷积在时间步上进行卷积，我觉得还是很合理的
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)  # 一维卷积
        self.dropout1 = nn.Dropout(dropout)  # 防止过拟合
        self.relu = nn.ReLU()  # 激活函数
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)  # 一维卷积
        self.dropout2 = nn.Dropout(dropout)  # 防止过拟合
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        assert x.shape[2] == self.hidden_dim, 'expected dim {} in input, but got {} instead.'.format(self.hidden_dim, x.shape[2])
        x = self.layer_norm(x)  # x: [batch_size, seq_len, hidden_dim]
        y = x.transpose(1, 2)  # x: [batch_size, hidden_dim, seq_len]
        y = self.conv1(y)  # x: [batch_size, hidden_dim, seq_len]
        y = self.dropout1(y)  # x: [batch_size, hidden_dim, seq_len]
        y = self.relu(y)  # x: [batch_size, hidden_dim, seq_len]
        y = self.conv2(y)  # x: [batch_size, hidden_dim, seq_len]
        y = self.dropout2(y)  # x: [batch_size, hidden_dim, seq_len]
        y = y.transpose(1, 2)  # x: [batch_size, seq_len, hidden_dim]
        return x + y  # y: [batch_size, seq_len, hidden_dim]  # 残差连接

class Self_Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, version='paper'):
        super(Self_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 对输入进行归一化，防止梯度爆炸
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        if version != None:
            assert version in ['paper', 'official'], 'version must be paper or official'
        self.version = version
        
    def forward(self, x, mask):
        # x: [batch_size, seq_len, hidden_dim]
        assert x.shape[2] == self.hidden_dim, 'hidden_dim must be equal to hidden_dim'
        x = x.transpose(0, 1)  # x: [seq_len, batch_size, hidden_dim]
        if self.version == 'paper':  
            x_norm = self.layer_norm(x)  # 对输入进行归一化，防止梯度爆炸
            y, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask) 
            y = self.dropout(y)
            return x + y  # 残差连接  # y: [seq_len, batch_size, hidden_dim]
        
        elif self.version == 'official':  # 这里是官方的实现，但是感觉似乎并没有理解为什么
            Q = self.layer_norm(x)  # 对输入进行归一化，防止梯度爆炸
            y, _ = self.attention(Q, x, x, attn_mask=mask)  # 这里只对Q进行标准化是官方的做法，但是感觉似乎没能理解为什么
            y = self.dropout(y)
            return Q + y  # 残差连接  # y: [seq_len, batch_size, hidden_dim]
        
        else:
            x_norm = self.layer_norm(x)  # 对输入进行归一化，防止梯度爆炸
            y, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)  # 这里只对Q进行标准化是官方的做法，但是感觉似乎没能理解为什么
            y = self.dropout(y)
            return x_norm + y  # 残差连接  # y: [seq_len, batch_size, hidden_dim]
        
class WeightedAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始权重均衡
    
    def forward(self, e1, e2):
        return self.alpha * e1 + (1 - self.alpha) * e2
    
class EfficientMultiHotEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        # 初始化可学习的嵌入矩阵
        self.embed_matrix = nn.Parameter(torch.randn(num_features, embed_dim))
        nn.init.xavier_uniform_(self.embed_matrix)
        
    def forward(self, multi_hot):
        # multi_hot: [batch_size, num_features]
        return torch.matmul(multi_hot.float(), self.embed_matrix) 

class SASRec(nn.Module):
    def __init__(self, item_num, hidden_dim, num_heads, num_blocks, device, l2_emb=0.0, dropout=0.2, max_len=200, temp=1, version='paper'):
        super(SASRec, self).__init__()
        self.device = device
        self.l2_emb = l2_emb  # 这个是实现里有的一个正则项
        self.num_blocks = num_blocks
        # 模型首先是一个embedding层
        # item_num + 1是因为要考虑到0这个位置的embedding为0，因为0是用来填充的
        self.item_emb = nn.Embedding(item_num + 1, hidden_dim, padding_idx=0) 
        self.genre_emb = EfficientMultiHotEmbedding(18, hidden_dim)
        self.emb_gate = WeightedAdd()
        # self.emb_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        # 接下来的PE不使用transformer的，论文里用的是一个可学习的，然后实现里用的是embeddding层
        self.pos_emb = nn.Embedding(max_len + 1, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList([Self_Attention(hidden_dim, num_heads, dropout, version) for _ in range(num_blocks)]) 
        self.feed_forward_blocks = nn.ModuleList([Point_Wise_Feed_Forward(hidden_dim, dropout) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 对输入进行归一化，防止梯度爆炸
        self.cl_loss = ContrastiveLoss(temp, device)
        self.init_network()  # 初始化网络参数
    
    def init_network(self):  # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def embed_item(self, log_seqs):
        if isinstance(log_seqs[0], tuple):
            log_seqs = list(zip(*log_seqs))
            item_seqs = self.item_emb(torch.LongTensor(np.array(log_seqs[0])).to(self.device))
            genre_seqs = self.genre_emb(torch.Tensor(np.array(log_seqs[1])).to(self.device))
            seqs = self.emb_gate(item_seqs, genre_seqs)  # seqs: [batch_size, seq_len, hidden_dim]
            log_seqs = np.array(log_seqs[0])
        else:
            log_seqs = np.array(log_seqs)
            seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))  # seqs: [batch_size, seq_len, hidden_dim]  # 这里的seq_len是不包括最后一个的，因为最后一个是用来预测的
        return seqs, log_seqs  # seqs: [batch_size, seq_len, hidden_dim]  # log_seqs: [batch_size, seq_len]
    
    def get_embedding(self, log_seqs):
        # seqs: [batch_size, seq_len]        
        # 更改了一下，如果输入时tuple，那么就有两个embedding
        seqs, log_seqs = self.embed_item(log_seqs)  # seqs: [batch_size, seq_len, hidden_dim]
        seqs *= self.item_emb.embedding_dim ** 0.5  # 似乎是经验设置
        
        # 提取位置信息
        # 这个其实会感觉有点奇怪，因为前面有可能被pad为0了，但或许还是可以提供相对位置信息？
        positions = torch.tile(torch.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])  # positions: [batch_size, seq_len]
        positions = positions * (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))  # seqs: [batch_size, seq_len, hidden_dim] 
        seqs = self.emb_dropout(seqs)  # seqs: [batch_size, seq_len, hidden_dim] 
        return seqs  # seqs: [batch_size, seq_len, hidden_dim]
        
    def log2feat(self, log_seqs):
        # log_seqs: [batch_size, seq_len]  # 这里的seq_len是不包括最后一个的，因为最后一个是用来预测的
        seqs = self.get_embedding(log_seqs)  # seqs: [batch_size, seq_len, hidden_dim]
        
        # 接下来是mask，感觉就是transformer的decoder的那个mask的做法
        seq_len = seqs.shape[1] 
        atten_mask = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))  # atten_mask: [seq_len, seq_len]  
        
        for i in range(self.num_blocks):  # 这里是多头注意力机制
            seqs = self.attention_blocks[i](seqs, atten_mask)  # seqs: [seq_len, batch_size, hidden_dim]
            seqs = seqs.transpose(0, 1)  # seqs: [batch_size, seq_len, hidden_dim] 
            seqs = self.feed_forward_blocks[i](seqs)  # seqs: [batch_size, seq_len, hidden_dim]
        
        log_feats = self.layer_norm(seqs)  # seqs: [batch_size, seq_len, hidden_dim]
        return log_feats  # log_feats: [batch_size, seq_len, hidden_dim]
    
    def forward(self, log_seqs, pos_seqs, neg_seqs):  # 这里的pos_seqs和neg_seqs都是用来计算损失的，pos_seqs是用来预测的
        # log_seqs: [batch_size, seq_len]  # 这里的seq_len是不包括最后一个的，因为最后一个是用来预测的
        log_feats = self.log2feat(log_seqs)  # log_feats: [batch_size, seq_len, hidden_dim]
        
        pos_embs, pos_seqs = self.embed_item(pos_seqs)  # pos_embs: [batch_size, seq_len, hidden_dim] 
        neg_embs, _ = self.embed_item(neg_seqs)  # neg_embs: [batch_size, seq_len, hidden_dim]
        
        # pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))  # pos_embs: [batch_size, seq_len, hidden_dim]  
        # neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))  # neg_embs: [batch_size, seq_len, hidden_dim] 
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # pos_logits: [batch_size, seq_len] 
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # neg_logits: [batch_size, seq_len]
        
        return pos_logits, neg_logits, pos_seqs  # pos_logits: [batch_size, seq_len]  # neg_logits: [batch_size, seq_len]
    
    def predict(self, log_seqs, item_indices):  # 这里的item_indices是用来预测的，就是没见过的物品
        # log_seqs: [batch_size, seq_len]  # 这里的seq_len是不包括最后一个的，因为最后一个是用来预测的
        log_feats = self.log2feat(log_seqs)  # log_feats: [batch_size, seq_len, hidden_dim]
        final_feat = log_feats[:, -1, :]  # final_feat: [batch_size, hidden_dim]  # 这里的-1是用来提取最后一个的
        item_embs, _ = self.embed_item(item_indices)
        # item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))  # item_embs: [batch_size, item_num, hidden_dim]
        if len(item_embs.shape) == 2:
            item_embs = item_embs.unsqueeze(0)
        logits = torch.einsum('bih,bh->bi', item_embs, final_feat)  # logits: [batch_size, item_num]
        return logits  # logits: [batch_size, item_num]
    
    def get_weight(self):
        return self.emb_gate.alpha.item()  # 获取权重
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRec(10, 10, 2, 2, device)
    model.to(device)
    log_seq = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])  # [batch_size, seq_len]
    pos_seq = torch.tensor([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])  # [batch_size, seq_len]
    neg_seq = torch.tensor([[3, 4, 5, 6, 7], [3, 4, 5, 6, 7]])  # [batch_size, seq_len]
    pos_logits, neg_logits = model(log_seq, pos_seq, neg_seq)  # [batch_size, seq_len]  # [batch_size, seq_len]
    print(pos_logits.shape)  # [batch_size, seq_len]  # [batch_size, seq_len]
    print(neg_logits.shape)  # [batch_size, seq_len]  # [batch_size, seq_len]
    item_indices = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])  # [batch_size, item_num]
    logits = model.predict(log_seq, item_indices)  # [batch_size, item_num]
    print(logits.shape)  # [batch_size, item_num
    