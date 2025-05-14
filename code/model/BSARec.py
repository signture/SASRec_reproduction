import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SASRec import SASRec

class FrequencyLayer(nn.Module):
    def __init__(self, hidden_dim, dropout, c=9):
        super(FrequencyLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.c = c // 2 + 1  # 用于区分高低频
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 初始化beta

    def forward(self, input_tensor):
        # [batch_size, seq_len, hidden_dim]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.dropout(sequence_emb_fft)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states

class BSARec(SASRec):
    def __init__(self, item_num, hidden_dim, num_heads, num_blocks, device, l2_emb=0.0, dropout=0.2, max_len=200, temp=1, version='paper', c=9):
        SASRec.__init__(self, item_num, hidden_dim, num_heads, num_blocks, device, l2_emb, dropout, max_len, temp, version)
        self.frequency_layers = nn.ModuleList([FrequencyLayer(hidden_dim, dropout, c) for _ in range(num_blocks)])
        self.alpha = 0.3
        
    def log2feat(self, log_seqs, **kwargs):
        # log_seqs: [batch_size, seq_len]  # 这里的seq_len是不包括最后一个的，因为最后一个是用来预测的
        if kwargs.get('time_seq') is not None:
            seqs = self.get_embedding(log_seqs, time_seq=kwargs['time_seq'])
        else:
            seqs = self.get_embedding(log_seqs)  # seqs: [batch_size, seq_len, hidden_dim]
        
        # 接下来是mask，感觉就是transformer的decoder的那个mask的做法
        seq_len = seqs.shape[1] 
        atten_mask = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))  # atten_mask: [seq_len, seq_len]  
        
        for i in range(self.num_blocks):  # 这里是多头注意力机制
            seqs_att = self.attention_blocks[i](seqs, atten_mask)  # seqs: [seq_len, batch_size, hidden_dim]
            seqs_att = seqs_att.transpose(0, 1)  # seqs: [batch_size, seq_len, hidden_dim] 
            seqs_freq = self.frequency_layers[i](seqs)  # seqs: [batch_size, seq_len, hidden_dim]
            seqs = self.alpha * seqs_freq + (1 - self.alpha) * seqs_att
            seqs = self.feed_forward_blocks[i](seqs)  # seqs: [batch_size, seq_len, hidden_dim]
        
        log_feats = self.layer_norm(seqs)  # seqs: [batch_size, seq_len, hidden_dim]
        return log_feats  # log_feats: [batch_size, seq_len, hidden_dim]
    
    def get_weight(self):
        return self.alpha.item()
        
        

        
    
        
        