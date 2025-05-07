import torch
from model.SASRec import SASRec

def load_model(path:str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRec(item_num=3416, hidden_dim=50, num_heads=1, num_blocks=2, device=device)
    model.load_state_dict(torch.load(path), strict=False)
    return model

if __name__ == "__main__":
    model_path = '../result/2025-05-07-18-14-16/checkpoint/SASRec_best_ndcg_0.6154_ht_0.8384_epoch_120.pth'
    model = load_model(model_path)
    print(model)