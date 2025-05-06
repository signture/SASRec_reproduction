import torch
from model.SASRec import SASRec

def load_model(path:str):
    model = SASRec()
    model.load_state_dict(torch.load(path), strict=False)
    return model