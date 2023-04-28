import torch
from torch import nn

class CatEmbLSTM(nn.Module):
    def __init__(self, num_input_size = 8, lstm_h = 128, lstm_num_layers = 2,
                 n_emb_cls = 256, emb_h = 128, batch_first = True):
        super().__init__()
        self.rnn = nn.LSTM(num_input_size + emb_h, lstm_h, lstm_num_layers, batch_first = batch_first)
        self.cat_to_emb = nn.Embedding(n_emb_cls, emb_h)
        self.norm = nn.LayerNorm(emb_h)
        self.lin_reg = nn.Linear(lstm_h, 1)

        self.norm_num_features =  nn.LayerNorm(num_input_size)

        
    def forward(self, x):
        num_feat, cat_feat = x

        normalized_num_feat = self.norm_num_features(num_feat)

        cat_emb = self.cat_to_emb(cat_feat)
        cat_emb = cat_emb.view(cat_emb.shape[0], cat_emb.shape[1], cat_emb.shape[-1])
        input_feat = torch.cat([normalized_num_feat, cat_emb], dim = -1)
        output_feat, (hn, cn) = self.rnn(input_feat)
        input_reg = output_feat.sum(1)
        output = self.lin_reg(input_reg)
        return output.view(-1)
