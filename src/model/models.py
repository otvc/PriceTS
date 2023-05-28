import torch
from torch import nn

class CatEmbLSTM(nn.Module):
    def __init__(self, num_input_size = 8, lstm_h = 128, lstm_num_layers = 2,
                 n_emb_cls = 256, emb_h = 128, batch_first = True, fc_hidden = 128,dropout=0.5):
        super().__init__()
        self.fc_in=nn.Linear(num_input_size,fc_hidden)
        self.rnn = nn.LSTM(fc_hidden+emb_h, lstm_h, lstm_num_layers, batch_first = batch_first,dropout=dropout)
        self.cat_to_emb = nn.Embedding(n_emb_cls, emb_h)
        self.norm = nn.LayerNorm(emb_h)
        self.fc1 = nn.Linear(lstm_h, fc_hidden)
        self.fc_out = nn.Linear(fc_hidden, 1)

        self.norm_num_features =  nn.BatchNorm1d(num_input_size)

        self.func_activ = nn.ReLU()
        self.drop=nn.Dropout(p=dropout)

        
    def forward(self, x):
        num_feat, cat_feat = x


        normalized_num_feat = self.norm_num_features(num_feat.permute(0, 2, 1)).permute(0,2,1)
        num_rnn_input =  self.func_activ(self.fc_in(normalized_num_feat))

        cat_emb = self.cat_to_emb(cat_feat)
        cat_emb = cat_emb.view(cat_emb.shape[0], cat_emb.shape[1], cat_emb.shape[-1])
        input_feat = torch.cat([num_rnn_input, cat_emb], dim = -1)
        output_feat, (hn, cn) = self.rnn(input_feat)
        # print(output_feat.shape)
        input_reg = self.drop(output_feat)
        fc_hid = self.func_activ(self.fc1(input_reg))
        output = self.fc_out(fc_hid)
        return output[:,-1,:1].view(-1)
