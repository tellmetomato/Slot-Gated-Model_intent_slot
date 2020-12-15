import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, T, D) -> (B , T, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, T, D) * (B, T, 1) -> (B, D)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class SlotFillingAndIntentDetermination(nn.Module):
    def __init__(self, vocab_size, label_size, mode='gru', bidirectional=True, cuda=True, is_training=True,intent_size=38):
        super(SlotFillingAndIntentDetermination, self).__init__()
        self.is_training = is_training
        embedding_dim = 100
        hidden_size = 75
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.GRU(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            batch_first=True)

        self.attention = Attention(hidden_size*2)
        self.fc = nn.Linear(2*hidden_size, label_size)
        self.fc_W = nn.Linear(2*hidden_size,2*hidden_size)
        self.fc_v = nn.Linear(2*hidden_size,1)
        self.fc_intent = nn.Linear(150,38)

    def forward(self, X):
        embed = self.embedding(X.long())

        embed = F.dropout(embed, p=0.2, training=self.is_training)    #dropout = 0.2
        

        outputs, intent_outs = self.rnn(embed)
        
        intent_outs_att,_ = self.attention(outputs)
        #print(intent_outs_att)
        gate = F.tanh(self.fc_v(F.tanh(self.fc_W(intent_outs_att) + outputs)).sum()) ##
        outputs = self.fc(outputs + outputs*gate)

        intent_outs = self.fc_intent(intent_outs.view(1,-1) + intent_outs_att)
        #print(intent_outs.size())

        return outputs,intent_outs



