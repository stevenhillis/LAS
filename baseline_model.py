import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence
import torch.nn.functional as F

from dropout import WeightDrop, LockedDropout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Encoder(nn.Module):
    def __init__(self, base=64):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(40, base, bidirectional=True, batch_first=True)
        self.lstm2 = self.__make_layer__(base*4, base)
        self.lstm3 = self.__make_layer__(base*4, base)
        self.lstm4 = self.__make_layer__(base*4, base)

        self.fc1 = nn.Linear(base*2, base*2)
        self.fc2 = nn.Linear(base*2, base*2)
        self.act = nn.SELU(True)

        self.drop = LockedDropout(.05)

    def _stride2(self, x):
        x = x[:, :x.size(1)//2*2]
        x = self.drop(x)
        x = x.reshape(x.size(0), x.size(1)//2, x.size(2)*2)
        return x

    def __make_layer__(self, in_dim, out_dim):
        lstm = nn.LSTM(input_size=in_dim, hidden_size=out_dim, bidirectional=True, batch_first=True)
        return WeightDrop(lstm, ['weight_hh_l0','weight_hh_l0_reverse'],
                          dropout=0.1, variational=True)

    def forward(self, x):
        x, _ = self.lstm1(x)                      # seq, batch, base*2
        x, seq_len = pad_packed_sequence(x, batch_first=True)
        x = self._stride2(x)                      # seq//2, batch, base*4

        x = pack_padded_sequence(x, seq_len//2, batch_first=True)
        x, _ = self.lstm2(x)                      # seq//2, batch, base*2
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self._stride2(x)                      # seq//4, batch, base*4

        x = pack_padded_sequence(x, seq_len//4, batch_first=True)
        x, _ = self.lstm3(x)                      # seq//4, batch, base*2
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self._stride2(x)                      # seq//8, batch, base*4

        x = pack_padded_sequence(x, seq_len//8, batch_first=True)
        x, (hidden, _) = self.lstm4(x)            # seq//8, batch, base*2
        x, _ = pad_packed_sequence(x, batch_first=True)

        key = self.act(self.fc1(x))
        value = self.act(self.fc2(x))
        hidden = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=1)
        return seq_len//8, key, value, hidden


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def forward(self, hidden2, keys, values, mask, device):
        energy = torch.bmm(keys, hidden2.unsqueeze(2))
        attention = F.softmax(energy, dim=1)
        masked_attention = F.normalize(attention * mask.unsqueeze(2).type(torch.FloatTensor).to(device), p=1)
        context = torch.bmm(masked_attention.permute(0,2,1), values)
        return context.squeeze(1), energy.cpu().squeeze(2).data.numpy()[0]



class Decoder(nn.Module):
    def __init__(self, out_dim, lstm_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(out_dim, lstm_dim)
        self.lstm1 = nn.LSTMCell(lstm_dim*2, lstm_dim)
        self.lstm2 = nn.LSTMCell(lstm_dim, lstm_dim)
        self.drop = nn.Dropout(0.05)
        self.fc = nn.Linear(lstm_dim, out_dim)
        self.embed.weight = self.fc.weight

    def forward(self, x, context, hidden1, cell1, hidden2, cell2, first_step):

        x = self.embed(x)
        x = torch.cat([x, context], dim=1)
        if first_step:
            hidden1, cell1 = self.lstm1(x)
            hidden2, cell2 = self.lstm2(hidden1)
        else:
            hidden1, cell1 = self.lstm1(x, (hidden1, cell1))
            hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))
        x = self.drop(hidden2)
        x = self.fc(x)
        return x, hidden1, cell1, hidden2, cell2



class Seq2Seq(nn.Module):
    def __init__(self, base, out_dim, device='cuda'):
        super().__init__()
        self.encoder = Encoder(base)
        self.decoder = Decoder(out_dim=out_dim, lstm_dim=base*2)
        self.attention = Attention()
        self.out_dim = out_dim
        self.device = device


    def forward(self, inputs, words, TF=0.7):
        testing = False
        if TF == 0:
            batch_size = words.shape[0]
            max_len = 250
            testing = True
        if TF > 0:
            batch_size, max_len = words.shape[0], words.shape[1]
        prediction = torch.zeros(max_len, batch_size, self.out_dim).to(self.device)

        word, hidden1, cell1, hidden2, cell2 = words[:,0], None, None, None, None

        lens, key, value, hidden2 = self.encoder(inputs)
        mask = torch.arange(lens.max()).unsqueeze(0) < lens.unsqueeze(1)
        mask = mask.to(self.device)
        figure = []
        for t in range(1, max_len):
            context, attention = self.attention(hidden2, key, value, mask, self.device)
            word_vec, hidden1, cell1, hidden2, cell2 = self.decoder(word, context, hidden1, cell1, hidden2, cell2, first_step=(t==1))
            prediction[t] = word_vec
            teacher_force = torch.rand(1) < TF
            if teacher_force:
                word = words[:,t]
            elif not testing:
                gumbel_noise = torch.FloatTensor(np.random.gumbel(size=word_vec.size())).to(self.device)
                noisy_word_vec = word_vec + gumbel_noise
                word = noisy_word_vec.max(1)[1]
            elif testing:
                word = word_vec.max(1)[1]

            figure.append(attention)
        prediction = prediction.permute(1,0,2)
        if TF == 0:
            return prediction
        if TF > 0:
            return prediction, np.stack(figure)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.LSTM or type(m) == nn.LSTMCell:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                upper = 1 / np.sqrt(m.hidden_size)
                nn.init.uniform_(param, -upper, upper)


def plot_grad_flow(named_parameters, gradient_path, epoch_num, batch_num):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.savefig(gradient_path + "/epoch{:}_batch{:}.png".format(epoch_num, batch_num), bbox_inches="tight")
