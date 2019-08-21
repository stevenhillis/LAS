import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from torch.distributions import Gumbel

from dropout import LockedDropout, WeightDrop


class Seq2Seq(nn.Module):
    def __init__(self, batch_size, embedding_size, attention_size, vocab_size, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.modules = []
        self.encoder = Listener_Encoder(batch_size, attention_size)
        self.modules.append(self.encoder)
        self.decoder = Speller_Decoder(batch_size, embedding_size, attention_size, vocab_size, device)
        self.modules.append(self.decoder)
        self.net = nn.Sequential(*self.modules)

    def forward(self, frames, transcripts, testing=False):
        keys, values, output_lengths = self.encoder.forward(frames)
        if not testing:
            probs, attention_across_timesteps = self.decoder.forward(keys, values, output_lengths.to(self.device), transcripts.to(self.device), testing)
            return probs, attention_across_timesteps
        else:
            probs = self.decoder.forward(keys, values, output_lengths.to(self.device), transcripts.to(self.device), testing)
            return probs

    def beam_decode(self):
        pass

class Listener_Encoder(nn.Module):
    def __init__(self, batch_size, attention_size):
        self.batch_size = batch_size
        self.attention_size = attention_size
        super(Listener_Encoder, self).__init__()

        self.modules = []
        self.bilstm = nn.LSTM(input_size=40,
                              hidden_size=256,
                              num_layers=3,
                              bidirectional=True,
                              batch_first=True)
        self.modules.append(self.bilstm)
        self.p_bilstm_a = WeightDrop(nn.LSTM(input_size=1024,
                                             hidden_size=256,
                                             num_layers=1,
                                             bidirectional=True,
                                             batch_first=True),
                                     ['weight_hh_l0', 'weight_hh_l0_reverse'],
                                     dropout=0.1, variational=True)
        self.p_bilstm_b = WeightDrop(nn.LSTM(input_size=1024,
                                             hidden_size=256,
                                             num_layers=1,
                                             bidirectional=True,
                                             batch_first=True),
                                     ['weight_hh_l0', 'weight_hh_l0_reverse'],
                                     dropout=0.1, variational=True)
        self.p_bilstm_c = WeightDrop(nn.LSTM(input_size=1024,
                                             hidden_size=256,
                                             num_layers=1,
                                             bidirectional=True,
                                             batch_first=True),
                                     ['weight_hh_l0', 'weight_hh_l0_reverse'],
                                     dropout=0.1, variational=True)
        self.modules.append(self.p_bilstm_a)
        self.modules.append(self.p_bilstm_b)
        self.modules.append(self.p_bilstm_c)
        self.value_projection = nn.Linear(512, self.attention_size)
        self.modules.append(self.value_projection)
        self.key_projection = nn.Linear(512, self.attention_size)
        self.modules.append(self.key_projection)
        self.act = nn.SELU(True)
        self.modules.append(self.act)
        self.drop = LockedDropout(0.5)
        self.modules.append(self.drop)

        self.net = nn.ModuleList(self.modules)

    def forward(self, frames):
        bilstm_output, (hidden_state, cell_state) = self.bilstm(frames)
        unpacked_bilstm_output, unpacked_bilstm_output_lengths = rnn.pad_packed_sequence(bilstm_output, batch_first=True)
        current_output_size = list(unpacked_bilstm_output.size())
        if current_output_size[1] % 2 != 0:
            unpacked_bilstm_output = unpacked_bilstm_output[:, :current_output_size[1] - 1, :]
            current_output_size = list(unpacked_bilstm_output.size())
        unpacked_bilstm_output = unpacked_bilstm_output.contiguous().view(current_output_size[0], current_output_size[1] // 2, current_output_size[2] * 2)
        packed_bilstm_output = rnn.pack_padded_sequence(unpacked_bilstm_output, unpacked_bilstm_output_lengths / 2, batch_first=True)

        p_bilstm_output_a, (hidden_state, cell_state) = self.p_bilstm_a(packed_bilstm_output)
        unpacked_p_bilstm_output_a, unpacked_p_bilstm_output_a_lengths = rnn.pad_packed_sequence(p_bilstm_output_a, batch_first=True)
        current_output_size = list(unpacked_p_bilstm_output_a.size())
        if current_output_size[1] % 2 != 0:
            unpacked_p_bilstm_output_a = unpacked_p_bilstm_output_a[:, :current_output_size[1] - 1, :]
            current_output_size = list(unpacked_p_bilstm_output_a.size())
        unpacked_p_bilstm_output_a = unpacked_p_bilstm_output_a.contiguous().view(current_output_size[0], current_output_size[1] // 2, current_output_size[2] * 2)
        # unpacked_p_bilstm_input_b = torch.cat((unpacked_bilstm_output, unpacked_p_bilstm_output_a), dim=1)
        # packed_p_bilstm_input_b = rnn.pack_padded_sequence(unpacked_p_bilstm_input_b, unpacked_bilstm_output_lengths / 2 + unpacked_p_bilstm_output_a_lengths / 2, batch_first=True)
        packed_p_bilstm_input_b = rnn.pack_padded_sequence(unpacked_p_bilstm_output_a, unpacked_p_bilstm_output_a_lengths / 2, batch_first=True)

        p_bilstm_output_b, (hidden_state, cell_state) = self.p_bilstm_b(packed_p_bilstm_input_b)
        unpacked_p_bilstm_output_b, unpacked_p_bilstm_output_b_lengths = rnn.pad_packed_sequence(p_bilstm_output_b, batch_first=True)
        current_output_size = list(unpacked_p_bilstm_output_b.size())
        if current_output_size[1] % 2 != 0:
            unpacked_p_bilstm_output_b = unpacked_p_bilstm_output_b[:, :current_output_size[1] - 1, :]
            current_output_size = list(unpacked_p_bilstm_output_b.size())
        unpacked_p_bilstm_output_b = unpacked_p_bilstm_output_b.contiguous().view(current_output_size[0], current_output_size[1] // 2, current_output_size[2] * 2)
        # unpacked_p_bilstm_input_c = torch.cat((unpacked_p_bilstm_output_a, unpacked_p_bilstm_output_b), dim=1)
        # packed_p_bilstm_input_c = rnn.pack_padded_sequence(unpacked_p_bilstm_input_c, unpacked_p_bilstm_output_a_lengths / 2 + unpacked_p_bilstm_output_b_lengths / 2, batch_first=True)
        packed_p_bilstm_input_c = rnn.pack_padded_sequence(unpacked_p_bilstm_output_b, unpacked_p_bilstm_output_b_lengths / 2, batch_first=True)

        p_bilstm_output_c, (hidden_state, cell_state) = self.p_bilstm_c(packed_p_bilstm_input_c)
        unpacked_p_bilstm_output_c, unpacked_p_bilstm_output_c_lengths = rnn.pad_packed_sequence(p_bilstm_output_c, batch_first=True)

        keys = self.act(self.key_projection(unpacked_p_bilstm_output_c))
        values = self.act(self.value_projection(unpacked_p_bilstm_output_c))

        return keys, values, unpacked_p_bilstm_output_c_lengths

class Speller_Decoder(nn.Module):
    def __init__(self, batch_size, embedding_size, attention_size, vocab_size, device):
        super(Speller_Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_dimension = embedding_size
        self.attention_size = attention_size
        self.hidden_size = 512
        self.vocab_size = vocab_size
        self.query = torch.zeros(batch_size, self.attention_size).to(device)
        self.teacher_forcing_prob = 0.9
        self.device = device

        self.modules = []
        self.embedding = nn.Linear(self.vocab_size, self.decoder_dimension)
        self.modules.append(self.embedding)
        self.lstm_a = nn.LSTMCell(self.decoder_dimension+self.attention_size, self.hidden_size)
        self.modules.append(self.lstm_a)
        self.lstm_b = nn.LSTMCell(self.hidden_size, self.decoder_dimension-self.attention_size)
        self.modules.append(self.lstm_b)
        self.drop = nn.Dropout(0.05)
        self.modules.append(self.drop)
        self.query_linear = nn.Linear(self.decoder_dimension-self.attention_size, self.attention_size)
        # self.probs_linear = nn.Linear(self.decoder_dimension, self.vocab_size)
        self.modules.append(self.query_linear)
        # self.modules.append(self.probs_linear)
        self.net = nn.Sequential(*self.modules)


    def update_teacher_forcing_prob(self, teacher_forcing_prob):
        self.teacher_forcing_prob = teacher_forcing_prob

    def get_teacher_forcing_prob(self):
        return self.teacher_forcing_prob

    def forward(self, keys, values, output_lengths, transcripts, testing):
        if not testing:
            sizes = list(transcripts.size())
            one_hot_transcripts = torch.zeros((sizes[0], sizes[1], self.vocab_size)).to(self.device)
            one_hot_transcripts.scatter(2, torch.unsqueeze(transcripts, 2), 1)
            transcripts = self.embedding(one_hot_transcripts.to(self.device)).to(self.device)
        else:
            sizes = [keys.size()[0], 250]
            predicted_next_word = torch.zeros((sizes[0], self.vocab_size)).to(self.device).scatter(1, torch.unsqueeze(transcripts, 1), 1)
            predicted_next_word = predicted_next_word.to(self.device)
            # transcripts = torch.zeros(sizes[0], 251).type(torch.LongTensor).to(self.device)
            # predicted_next_word = transcripts[:,0].type(torch.LongTensor)
            # predicted_next_word = torch.zeros(sizes[0]).type(torch.LongTensor)
        output_logits = torch.zeros((sizes[0], sizes[1], self.vocab_size))

        attentions_across_timesteps = []

        for timestep in range(sizes[1]):
            if timestep == 0:
                context = torch.zeros((sizes[0], 1, self.attention_size)).to(self.device)
                query = torch.zeros(sizes[0], self.attention_size).to(self.device)

            teacher_forcing_chooser = np.random.random_sample()
            # teacher_forcing_chooser = 0

            if testing:
                if timestep == 0:
                    hidden_a, cell_a = self.lstm_a(
                        torch.cat((self.embedding(predicted_next_word.to(self.device)).unsqueeze(1), context),
                                  2).squeeze(1))
                else:
                    hidden_a, cell_a = self.lstm_a(
                        torch.cat((self.embedding(predicted_next_word.to(self.device)).unsqueeze(1), context),
                                  2).squeeze(1), (hidden_a, cell_a))
            elif not testing:
                if timestep == 0:
                    hidden_a, cell_a = self.lstm_a(
                        torch.cat((transcripts[:, timestep:timestep + 1], context), 2).squeeze(1))
                elif teacher_forcing_chooser >= self.teacher_forcing_prob:
                    hidden_a, cell_a = self.lstm_a(
                        torch.cat((self.embedding(predicted_next_word.to(self.device)).unsqueeze(1), context),
                                  2).squeeze(1), (hidden_a, cell_a))
                else:
                    hidden_a, cell_a = self.lstm_a(
                        torch.cat((transcripts[:, timestep:timestep + 1], context), 2).squeeze(1), (hidden_a, cell_a))

            if timestep == 0:
                hidden_b, cell_b = self.lstm_b(hidden_a)
            else:
                hidden_b, cell_b = self.lstm_b(hidden_a, (hidden_b, cell_b))

            hidden_b = self.drop(hidden_b)

            energy = torch.bmm(keys, query.unsqueeze(2))
            attention = F.softmax(energy, dim=1)
            mask = torch.zeros_like(attention)
            for batch_index, output_length in enumerate(output_lengths.cpu().numpy()):
                mask[batch_index, :, :output_length] = 1
            masked_attention = F.normalize(attention * mask, p=1)
            # if float('%.3f' % (masked_attention[np.random.randint(0, masked_attention.size()[0]-1)].sum().item())) != 1:
            #     print("Attention isn't summing to 1 across row")
            context = torch.bmm(masked_attention.squeeze(2).unsqueeze(1), values)

            attentions_across_timesteps.append(masked_attention[0].cpu().detach().numpy())

            # project from lstm output to probability over vocab
            # logits = self.probs_linear(torch.cat((hidden_b, context.squeeze(1)), dim=1))
            logits = torch.transpose(torch.matmul(self.embedding.weight.t(), torch.transpose(torch.cat((hidden_b, context.squeeze(1)), dim=1), 0, 1)), 0, 1)

            if not testing:
                gumbel_noise = torch.FloatTensor(np.random.gumbel(size=logits.size())).to(self.device)
                noisy_logits = logits + gumbel_noise
                predicted_next_word = F.softmax(noisy_logits, dim=1)
            else:
                predicted_next_word = F.softmax(logits, dim=1)

            output_logits[:, timestep] = logits

            # project from lstm_output to query
            query = F.softmax(self.query_linear(hidden_b), dim=1)

        if not testing:
            return output_logits, np.stack(attentions_across_timesteps).squeeze(2)
        else:
            return output_logits

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.LSTM or type(m) == nn.LSTMCell:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                upper = 1/np.sqrt(m.hidden_size)
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
