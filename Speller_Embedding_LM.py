import os
import pickle
import sys
import time
from collections import OrderedDict

import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import train_collate, Train_Dataset
from dropout import WeightDrop, LockedDropout
from linear_model import init_weights


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

    def forward(self, inputs):
        output_logits = self.decoder.forward(inputs)
        return output_logits


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

    def forward(self):
        pass

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
        self.embedding = nn.Embedding(self.vocab_size, self.decoder_dimension)
        self.modules.append(self.embedding)
        self.lstm_a = nn.LSTMCell(self.decoder_dimension+self.attention_size, self.hidden_size)
        self.modules.append(self.lstm_a)
        self.lstm_b = nn.LSTMCell(self.hidden_size, self.decoder_dimension-self.attention_size)
        self.modules.append(self.lstm_b)
        self.drop = nn.Dropout(0.05)
        self.modules.append(self.drop)
        self.query_linear = nn.Linear(self.decoder_dimension-self.attention_size, self.attention_size)
        self.probs_linear = nn.Linear(self.decoder_dimension, self.vocab_size)
        self.modules.append(self.query_linear)
        self.modules.append(self.probs_linear)
        self.net = nn.ModuleList(self.modules)

        self.probs_linear.weight = self.embedding.weight

    def forward(self, inputs):
        sizes = list(inputs.size())
        inputs = np.ndarray(shape=(sizes[0], sizes[1]))
        inputs.fill(0)
        inputs = torch.LongTensor(inputs).to(self.device)

        output_logits = torch.zeros((sizes[0], sizes[1], self.vocab_size)).to(self.device)

        context = torch.zeros((sizes[0], 1, self.attention_size)).to(self.device)

        for timestep in range(sizes[1]):
            if timestep == 0:
                hidden_a, cell_a = self.lstm_a(
                    torch.cat((self.embedding(inputs[:,timestep].to(self.device)).unsqueeze(1), context),
                              2).squeeze(1))
                hidden_b, cell_b = self.lstm_b(hidden_a)
            else:
                hidden_a, cell_a = self.lstm_a(
                    torch.cat((self.embedding(inputs[:,timestep].to(self.device)).unsqueeze(1), context),
                              2).squeeze(1), (hidden_a, cell_a))
                hidden_b, cell_b = self.lstm_b(hidden_a, (hidden_b, cell_b))

            hidden_b = self.drop(hidden_b)

            # project from lstm output to probability over vocab
            logits = self.probs_linear(torch.cat((hidden_b, context.squeeze(1)), dim=1))

            output_logits[:, timestep] = logits

        return output_logits

class SpellerLMTrainer:
    def __init__(self):
        self.batch_size = 64
        self.embedding_size = 256
        self.attention_size = 128
        self.num_workers = 8
        self.train_data_params = {'batch_size': self.batch_size,
                                  'shuffle': True,
                                  'num_workers': self.num_workers,
                                  'pin_memory': True,
                                  'collate_fn': train_collate}
        self.val_data_params = {'batch_size': self.batch_size,
                                'shuffle': False,
                                'num_workers': self.num_workers,
                                'pin_memory': True,
                                'collate_fn': train_collate}

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        with open('data/character_dict.pkl', 'rb') as file:
            self.character_to_index_dict = pickle.load(file)
            self.index_to_character_dict = dict(map(reversed, self.character_to_index_dict.items()))
            self.vocab_size = len(self.character_to_index_dict)

    def train(self, epochs, gpu, model_path=None):
        device = torch.device('cuda' if gpu else 'cpu')

        net = Seq2Seq(self.batch_size, self.embedding_size, self.attention_size, self.vocab_size, device)

        if not gpu:
            self.batch_size = 2
            self.train_data_params['batch_size'] = self.batch_size
            self.val_data_params['batch_size'] = self.batch_size
            if model_path is None:
                net.apply(init_weights)
            else:
                net.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            if model_path is None:
                net.apply(init_weights)
                print("Initialized model weights.")
            elif model_path is not None:
                net.load_state_dict(torch.load(model_path))
                print("Loaded saved model.")

        net = net.to(device)

        training_gen_start_time = time.time()
        print('Creating the training dataset.')
        if not gpu:
            training_dataset = Train_Dataset("data/small_train.npy", "data/small_indexed_train_transcripts.npy")
        else:
            training_dataset = Train_Dataset("data/train.npy", "data/indexed_train_transcripts.npy")
        training_generator = DataLoader(training_dataset, **self.train_data_params)
        print('Creating the training dataset took {:0.2f} seconds'.format(time.time() - training_gen_start_time))
        print(
            'Num training batches per epoch is ' + repr(math.ceil(len(training_dataset) / self.batch_size)) + '.')

        print('Creating the validation dataset.')
        validation_dataset = Train_Dataset("data/dev.npy", "data/indexed_dev_transcripts.npy")
        validation_generator = DataLoader(validation_dataset, **self.val_data_params)

        basepath = os.getcwd()
        backup_path = basepath + "/pretrained_models/" + repr(training_gen_start_time)
        os.mkdir(backup_path)

        optimizer = torch.optim.Adam(net.parameters(), 1e-4, weight_decay=0)

        print("Number of trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
        print('Beginning training.')
        for epoch in range(epochs):
            start = time.time()
            print('Device is ' + repr(device) + ', and start time is ' + time.ctime(start))
            net = net.to(device)
            count = 0
            cumulative_train_loss = 0.0
            cumulative_perplexity = 0.0

            for batch in training_generator:
                frames, transcripts, frame_lengths, transcript_lengths, unsort_index = batch
                net.train()
                if (count % 50 == 0 and count > 0):
                    print(
                        "Training on {:} batches has taken {:.2f} minutes. Average training loss is {:.2f}. Average perplexity is {:.2f}."
                        .format(count, (time.time() - start) / 60, cumulative_train_loss / count, cumulative_perplexity / count))

                output_logits = net(transcripts[:, :-1].to(device))
                targets = transcripts[:, 1:].to(device)
                num_chars = np.sum([o.size()[0] for o in targets])
                loss = self.criterion(torch.cat(tuple(output_logits), 0).to(device),
                                      torch.cat(tuple(targets), 0).to(device))
                mask = torch.zeros_like(targets)
                for batch_num, length in enumerate(transcript_lengths):
                    mask[batch_num, :length-1] = 1
                loss = (loss * torch.cat(tuple(mask), 0).type(torch.FloatTensor).to(device)).sum() / self.batch_size
                perplexity = math.exp(loss / num_chars * self.batch_size)
                cumulative_train_loss += loss.item()
                cumulative_perplexity += perplexity
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.25)
                optimizer.step()
                optimizer.zero_grad()
                count += 1

            print("After epoch ", repr(epoch))
            print("Training loss: {:.2f}".format(cumulative_train_loss / count))
            print("Training perplexity: {:.2f}".format(cumulative_perplexity / count))

            net.eval()
            cumulative_val_loss = 0.0
            cumulative_val_perplexity = 0.0
            val_count = 0
            with torch.set_grad_enabled(False):
                for batch in validation_generator:
                    frames, transcripts, frame_lengths, transcript_lengths, unsort_index = batch
                    output_logits = net(transcripts[:, :-1].to(device))
                    targets = transcripts[:, 1:].to(device)
                    num_chars = np.sum([o.size()[0] for o in targets])
                    loss = self.criterion(torch.cat(tuple(output_logits), 0).to(device),
                                          torch.cat(tuple(targets), 0).to(device))
                    mask = torch.zeros_like(targets)
                    for batch_num, length in enumerate(transcript_lengths):
                        mask[batch_num, :length - 1] = 1
                    loss = (loss * torch.cat(tuple(mask), 0).type(torch.FloatTensor).to(device)).sum() / self.batch_size
                    perplexity = math.exp(loss / num_chars * self.batch_size)
                    cumulative_val_loss += loss.item()
                    cumulative_val_perplexity += perplexity
                    val_count += 1
            print("Validation loss: {:.2f}".format(cumulative_val_loss / val_count))
            print("Validation perplexity: {:.2f}".format(cumulative_val_perplexity / val_count))

            stop = time.time()
            print("This epoch took {:.2f} minutes.".format((stop - start) / 60))
            backup_file = backup_path + "/epoch_{:}_trainLoss_{:.2f}_trainPerp_{:.2f}_valLoss_{:.2f}_valPerp_{:.2f}.pt".format(
                epoch, (cumulative_train_loss / count), (cumulative_perplexity / count), (cumulative_val_loss / val_count),
                (cumulative_val_perplexity / val_count))
            torch.save(net.state_dict(), backup_file)

        net = net.cpu()
        print("Finished training.")


def main():
    tester = SpellerLMTrainer()
    tester.train(10, False, None)



if __name__ == '__main__':
    main()