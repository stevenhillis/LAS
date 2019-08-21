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
from baseline_model import init_weights


class Seq2Seq(nn.Module):
    def __init__(self, base, out_dim, device='cuda'):
        super().__init__()
        self.encoder = Encoder(base)
        self.lstm_dim = base*2
        self.decoder = Decoder(out_dim=out_dim, lstm_dim=self.lstm_dim)
        self.out_dim = out_dim
        self.device = device

    def forward(self, words):
        batch_size, max_len = words.shape[0], words.shape[1]
        output_logits = torch.zeros((batch_size, max_len,self.out_dim)).to(self.device)
        word, hidden1, cell1, hidden2, cell2 = words[:, 0], None, None, None, None
        for t in range(1, max_len):
            word_vec, hidden1, cell1, hidden2, cell2 = self.decoder(word, torch.zeros((batch_size, self.lstm_dim)).to(self.device), hidden1, cell1, hidden2, cell2,
                                                                first_step=(t == 1))
            output_logits[:, t] = word_vec
        return output_logits

class Encoder(nn.Module):
    def __init__(self, base=64):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(40, base, bidirectional = True, batch_first=True)
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

    def forward(self):
        pass

class Decoder(nn.Module):
    def __init__(self, out_dim, lstm_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(out_dim, lstm_dim)
        self.lstm1 = nn.LSTMCell(lstm_dim*2, lstm_dim)
        self.lstm2 = nn.LSTMCell(lstm_dim, lstm_dim)
        self.drop = nn.Dropout(0.05)
        self.fc = nn.Linear(lstm_dim, out_dim)
        self.embed.weight = self.fc.weight
        self.fc.weight = self.embed.weight


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

class SpellerLMTrainer:
    def __init__(self):
        self.batch_size = 64
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

        net = Seq2Seq(base=64, out_dim=self.vocab_size, device=device)

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

        optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=0)

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
                loss = self.criterion(torch.cat(tuple(output_logits), 0).to(device),torch.cat(tuple(targets), 0).to(device))
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
    tester.train(10, True, None)


if __name__ == '__main__':
    main()