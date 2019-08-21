import os

import math
import pickle
import time

# pip install python-Levenshtein
import Levenshtein
import torch

import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn

from dataloader import Train_Dataset, train_collate
from dataloader import test_collate, Test_Dataset
from torch.utils.data import DataLoader
from baseline_model import init_weights, Seq2Seq, plot_grad_flow

import matplotlib.pyplot as plt


class ListenAttendSpell:
    def __init__(self):
        self.batch_size = 100
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


    def train(self, epochs, gpu, model_path=None, lr=1e-4, weight_decay=1e-5):
        device = torch.device('cuda' if gpu else 'cpu')

        # net = Seq2Seq(self.batch_size, self.embedding_size, self.attention_size, self.vocab_size, device)
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
        gradient_path = basepath + "/gradients/" + repr(training_gen_start_time)
        os.mkdir(gradient_path)
        attention_path = basepath + "/attention_plots/" + repr(training_gen_start_time)
        os.mkdir(attention_path)
        backup_path = basepath + "/models/" + repr(training_gen_start_time)
        os.mkdir(backup_path)

        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01, verbose=True)
        teacher_forcing_prob = 0.50

        print("Number of trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
        print('Beginning training.')
        for epoch in range(epochs):
            if epoch > 0 and epoch % 5 == 0 and epoch < 21:
                teacher_forcing_prob -= 0.05
            start = time.time()
            print('Device is ' + repr(device) + ', tf_prob is ' + repr(teacher_forcing_prob) + ' and start time is ' + time.ctime(start))
            # print('Device is ' + repr(device) + ', and start time is ' + time.ctime(start))
            net = net.to(device)

            count = 0
            cumulative_train_loss = 0.0
            cumulative_perplexity = 0.0

            for batch in training_generator:
                frames, transcripts, frame_lengths, transcript_lengths, unsort_index = batch
                net.train()
                if (count % 35 == 0 and count > 0):
                    print(
                        "Training on {:} batches has taken {:.2f} minutes. Average training loss is {:.2f}. Average perplexity is {:.2f}."
                        .format(count, (time.time() - start) / 60, cumulative_train_loss / count, cumulative_perplexity / count))
                    # print("\tIn that last batch, the average query range over timesteps for the first utterance was {:.4f}".format(av_query_range_first_utt))
                frames, transcripts, frame_lengths = frames.to(device), transcripts.to(device), frame_lengths.to(device)
                frames = rnn.pack_padded_sequence(frames, frame_lengths, batch_first=True)
                output, attention_across_timesteps = net(frames, transcripts, TF=teacher_forcing_prob)
                num_chars = np.sum([o.size()[0] for o in transcripts])
                transcripts = transcripts.to(device)
                loss = self.criterion(torch.cat(tuple(output), 0).to(device), torch.cat(tuple(transcripts), 0).to(device))
                mask = torch.zeros_like(transcripts)
                for batch_num, length in enumerate(transcript_lengths):
                    mask[batch_num, :length] = 1
                loss = (loss * torch.cat(tuple(mask), 0).type(torch.FloatTensor).to(device)).sum() / self.batch_size
                perplexity = math.exp(loss / num_chars * self.batch_size)
                cumulative_train_loss += loss.item()
                cumulative_perplexity += perplexity

                loss.backward()
                if count % 50 == 0:
                    plot_grad_flow(net.named_parameters(), gradient_path, epoch, count)
                nn.utils.clip_grad_norm_(net.parameters(), 0.25)
                optimizer.step()
                optimizer.zero_grad()

                if count == 0:
                    fig = plt.figure()
                    plt.xlabel("Frame lengths // 8")
                    plt.ylabel("Timesteps")
                    plt.imshow(attention_across_timesteps)
                    fig.savefig(attention_path + "/epoch{:}.png".format(epoch))
                    plt.close()

                count += 1

            print("After epoch ", repr(epoch))
            print("Training loss: {:.2f}".format(cumulative_train_loss / count))
            print("Training perplexity: {:.2f}".format(cumulative_perplexity / count))

            net.eval()
            cumulative_edit_distance = 0.0
            cumulative_val_loss = 0.0
            cumulative_val_perplexity = 0.0
            val_start = time.time()
            val_count = 0
            with torch.set_grad_enabled(False):
                for batch in validation_generator:
                    frames, transcripts, frame_lengths, transcript_lengths, unsort_index = batch
                    frames, transcripts, frame_lengths = frames.to(device), transcripts.to(device), frame_lengths.to(device)
                    frames = rnn.pack_padded_sequence(frames, frame_lengths, batch_first=True)
                    seed = np.ndarray(shape=(transcripts.size()[0], 1))
                    seed.fill(self.character_to_index_dict["#"])
                    seed = torch.LongTensor(seed).to(device)
                    output = net(frames, seed, TF=0)

                    output = output[:, :transcripts.size()[1]]
                    num_chars = np.sum([o.size()[0] for o in transcripts])
                    loss = self.criterion(torch.cat(tuple(output), 0).to(device),
                                          torch.cat(tuple(transcripts), 0).to(device))
                    mask = torch.zeros_like(transcripts)
                    for batch_num, length in enumerate(transcript_lengths):
                        mask[batch_num, :length] = 1
                    loss = (loss * torch.cat(tuple(mask), 0).type(torch.FloatTensor).to(device)).sum() / self.batch_size
                    perplexity = math.exp(loss / num_chars * self.batch_size)
                    cumulative_val_loss += loss.item()
                    cumulative_val_perplexity += perplexity

                    output = output.cpu().detach()
                    output = [np.argmax(output[batch_num, :length].numpy(), axis=1) for batch_num, length in enumerate(transcript_lengths)]
                    output = [[self.index_to_character_dict[char.item()] for char in o] for o in output]
                    output = ''.join([item for sublist in output for item in sublist])
                    transcripts = transcripts.cpu().detach()
                    transcripts = [transcripts[batch_num, :length].numpy() for batch_num, length in enumerate(transcript_lengths)]
                    transcripts = [[self.index_to_character_dict[char.item()] for char in t] for t in transcripts]
                    transcripts = ''.join([item for sublist in transcripts for item in sublist])
                    if epoch == 0 and val_count == 0:
                        with open("val_transcript_batch_0.txt", "w+") as out_file:
                            out = "transcripts:\n" + transcripts + "\n"
                            out_file.write(out)
                    if val_count == 0:
                        with open("val_transcript_batch_0.txt", "a+") as out_file:
                            out = "outputs:\n" + output + "\n"
                            out_file.write(out)

                    edit_distance = Levenshtein.distance(output, transcripts) / self.batch_size
                    cumulative_edit_distance += edit_distance
                    val_count += 1

            print("Validation took {:.2f} minutes.".format((time.time() - val_start)/60))
            print("Validation edit distance: {:.2f}".format(cumulative_edit_distance / val_count))
            print("Validation loss: {:.2f}".format(cumulative_val_loss / val_count))
            print("Validation perplexity: {:.2f}".format(cumulative_val_perplexity / val_count))

            scheduler.step(cumulative_edit_distance / val_count)

            stop = time.time()
            print("This epoch took {:.2f} minutes.".format((stop - start) / 60))
            backup_file = backup_path + "/epoch_{:}_trainLoss_{:.2f}_trainPerp_{:.2f}_valEdit_{:.2f}_valLoss_{:.2f}_valPerp_{:.2f}.pt".format(epoch, (cumulative_train_loss / count), (cumulative_perplexity / count), (cumulative_edit_distance / val_count), (cumulative_val_loss / val_count), (cumulative_val_perplexity / val_count))
            torch.save(net.state_dict(), backup_file)

        net = net.cpu()
        print("Finished training.")

    def test(self, model_path, gpu):
        device = torch.device('cuda' if gpu else 'cpu')
        batch_size = 1

        net = Seq2Seq(base=64, out_dim=self.vocab_size, device=device)
        if gpu:
            net.load_state_dict(torch.load(model_path))
        else:
            net.load_state_dict(torch.load(model_path, map_location='cpu'))

        net = net.to(device)
        net.eval()

        print('Creating the testing generator.')
        test_data_params = {'batch_size': batch_size,
                            'shuffle': False,
                            'num_workers': 8,
                            'pin_memory': True,
                            'collate_fn': test_collate}
        test_dataset = Test_Dataset("data/test.npy")
        test_generator = DataLoader(test_dataset, **test_data_params)
        print('Num utterances to test on is ' + repr(math.ceil(len(test_dataset))) + '.')
        print('Beginning testing.')
        count = 0
        out_line = 0
        start = time.time()
        out_file = open("output.csv", "w")
        out_file.write("Id,Predicted\n")
        out_file.close()

        random = False

        with torch.set_grad_enabled(False):
            for batch in test_generator:
                frames, frame_lengths, unsort_index = batch
                if (count % 100 == 0 and count > 0):
                    print("So far, testing on {:} examples has taken {:.2f} minutes.".format(count,(time.time() - start) / 60))
                frames, frame_lengths = frames.to(device), frame_lengths.to(device)
                frames = rnn.pack_padded_sequence(frames, frame_lengths, batch_first=True)
                best_loss = 1e309
                if random:
                    for i in range(3):
                        seed = np.ndarray(shape=(frame_lengths.size()[0], 1))
                        seed.fill(self.character_to_index_dict["#"])
                        seed = torch.LongTensor(seed).to(device)
                        output = net(frames, seed, TF=0)

                        output = output.cpu().detach()
                        output = output[unsort_index.data.numpy()]
                        output = torch.argmax(output, dim=2)

                        # only works for batch size one
                        if len(np.nonzero(output.detach().cpu().numpy())[1]) < len(output[0])-1:
                            output = output[:,1:np.where(output.detach().cpu().numpy() == 0)[1][0]]
                        else:
                            output = output[:, 1:]

                        output = output.to(device)

                        second_output, _ = net(frames, torch.cat((torch.cat((seed, output), dim=1), seed), dim=1).to(device), TF=1e-10)
                        loss = self.criterion(second_output.squeeze(0), torch.cat((torch.cat((seed, output), dim=1), seed), dim=1).squeeze(0).to(device)).sum()
                        if loss < best_loss:
                            best_loss = loss
                            best_output = output
                else:
                    seed = np.ndarray(shape=(frame_lengths.size()[0], 1))
                    seed.fill(self.character_to_index_dict["#"])
                    seed = torch.LongTensor(seed).to(device)
                    output = net(frames, seed, TF=0)

                    output = output.cpu().detach()
                    output = output[unsort_index.data.numpy()]
                    output = torch.argmax(output, dim=2)

                    # only works for batch size one
                    if len(np.nonzero(output.detach().cpu().numpy())[1]) < len(output[0]) - 1:
                        best_output = output[:, 1:np.where(output.detach().cpu().numpy() == 0)[1][0]]
                    else:
                        best_output = output[:, 1:]

                output = [''.join([self.index_to_character_dict[char.item()] for char in o]) for o in best_output]
                print(output)
                out_file = open("output.csv", "a+")
                for prediction in output:
                    out = repr(out_line) + "," + prediction + "\n"
                    out_file.write(out)
                    out_line += 1
                out_file.close()
                count += 1

        print('Finished testing.')
        net = net.cpu()



