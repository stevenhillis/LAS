import torch
import numpy as np
from torch.utils import data
import torch.nn.utils.rnn as rnn

class Train_Dataset(data.Dataset):
    def __init__(self, feature_data_path, transcript_data_path):
        self.feature_data = np.load(feature_data_path, fix_imports=True, encoding='bytes', allow_pickle=True)
        self.transcript_data = np.load(transcript_data_path, fix_imports=True, encoding='bytes', allow_pickle=True)

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, item):
        return torch.Tensor(self.feature_data[item]), torch.Tensor(self.transcript_data[item])

class Test_Dataset(data.Dataset):
    def __init__(self, feature_data_path):
        self.feature_data = np.load(feature_data_path, fix_imports=True, encoding='bytes', allow_pickle=True)

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, item):
        return torch.Tensor(self.feature_data[item])

def train_collate(batch):
    frames = [instance[0] for instance in batch]
    transcripts = np.array([instance[1].numpy() for instance in batch])

    frame_lengths = torch.LongTensor([len(frame) for frame in frames])
    padded_frames = rnn.pad_sequence(frames, batch_first=True)
    sorted_frame_lengths, permute_index = frame_lengths.sort(0, descending=True)
    sorted_padded_frames = padded_frames[permute_index.data.numpy()]

    sorted_transcripts = transcripts[permute_index.data.numpy()]
    sorted_transcript_lengths = [len(t) for t in sorted_transcripts]
    sorted_transcripts = [torch.LongTensor(transcript) for transcript in sorted_transcripts]
    sorted_padded_transcripts = rnn.pad_sequence(sorted_transcripts, batch_first=True)

    _, unpermute_index = permute_index.sort(0)

    return sorted_padded_frames, sorted_padded_transcripts, sorted_frame_lengths, sorted_transcript_lengths, unpermute_index

def test_collate(frames):
    frame_lengths = torch.LongTensor([len(frame) for frame in frames])
    padded_frames = rnn.pad_sequence(frames, batch_first=True)

    sorted_frame_lengths, permute_index = frame_lengths.sort(0, descending=True)
    sorted_padded_frames = padded_frames[permute_index.data.numpy()]
    _, unpermute_index = permute_index.sort(0)

    return sorted_padded_frames, sorted_frame_lengths, unpermute_index

