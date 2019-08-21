import pickle

import numpy as np
import functools
import operator

# Run only once on train to write list of all characters in vocab to file. Zeroeth should be #
# Write out a new train, dev, and test that have numbers instead of characters
# TODO: add hash before and after?
def main():
    character_dict = dict()
    character_dict["#"] = 0
    transcripts = np.load('data/train_transcripts.npy', allow_pickle=True)
    transcripts = [[word.decode("utf-8") for word in transcript]for transcript in transcripts]
    transcripts = [' '.join([*transcript]) for transcript in transcripts]
    transcripts = [["#", *transcript, "#"] for transcript in transcripts]

    indexed_transcripts = [[([character_dict.setdefault(char, len(character_dict)) for char in list(word)]) for word in transcript] for transcript in transcripts]
    indexed_transcripts = [functools.reduce(operator.iconcat, indexed_transcript, []) for indexed_transcript in indexed_transcripts]
    np.save("data/indexed_train_transcripts.npy", indexed_transcripts)

    transcripts = np.load('data/dev_transcripts.npy', allow_pickle=True)
    transcripts = [[word.decode("utf-8") for word in transcript]for transcript in transcripts]
    transcripts = [' '.join([*transcript]) for transcript in transcripts]
    transcripts = [["#", *transcript, "#"] for transcript in transcripts]
    indexed_transcripts = [[([character_dict.setdefault(char, len(character_dict)) for char in list(word)]) for word in transcript] for transcript in transcripts]
    indexed_transcripts = [functools.reduce(operator.iconcat, indexed_transcript, []) for indexed_transcript in indexed_transcripts]
    np.save("data/indexed_dev_transcripts.npy", indexed_transcripts)

    with open("data/character_dict.pkl", "wb") as file:
        pickle.dump(character_dict, file)

    small_indexed_transcripts = indexed_transcripts[:20]
    np.save("data/small_indexed_train_transcripts.npy", small_indexed_transcripts)

    train = np.load('data/train.npy', encoding='bytes', allow_pickle=True)
    small_train = train[:20]
    np.save("data/small_train.npy", small_train)


if __name__ == '__main__':
    main()