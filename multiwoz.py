import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from util import *


class MultiWoz(Dataset):
    def __init__(self, root, word_vectors, ontology, domains, max_utterance_length, max_turn_length, vector_dimension):
        self.root = root
        self.word_vectors = word_vectors
        self.ontology = ontology
        self.domains = domains
        self.max_utterance_length = max_utterance_length
        self.max_turn_length = max_turn_length
        self.vector_dimension = vector_dimension
        self.dialogues, _ = load_woz_data(root, word_vectors, ontology, domains, max_utterance_length, vector_dimension)

    def __getitem__(self, index):
        (num_turn, user_vecs, sys_vecs, turn_labels, turn_domain_labels) = self.dialogues[index]
        user_uttr = np.zeros((self.max_turn_length, self.max_utterance_length, self.vector_dimension), dtype='float32')
        sys_uttr = np.zeros((self.max_turn_length, self.max_utterance_length, self.vector_dimension), dtype='float32')
        user_uttr_len = np.zeros(self.max_turn_length, dtype='int32')
        sys_uttr_len = np.zeros(self.max_turn_length, dtype='int32')
        labels = np.zeros((self.max_turn_length, len(self.ontology)), dtype='float32')
        domain_labels = np.zeros((self.max_turn_length, len(self.ontology)), dtype='float32')

        for i in range(num_turn):
            user_uttr_len[i] = user_vecs[i].shape[0]
            sys_uttr_len[i] = sys_vecs[i].shape[0]
            user_uttr[i, :user_uttr_len[i], :] = user_vecs[i]
            sys_uttr[i, :sys_uttr_len[i], :] = sys_vecs[i]
            labels[i] = turn_labels[i]
            domain_labels[i] = turn_domain_labels[i]

        return num_turn, user_uttr, sys_uttr, user_uttr_len, sys_uttr_len, labels, domain_labels

    def __len__(self):
        return len(self.dialogues)


def collate_fn(data):
    num_turns, user_uttrs, sys_uttrs, user_uttr_lens, sys_uttr_lens, turn_labels, turn_domain_labels = zip(*data)
    num_turns = torch.tensor(num_turns)
    user_uttrs = torch.from_numpy(np.array(user_uttrs))
    sys_uttrs = torch.from_numpy(np.array(sys_uttrs))
    user_uttr_lens = torch.from_numpy(np.array(user_uttr_lens))
    sys_uttr_lens = torch.from_numpy(np.array(sys_uttr_lens))
    turn_labels = torch.from_numpy(np.array(turn_labels))
    turn_domain_labels = torch.from_numpy(np.array(turn_domain_labels))

    return num_turns, user_uttrs, sys_uttrs, user_uttr_lens, sys_uttr_lens, turn_labels, turn_domain_labels
