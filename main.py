import torch
import numpy as np
import json
import click
from model import max_utterance_length, vector_dimension, max_turn_length
from util import *
from multiwoz import *


TRAINING_FILE = "data/train.json"
VALIDATION_FILE = "data/validate.json"
TESTING_FILE = "data/test.json"
ONTOLOGY_FILE = "data/ontology.json"
WORD_VECTORS_FILE = "word-vectors/paragram_300_sl999.txt"
DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']

num_slots = 0


@click.group()
def main():
    pass


@main.command()
def train():
    # Load word vectors
    word_vectors = load_word_vectors(WORD_VECTORS_FILE)

    # Load ontology
    ontology, ontology_vectors, slots = load_ontoloty(ONTOLOGY_FILE, word_vectors, DOMAINS)

    # Load dialogues
    dataset = MultiWoz(TRAINING_FILE, word_vectors, ontology, DOMAINS, max_utterance_length, max_turn_length, vector_dimension)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3, collate_fn=collate_fn)


@main.command()
def test():
    print("test")


if __name__ == '__main__':
    main()
