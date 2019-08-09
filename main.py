import numpy as np
import json
import click
from collections import OrderedDict
from model import max_utterance_length, vector_dimension, max_no_turns
from util import *


TRAINING_FILE = "data/train.json"
VALIDATION_FILE = "data/validate.json"
TESTING_FILE = "data/test.json"
ONTOLOGY_FILE = "data/ontology.json"
WORD_VECTORS_FILE = "word-vectors/paragram_300_sl999.txt"
DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']

num_slots = 0
booking_slots = {}


def load_word_vectors(path):
    """
    Load the pretrained word vectors
    :param path:
    :return:
    """
    word_vectors = {}
    print("[Info] Loading pretrained word vectors")
    with open(path, mode='r', encoding='utf8') as f:
        for l in f:
            l = l.split(" ", 1)
            key = l[0]
            word_vectors[key] = np.fromstring(l[1], dtype="float32", sep=" ")
    print("[Info] The vocabulary contains about %d word vectors" % (len(word_vectors)))
    return normalise_word_vectors(word_vectors)


def load_ontoloty(path, word_vectors):
    """
    Load the ontology data
    :param path:
    :param word_vectors:
    :return:
    """
    global num_slots
    print("[Info] Loading ontology")
    data = json.load(open(path, mode='r', encoding='utf8'), object_pairs_hook=OrderedDict)
    slot_values = []
    ontology = []
    slots_values = []
    ontology_vectors = []
    for slots in data:
        [domain, slot] = slots.split("-")
        if domain not in DOMAINS or slot == "name":
            continue
        values = data[slots]
        if "book" in slot:
            [slot, value] = slot.split(" ")
            booking_slots[domain+'-'+value] = values
            values = [value]
        elif slot == "departure" or slot == "destination":
            values = ["place"]
        domain_vec = np.sum(process_text(domain, word_vectors), axis=0)
        if domain not in word_vectors:
            word_vectors[domain.replace(" ", "")] = domain_vec
        slot_vec = np.sum(process_text(slot, word_vectors), axis=0)
        # if domain+'-'+slot not in slots_values:
        #     slots_values.append(domain+'-'+slot)
        slots_values.append(domain+'-'+slot)


        if slot not in word_vectors:
            word_vectors[slot.replace(" ", "")] = slot_vec
        slot_values.append(len(values))
        for value in values:
            ontology.append(domain + '-' + slot + '-' + value)
            value_vec = np.sum(process_text(value, word_vectors, print_mode=True), axis=0)
            if value not in word_vectors:
                word_vectors[value.replace(" ", "")] = value_vec
            ontology_vectors.append(np.concatenate((domain_vec, slot_vec, value_vec)))

    num_slots = len(slots_values)
    print("[Info] We have about %d values" % len(ontology))
    print("[Info] The slots in this ontology:")
    print(', '.join(slots_values))
    return ontology, np.asarray(ontology_vectors, dtype='float32'), slot_values


def load_woz_data(path, word_vectors, ontology):
    print("[Info] Loading woz data from file")
    data = json.load(open(path, mode='r', encoding='utf8'))

    dialogues = []
    actual_dialogues = []
    for dialogue in data:
        turn_ids = []
        for key in dialogue.keys():
            if key.isdigit():
                turn_ids.append(int(key))
        turn_ids.sort()
        num_turns = len(turn_ids)
        user_vecs = []
        sys_vecs = []
        turn_labels = []
        turn_domain_labels = []
        add = False
        good = True
        pre_sys = np.zeros([max_utterance_length, vector_dimension], dtype="float32")
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_v, sys_v, labels, domain_labels = process_turn(turn, word_vectors, ontology)
            if good and (user_v.shape[0] > max_utterance_length or pre_sys.shape[0] > max_utterance_length):
                good = False
                break
            user_vecs.append(user_v)
            sys_vecs.append(pre_sys)
            turn_labels.append(labels)
            turn_domain_labels.append(domain_labels)
            if not add and sum(labels) > 0:
                add = True
            pre_sys = sys_v
        if add and good:
            dialogues.append((num_turns, user_vecs, sys_vecs, turn_labels, turn_domain_labels))
            actual_dialogues.append(dialogue)
    print("[Info] The data contains about %d dialogues" % len(dialogues))
    return dialogues, actual_dialogues


def process_text(text, word_vectors, ontology=None, print_mode=False):
    """
    Process a line/sentence converting words to feature vectors
    :param text:
    :param word_vectors:
    :param ontology:
    :param print_mode:
    :return:
    """
    text = text.replace("(", "").replace(")", "").replace('"', "").replace(u"’", "'").replace(u"‘", "'")
    text = text.replace("\t", "").replace("\n", "").replace("\r", "").strip().lower()
    text = text.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('-', ' ').replace('/', ' / ').replace(':', ' ')
    if ontology:
        for slot in ontology:
            [domain, slot, value] = slot.split('-')
            text.replace(domain, domain.replace(" ", "")) \
                .replace(slot, slot.replace(" ", "")) \
                .replace(value, value.replace(" ", ""))

    words = text.split()

    vectors = []
    for word in words:
        word = word.replace("'", "").replace("!", "")
        if word == "":
            continue
        if word not in word_vectors:
            length = len(word)
            for i in range(1, length)[::-1]:
                if word[:i] in word_vectors and word[i:] in word_vectors:
                    vec = word_vectors[word[:i]] + word_vectors[word[i:]]
                    break
            else:
                vec = xavier_vector(word)
                word_vectors[word] = vec
                if print_mode:
                    print("[Info] Adding new word: %s" % word)
        else:
            vec = word_vectors[word]
        vectors.append(vec)
    return np.asarray(vectors, dtype='float32')


def process_turn(turn, word_vectors, ontology):
    user_input = turn['user']['text']
    sys_res = turn['system']
    state = turn['user']['belief_state']
    user_v = process_text(user_input, word_vectors, ontology)
    sys_v = process_text(sys_res, word_vectors, ontology)
    labels = np.zeros(len(ontology), dtype='float32')
    domain_labels = np.zeros(len(ontology), dtype='float32')
    for domain in state:
        if domain not in DOMAINS:
            continue
        slots = state[domain]['semi']
        domain_mention = False
        for slot in slots:
            if slot == 'name':
                continue
            value = slots[slot]
            if "book" in slot:
                [slot, value] = slot.split(" ")
            if value != '' and value != 'corsican':
                if slot == "destination" or slot == "departure":
                    value = "place"
                elif value == '09;45':
                    value = '09:45'
                elif 'alpha-milton' in value:
                    value = value.replace('alpha-milton', 'alpha milton')
                elif value == 'east side':
                    value = 'east'
                elif value == ' expensive':
                    value = 'expensive'
                labels[ontology.index(domain + '-' + slot + '-' + value)] = 1
                domain_mention = True
        if domain_mention:
            for idx, slot in enumerate(ontology):
                if domain in slot:
                    domain_labels[idx] = 1

    return user_v, sys_v, labels, domain_labels


@click.group()
def main():
    pass


@main.command()
def train():
    # Load word vectors
    word_vectors = load_word_vectors(WORD_VECTORS_FILE)

    # Load ontology
    ontology, ontology_vectors, slots = load_ontoloty(ONTOLOGY_FILE, word_vectors)

    # Load dialogues
    dialogues, _ = load_woz_data(TRAINING_FILE, word_vectors, ontology)
    val_dialogues, _ = load_woz_data(VALIDATION_FILE, word_vectors, ontology)


@main.command()
def test():
    print("test")


if __name__ == '__main__':
    main()
