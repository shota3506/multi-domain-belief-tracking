# -*- coding: utf-8 -*-

import string
import numpy as np
import math
import json
from collections import OrderedDict


def hash_string(s):
    return abs(hash(s)) % (10 ** 8)


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt(sum(word_vectors[word] ** 2) + 1e-6)
        word_vectors[word] *= norm
    return word_vectors


def xavier_vector(word, D=300):
    """
    Returns a D-dimensional vector for the word.

    We hash the word to always get the same vector for the given word.
    """

    seed_value = hash_string(word)
    np.random.seed(seed_value)

    neg_value = - math.sqrt(6) / math.sqrt(D)
    pos_value = math.sqrt(6) / math.sqrt(D)

    rsample = np.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = np.linalg.norm(rsample)
    rsample_normed = rsample / norm

    return rsample_normed


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


def load_ontoloty(path, word_vectors, domains):
    print("[Info] Loading ontology")
    data = json.load(open(path, mode='r', encoding='utf8'), object_pairs_hook=OrderedDict)
    slot_values = []
    ontology = []
    slots_values = []
    ontology_vectors = []
    for slots in data:
        [domain, slot] = slots.split("-")
        if domain not in domains or slot == "name":
            continue
        values = data[slots]
        if "book" in slot:
            [slot, value] = slot.split(" ")
            # booking_slots[domain+'-'+value] = values
            values = [value]
        elif slot == "departure" or slot == "destination":
            values = ["place"]
        domain_vec = np.sum(process_text(domain, word_vectors), axis=0)
        if domain not in word_vectors:
            word_vectors[domain.replace(" ", "")] = domain_vec
        slot_vec = np.sum(process_text(slot, word_vectors), axis=0)
        if domain+'-'+slot not in slots_values:
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


def load_woz_data(path, word_vectors, ontology, domains, max_utterance_length, vector_dimension):
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
        num_turn = len(turn_ids)
        user_vecs = []
        sys_vecs = []
        turn_labels = []
        turn_domain_labels = []
        add = False
        good = True
        pre_sys = np.zeros([max_utterance_length, vector_dimension], dtype="float32")
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_v, sys_v, labels, domain_labels = process_turn(turn, word_vectors, ontology, domains)
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
            dialogues.append((num_turn, user_vecs, sys_vecs, turn_labels, turn_domain_labels))
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


def process_turn(turn, word_vectors, ontology, domains):
    user_input = turn['user']['text']
    sys_res = turn['system']
    state = turn['user']['belief_state']
    user_v = process_text(user_input, word_vectors, ontology)
    sys_v = process_text(sys_res, word_vectors, ontology)
    labels = np.zeros(len(ontology), dtype='float32')
    domain_labels = np.zeros(len(ontology), dtype='float32')
    for domain in state:
        if domain not in domains:
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


def process_dialogues(data, ontology, max_no_turns=-1):
    for name, dialogue in data.items():

        keylist = list(range(dialogue['len']))
        no_turns = len(keylist)
        if no_turns > max_no_turns:
            max_no_turns = no_turns
        for i, key in enumerate(keylist):
            turn = dialogue[str(key)]
            belief_state = turn['user']['belief_state']
            for domain in belief_state:
                slots = belief_state[domain]['semi']
                bookings = belief_state[domain]['book']
                for booking in bookings:
                    if booking != "booked" and bookings[booking] != "":
                        slots["book " + booking] = bookings[booking]

                new_slots = {}
                for slot in slots:
                    value = slots[slot]
                    slot, value = clean_domain(domain, slot, value)
                    new_slots[slot] = value
                    assert value != "not mentioned"

                    if value != "":
                        key = domain + "-" + slot
                        if key in ontology:
                            if value not in ontology[key]:
                                ontology[key].append(value)
                        else:
                            ontology[key] = []
                belief_state[domain]['semi'] = new_slots
    return max_no_turns


def clean_text(text):
    text = text.strip()
    text = text.lower()
    text = text.replace(u"’", "'")
    text = text.replace(u"‘", "'")
    text = text.replace("don't", "do n't")
    return text


def clean_domain(domain, slot, value):
    value = clean_text(value)
    if not value:
        value = ''
    elif value == 'not mentioned':
        value = ''
    elif domain == 'attraction':
        if slot == 'name':
            if value == 't':
                value = ''
            if value == 'trinity':
                value = 'trinity college'
        elif slot == 'area':
            if value in ['town centre', 'cent', 'center', 'ce']:
                value = 'centre'
            elif value in ['ely', 'in town', 'museum', 'norwich', 'same area as hotel']:
                value = ""
            elif value in ['we']:
                value = "west"
        elif slot == 'type':
            if value in ['m', 'mus', 'musuem']:
                value = 'museum'
            elif value in ['art', 'architectural']:
                value = "architecture"
            elif value in ['churches']:
                value = "church"
            elif value in ['coll']:
                value = "college"
            elif value in ['concert', 'concerthall']:
                value = 'concert hall'
            elif value in ['night club']:
                value = 'nightclub'
            elif value in ['mutiple sports', 'mutliple sports', 'sports', 'galleria']:
                value = 'multiple sports'
            elif value in ['ol', 'science', 'gastropub', 'la raza']:
                value = ''
            elif value in ['swimmingpool', 'pool']:
                value = 'swimming pool'
            elif value in ['fun']:
                value = 'entertainment'

    elif domain == 'hotel':
        if slot == 'area':
            if value in ['cen', 'centre of town', 'near city center', 'center']:
                value = 'centre'
            elif value in ['east area', 'east side']:
                value = 'east'
            elif value in ['in the north', 'north part of town']:
                value = 'north'
            elif value in ['we']:
                value = "west"
        elif slot == "book day":
            if value == "monda":
                value = "monday"
            elif value == "t":
                value = "tuesday"
        elif slot == 'name':
            if value == 'uni':
                value = 'university arms hotel'
            elif value == 'university arms':
                value = 'university arms hotel'
            elif value == 'acron':
                value = 'acorn guest house'
            elif value == 'ashley':
                value = 'ashley hotel'
            elif value == 'arbury lodge guesthouse':
                value = 'arbury lodge guest house'
            elif value == 'la':
                value = 'la margherit'
            elif value == 'no':
                value = ''
        elif slot == 'internet':
            if value == 'does not':
                value = 'no'
            elif value in ['y', 'free', 'free internet']:
                value = 'yes'
            elif value in ['4']:
                value = ''
        elif slot == 'parking':
            if value == 'n':
                value = 'no'
            elif value in ['free parking']:
                value = 'free'
            elif value in ['y']:
                value = 'yes'
        elif slot in ['pricerange', 'price range']:
            slot = 'price range'
            if value == 'moderately':
                value = 'moderate'
            elif value in ['any']:
                value = "do n't care"
            elif value in ['any']:
                value = "do n't care"
            elif value in ['inexpensive']:
                value = "cheap"
            elif value in ['2', '4']:
                value = ''
        elif slot == 'stars':
            if value == 'two':
                value = '2'
            elif value == 'three':
                value = '3'
            elif value in ['4-star', '4 stars', '4 star', 'four star', 'four stars']:
                value = '4'
        elif slot == 'type':
            if value == '0 star rarting':
                value = ''
            elif value == 'guesthouse':
                value = 'guest house'
            elif value not in ['hotel', 'guest house', "do n't care"]:
                value = ''
    elif domain == 'restaurant':
        if slot == "area":
            if value in ["center", 'scentre', "center of town", "city center", "cb30aq", "town center",
                         'centre of cambridge', 'city centre']:
                value = "centre"
            elif value == "west part of town":
                value = "west"
            elif value == "n":
                value = "north"
            elif value in ['the south']:
                value = 'south'
            elif value not in ['centre', 'south', "do n't care", 'west', 'east', 'north']:
                value = ''
        elif slot == "book day":
            if value == "monda":
                value = "monday"
            elif value == "t":
                value = "tuesday"
        elif slot in ['pricerange', 'price range']:
            slot = 'price range'
            if value in ['moderately', 'mode', 'mo']:
                value = 'moderate'
            elif value in ['not']:
                value = ''
            elif value in ['inexpensive', 'ch']:
                value = "cheap"
        elif slot == "food":
            if value == "barbecue":
                value = "barbeque"
        elif slot == "pricerange":
            slot = "price range"
            if value == "moderately":
                value = "moderate"
        elif slot == "book time":
            if value == "9:00":
                value = "09:00"
            elif value == "9:45":
                value = "09:45"
            elif value == "1330":
                value = "13:30"
            elif value == "1430":
                value = "14:30"
            elif value == "9:15":
                value = "09:15"
            elif value == "9:30":
                value = "09:30"
            elif value == "1830":
                value = "18:30"
            elif value == "9":
                value = "09:00"
            elif value == "2:00":
                value = "14:00"
            elif value == "1:00":
                value = "13:00"
            elif value == "3:00":
                value = "15:00"
    elif domain == 'taxi':
        if slot in ['arriveBy', 'arrive by']:
            slot = 'arrive by'
            if value == '1530':
                value = '15:30'
            elif value == '15 minutes':
                value = ''
        elif slot in ['leaveAt', 'leave at']:
            slot = 'leave at'
            if value == '1:00':
                value = '01:00'
            elif value == '21:4':
                value = '21:04'
            elif value == '4:15':
                value = '04:15'
            elif value == '5:45':
                value = '05:45'
            elif value == '0700':
                value = '07:00'
            elif value == '4:45':
                value = '04:45'
            elif value == '8:30':
                value = '08:30'
            elif value == '9:30':
                value = '09:30'
            value = value.replace(".", ":")

    elif domain == 'train':
        if slot in ['arriveBy', 'arrive by']:
            slot = 'arrive by'
            if value == '1':
                value = '01:00'
            elif value in ['does not care', 'doesnt care', "doesn't care"]:
                value = "do n't care"
            elif value == '8:30':
                value = '08:30'
            elif value == 'not 15:45':
                value = ''
            value = value.replace(".", ":")
        elif slot == 'day':
            if value == 'doesnt care' or value == "doesn't care":
                value = "do n't care"
        elif slot in ['leaveAt', 'leave at']:
            slot = 'leave at'
            if value == '2:30':
                value = '02:30'
            elif value == '7:54':
                value = '07:54'
            elif value == 'after 5:45 pm':
                value = '17:45'
            elif value in ['early evening', 'friday', 'sunday', 'tuesday', 'afternoon']:
                value = ''
            elif value == '12':
                value = '12:00'
            elif value == '1030':
                value = '10:30'
            elif value == '1700':
                value = '17:00'
            elif value in ['does not care', 'doesnt care', 'do nt care', "doesn't care"]:
                value = "do n't care"

            value = value.replace(".", ":")
    if value in ['dont care', "don't care", "do nt care", "doesn't care"]:
        value = "do n't care"

    return slot, value
