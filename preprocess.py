import json
from util import *
from zipfile import ZipFile
from urllib.request import urlopen, urlretrieve
import os
from io import BytesIO
import argparse


def main(args):
    word_vectors_file = "word-vectors/paragram_300_sl999.txt"
    vectors_url = "https://www.dropbox.com/s/liverep9vmsm9wu/paragram_300_sl999.zip?dl=1"

    val_list_file = os.path.join(args.data_path, "valListFile.json")
    test_list_file = os.path.join(args.data_path, "testListFile.json")
    data_file = os.path.join(args.data_path, "data.json")

    domains = ["restaurant", "taxi", "train", "attraction", "hotel"]

    data_train = {}
    data_test = {}
    data_val = {}

    if not os.path.isfile(word_vectors_file):
        if not os.path.exists("data"):
            os.makedirs("word-vectors")
        print("Downloading and unzipping the pre-trained word embeddings")
        resp = urlopen(vectors_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall("word-vectors")
        zip_ref.close()

    if os.path.isfile("data/train.json"):
        exit()

    print("Preprocessing the data and creating the ontology")

    if not os.path.exists("data"):
        os.makedirs("data")

    if not (os.path.exists(data_file) and os.path.exists(data_file) and os.path.exists(data_file)):
        print("Invalid data path")
        exit()

    val_list = []
    with open(val_list_file, 'r') as f:
        for line in f:
            val_list.append(line.strip())

    test_list = []
    with open(test_list_file, 'r') as f:
        for line in f:
            test_list.append(line.strip())

    data = json.load(open(data_file))
    for filename in data:
        if 'SSNG' not in filename and 'SMUL' not in filename:
            dialog = data[filename]
            dialogue = {}
            for domain in domains:
                if dialog["goal"][domain]:
                    dialogue[domain] = True
                else:
                    dialogue[domain] = False
            for idx, turn in enumerate(dialog["log"]):
                if idx % 2 == 0:
                    dialogue[str(idx // 2)] = {}
                    dialogue[str(idx // 2)]["user"] = {"text": turn["text"]}
                else:
                    system_meta = turn["metadata"]
                    dialogue[str(idx // 2)]["user"]["belief_state"] = system_meta
                    dialogue[str(idx // 2)]["system"] = turn["text"]
            if len(dialog["log"]) % 2 != 0:
                dialogue[str(idx // 2)]["user"]["belief_state"] = system_meta
                dialogue[str(idx // 2)]["system"] = ""
            dialogue["len"] = idx // 2 + 1
            if filename in val_list:
                data_val[filename] = dialogue
            elif filename in test_list:
                data_test[filename] = dialogue
            else:
                data_train[filename] = dialogue
    ontology = {}
    max_turns = process_dialogues(data_train, ontology)
    max_turns = process_dialogues(data_val, ontology, max_turns)
    max_turns = process_dialogues(data_test, ontology, max_turns)

    print("The maximum number of turns in these dialogues is {}".format(max_turns))

    train_data = [data_train[k] for k in data_train]
    val_data = [data_val[k] for k in data_val]
    test_data = [data_test[k] for k in data_test]

    with open('data/train.json', 'w') as outfile:
        json.dump(train_data, outfile, indent=4)
    with open('data/validate.json', 'w') as outfile:
        json.dump(val_data, outfile, indent=4)
    with open('data/test.json', 'w') as outfile:
        json.dump(test_data, outfile, indent=4)

    with open('data/ontology.json', 'w') as outfile:
        json.dump(ontology, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
