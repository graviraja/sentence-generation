'''This code contains the data preprocessing pipeline.

'''

import re
import dill
import spacy
import unicodedata
import torchtext
from torchtext.data import Example, Field, BucketIterator

# load the spacy english model
spacy_en = spacy.load('en')


def unicodeToAscii(s):
    # convert the data in unicode format to ascii format
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def tokenizer(text):
    # tokenizes the english text into a list of strings(tokens)
    return [tok.text for tok in spacy_en.tokenizer(text)]


def get_src_trg_line(line):
    # create the src and target data from the given line
    # target is src word shifted by right.
    words = line.split()
    src = line
    trg = " ".join(words[1:])
    return src, trg


# define the field for the sentence
SENTENCE = Field(init_token='<sos>', eos_token='<eos>', tokenize=tokenizer, sequential=True)
fields = [('src', SENTENCE), ('trg', SENTENCE)]


def create_vocab(train_file_path, min_occ=1):
    # create the vocabulary from the train data
    with open(train_file_path, 'r') as file:
        examples = []
        for i, line in enumerate(file):
            line = normalizeString(line)
            src, trg = get_src_trg_line(line)
            examples.append(Example.fromlist([src, trg], fields))
    train_data = torchtext.data.Dataset(examples, fields)
    SENTENCE.build_vocab(train_data)
    with open('data/SENTENCE.Field', 'wb') as f:
        dill.dump(SENTENCE, f)


def load_vocab():
    # loads the vocab
    with open('data/SENTENCE.Field', 'rb') as f:
        SENTENCE = dill.load(f)
    return SENTENCE


def create_data_from_file(file_path):
    # create the Dataset from the given file_path
    with open(file_path, 'r') as file:
        examples = []
        for i, line in enumerate(file):
            line = normalizeString(line)
            src, trg = get_src_trg_line(line)
            examples.append(Example.fromlist([src, trg], fields))
    data = torchtext.data.Dataset(examples, fields)
    return data


def create_data_from_lines(lines):
    # create the Dataset from the given lines
    examples = []
    for i, line in enumerate(lines):
        line = normalizeString(line)
        src, trg = get_src_trg_line(line)
        examples.append(Example.fromlist([src, trg], fields))
    data = torchtext.data.Dataset(examples, fields)
    return data

create_vocab('./data/ptb.train.txt')
