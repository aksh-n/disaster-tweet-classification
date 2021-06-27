"""Disaster Tweet Classification, Helpers

This module provides helper functions, to be used for the naive Bayes algorithm of main.py. The 
purposes of the helper functions relate to reading files, checking render, dividing original
training set into two sets, and tokenizing and normalizing the text data.
They are not meant for standalone purposes.

Copyright (c) 2021 Akshat Naik.
Licensed under the MIT License. See LICENSE in the project root for license information.
"""
import csv, re, random
from pprint import pprint as pp
from typing import Iterable


def get_dataset(filename: str) -> list[list[str]]:
    """Returns the dataset of the filename, rendered using csv, where filename takes four possible
    parameters: 'train', 'new_train', 'dev' or 'test'
    """
    if filename in ('train', 'test'):
        filename = f'nlp-getting-started/{filename}.csv'
    elif filename in ('new_train', 'dev'):
        filename = f'data/{filename}.csv'
    else:
        raise ValueError('Incorrect value passed to filename parameter.')
    with open(filename, 'r', encoding='utf-8', newline='') as csvfile:
        tweetsreader = csv.reader(csvfile)
        next(tweetsreader)
        return list(tweetsreader)


def check_render(dataset: list[list[str]], is_test: bool) -> bool:
    """Returns whether the dataset has been rendered correctly by the csv module."""
    for line in dataset:
        if not is_test:  # if it is a training dataset, then target is also there
            id, keyword, location, text, target = line
        else:
            id, keyword, location, text = line
        if not(re.fullmatch('[0-9]+', id) is not None and text != ''):
            print(line)
            return False
        elif not is_test and target not in ('0', '1'):
            print(line)
            return False
    return True


def divide_train_set() -> None:
    """Divides the original training set into a new training set and a development set."""
    random.seed(1926)  # for reproducibility

    # divides the original set into a new training set and a development set
    original_train_set = get_dataset('train')
    n_dev_set = len(original_train_set) // 10
    dev_set = random.sample(original_train_set, k=n_dev_set)
    new_train_set = []
    for line in original_train_set:
        if line not in dev_set:
            new_train_set.append(line)

    # writes the new sets to their respective new files
    with open('data/new_train.csv', 'w', encoding='utf-8', newline='') as csvfile:
        new_train_writer = csv.writer(csvfile)
        new_train_writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        new_train_writer.writerows(new_train_set)

    with open('data/dev.csv', 'w', encoding='utf-8', newline='') as csvfile:
        dev_writer = csv.writer(csvfile)
        dev_writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        dev_writer.writerows(dev_set)


def process_tweets(text: str) -> list[str]:
    """Returns a list of tokens from the text of the tweet after normalizing and tokenizing the
    text of the tweet.
    """
    text = text.lower()

    # Remove urls
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove user @ references and # from tweet
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuations that occur either start of the string or end-of-word
    # and "words" that contain only numbers and punctuations (like dates or time),
    # and condense all whitespaces to one whitespace
    text = re.sub(r'^[^A-Za-z]+|\b[^A-Za-z]+', '', text)
    # Tokenize by splitting by whitespaces
    list_of_words = text.split()

    return list_of_words


def get_vocab(dataset: list[list[str]]) -> set[str]:
    """Returns a set of words that are found in the text field of the dataset."""
    vocab = set()
    for line in dataset:
        text = line[3]
        words = process_tweets(text)
        vocab.update(words)
    return vocab


def divide_dataset_targets(dataset: list[list[str]]) -> (list[list[str]], list[list[str]]):
    """Return two datasets, one with only 0 as the target and one with only 1 as the target in the
    text field of the dataset.
    """
    data_0 = []
    data_1 = []
    for line in dataset:
        if line[-1] == '0':
            data_0.append(line)
        elif line[-1] == '1':
            data_1.append(line)
        else:
            raise ValueError('The value of the target is neither 0 nor 1')
    return data_0, data_1


if __name__ == "__main__":
    train = get_dataset('train')
    test = get_dataset('test')
    # divide_train_set()
    new_train = get_dataset('new_train')
    dev = get_dataset('dev')
    assert len(new_train) + len(dev) == len(train)

    for i in (train, new_train, dev):
        print(check_render(i, False))
    print(check_render(test, True))
