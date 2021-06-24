"""Disaster Tweet Classification, Helpers

This module provides helper functions, to be used for the naive Bayes algorithm of main.py. The 
purposes of the helper functions relate to reading files, checking render, dividing original
training set into two sets.
They are not meant for standalone purposes.

Copyright (c) 2021 Akshat Naik.
Licensed under the MIT License. See LICENSE in the project root for license information.
"""
import csv, re, random
from pprint import pprint as pp
from typing import Iterable


def read_csv(filename: str) -> list[list[str]]:
    """Returns a list of the reader object of the filename csv where filename takes four possible
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


def check_render(list_of_lines: list[list[str]], is_test: bool) -> bool:
    """Returns whether the list_of_lines has been rendered correctly by the csv module."""
    for line in list_of_lines:
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
    original_train_set = read_csv('train')
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

if __name__ == "__main__":
    train = read_csv('train')
    test = read_csv('test')
    # divide_train_set()
    new_train = read_csv('new_train')
    dev = read_csv('dev')
    assert len(new_train) + len(dev) == len(train)

    for i in (train, new_train, dev):
        print(check_render(i, False))
    print(check_render(test, True))
