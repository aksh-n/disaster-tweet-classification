"""Disaster Tweet Classification, Main

This module implements the multinomial naive Bayes algorithm with Laplace smoothing.
The accuracy of the algorithm is evaluated as well.

Copyright (c) 2021 Akshat Naik.
Licensed under the MIT License. See LICENSE in the project root for license information.
"""
from math import log
from collections import Counter, defaultdict
import helpers


def train_naive_bayes(dataset: list[list[str]]) -> (int, int, dict[str, int], dict[str, int]):
    """Returns the logprior of the targets 0 and 1, and the loglikelihood of each word/token of
    the targets 0 and 1.
    
    This function implements the naive multinomial Bayes algorithm with Laplace smoothing. It
    trains the algorithm on the new_train dataset.
    """
    N_doc = len(dataset)

    # gets all possible words in the dataset
    vocab = helpers.get_vocab(dataset)

    # divides the dataset into two sub-datasets: one with all tweets with target 0,
    # another with all tweets with target 1
    data_0, data_1 = helpers.divide_dataset_targets(dataset)
    N_0, N_1 = len(data_0), len(data_1)

    # calculates log prior or P(c)
    logprior_0, logprior_1 = log(N_0) - log(N_doc), log(N_1) - log(N_doc)  # using log identities

    # gets count of each words occurences
    count_0, count_1 = get_count(data_0), get_count(data_1)

    # calculates log likelihood or P(w|c)
    # defaultdict takes care of unknown words in the testdoc passed to test_naive_bayes
    loglikelihood_0, loglikelihood_1 = defaultdict(int), defaultdict(int) 
    denom_0, denom_1 = sum(count_0[w_] for w_ in vocab), sum(count_1[w_] for w_ in vocab)
    for word in vocab:
        # +1 is because of Laplace smoothing
        loglikelihood_0[word] = log(count_0[word] + 1) - log(denom_0)
        loglikelihood_1[word] = log(count_1[word] + 1) - log(denom_1)

    return logprior_0, logprior_1, loglikelihood_0, loglikelihood_1


def get_count(dataset: list[list[str]]) -> Counter:
    """Returns a Counter object that counts the occurences of each word in the text field of the
    dataset.
    """
    count = Counter()
    for line in dataset:
        text = line[3]
        words = helpers.process_tweets(text)
        count.update(words)
    return count


def test_naive_bayes(testdoc: list[str], logprior_0: int, logprior_1: int, 
                     loglikelihood_0: dict[str, int], loglikelihood_1: dict[str, int]) -> int:
    """Returns the best target, 0 or 1, classified for the testdoc based on the naive multinomial
    Bayes algorithm.
    """
    sum_0, sum_1 = 0, 0
    text = testdoc[3]
    words = helpers.process_tweets(text)
    for word in words:
        # if word in vocab - this is no longer needed because defaultdict defaults to value 0
        # for unknown words
        sum_0 += loglikelihood_0[word]
        sum_1 += loglikelihood_1[word]
    if sum_0 >= sum_1:
        return 0
    else:
        return 1


def mass_test_naive_bayes(train_dataset: list[list[str]], test_dataset: list[list[str]]) -> list[int]:
    """Returns a list of integers either 0 or 1, each integer corresponding to the target of the
    each document.
    """
    logprior_0, logprior_1, loglikelihood_0, loglikelihood_1 = train_naive_bayes(train_dataset)
    test_results = []
    for line in test_dataset:
        result = test_naive_bayes(line, logprior_0, logprior_1, loglikelihood_0, loglikelihood_1)
        test_results.append(result)
    return test_results


if __name__ == "__main__":
    pass
