"""Disaster Tweet Classification, Testing

This module tests and evaluates the accuracy of the naive Bayes multinomial algorithm, trained
on the new_train dataset, using the development dataset.

Copyright (c) 2021 Akshat Naik.
Licensed under the MIT License. See LICENSE in the project root for license information.
"""
import csv
from pprint import pprint as pp
from main import mass_test_naive_bayes
import helpers


def evaluate_accuracy(write_results: bool=False) -> int:
    """Returns the classification accuracy of naive Bayes multinomial algorithm, trained on
    new_train dataset, using the dev dataset.
    """
    train_dataset = helpers.get_dataset('new_train')
    test_dataset = helpers.get_dataset('dev')
    test_results = mass_test_naive_bayes(train_dataset, test_dataset)
    assert len(test_dataset) == len(test_results)
    
    # calculating the accuracy
    correct_predictions = 0
    comparisons = []
    for ind, line in enumerate(test_dataset):
        expected_target = line[-1]
        predicted_target = str(test_results[ind])
        if expected_target == predicted_target:
            correct_predictions += 1
        if write_results:
            id, text = line[0], line[3]
            comparisons.append([id, text, predicted_target, expected_target])
    
    if write_results:
        with open('data/test_results.csv', 'w', encoding='utf-8', newline='') as csvfile:
            test_writer = csv.writer(csvfile)
            test_writer.writerow(['id', 'text', 'predicted_target', 'expected_target'])
            test_writer.writerows(comparisons) 
    
    accuracy = correct_predictions / len(test_results)
    return accuracy


if __name__ == "__main__":
    print(evaluate_accuracy(write_results=True))
