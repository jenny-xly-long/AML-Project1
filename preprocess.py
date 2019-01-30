import numpy as np
import pandas as pd
from collections import Counter

def preprocess(dataset, nb_words = 160):

    # Feature: How many of each most common word does the comment have ?
    l = np.concatenate(dataset["text"])
    most_common_words = [word for word, word_count in Counter(l).most_common(nb_words)]
    zeros = np.zeros(shape = (dataset.shape[0], nb_words))
    word_count_features = pd.DataFrame(zeros, columns = most_common_words)

    # Feature: Does the comment contain a question mark ?
    qmarks = np.zeros((dataset.shape[0]))

    # Feature: Does the comment contain and exclamation mark ?
    emarks = np.zeros(dataset.shape[0])

    # Feature: Does the comment contain a quote ?
    has_quote = np.zeros(dataset.shape[0])


    # Iterate over comments
    for i in range(dataset.shape[0]):

        # stores the comment in a local variable
        txt = dataset.iloc[i]["text"]

        # Iterate over words and computes the features
        for w in txt:

            # most common words
            for target in most_common_words:
                if w == target:
                    word_count_features.iloc[i][target] += 1

            # question mark check
            if "?" in w:
                qmarks[i] = 1

            # exclamation mark check
            if "!" in w:
                emarks[i] = 1

            # quote check
            if "\"" in w or "\'" in w:
                has_quote[i] = 1

    # Add feature columns to dataset

    # Most common words
    dataset = pd.concat([dataset, word_count_features], axis=1)

    # Question marks
    dataset = dataset.assign(qmarks = qmarks)

    # emarks
    dataset = dataset.assign(emarks = emarks)

    # has_quote
    dataset = dataset.assign(has_quote = has_quote)

    # sqrt of children
    dataset = dataset.assign(sqrt_child = np.sqrt(dataset.iloc[:,0]))

    # Add bias term
    ones = np.ones((dataset.shape[0]))
    dataset = dataset.assign(bias = ones)

    # Drop text column
    dataset = dataset.drop(["text"], axis = 1)

    # Move y value to the end and bias to the first column
    dataset = dataset[["bias"] + [c for c in dataset if c not in ["popularity_score","bias"]] + ["popularity_score"]]

    # returns the dataset and the most common words
    return (dataset, most_common_words)
