import numpy as np
import pandas as pd
import re
from collections import Counter
from spellchecker import SpellChecker

def feature_exploration(dataset):

    # Feature: Does the comment contain a question mark ?
    qmarks = np.zeros((dataset.shape[0]))

    # Feature: What is the number of words contained in the comment ?
    n_words = np.zeros((dataset.shape[0]))

    # Feature: What is the average number of letter per word in the comment ?
    letters_per_word = np.zeros((dataset.shape[0]))

    # Feature: How many punctuation signs per word does the comment have ?
    punctuation_count = np.zeros((dataset.shape[0]))
    punct = [',', '.', '!', '?', ':', ';']

    # Feature: What is the number of misspelled words in the comment ?
    misspelled_feature = np.zeros(dataset.shape[0])
    spell = SpellChecker()

    # Feature: How many swear words does the comment contain ?
    swear_words = pd.read_csv("swearWords.csv")
    s_words = np.zeros(dataset.shape[0])

    # Feature: Does the comment contain and exclamation mark ?
    emarks = np.zeros(dataset.shape[0])

    # Feature: Does the comment reference to another reddit user ?
    has_r = np.zeros(dataset.shape[0])

    # Feature: Does the comment contain a quote ?
    has_quote = np.zeros(dataset.shape[0])

    # Feature: Does the comment contain a link ?
    has_link = np.zeros(dataset.shape[0])


    # Iterate over comments
    for i in range(dataset.shape[0]):

        # stores the comment in a local variable
        txt = dataset.iloc[i]["text"]

        # Iterate over words and computes the features
        for w in txt:

            # swear words count
            for target in swear_words:
                if w == target:
                    s_words[i]+=1

            # punctuation count
            for x in punct:
                punctuation_count[i] += w.count(x)

            # question mark check
            if "?" in w:
                qmarks[i] = 1

            # comment length
            n_words[i] += 1

            # number of letters
            letters_per_word[i] += len(w)

            # exclamation mark check
            if "!" in w:
                emarks[i] = 1

            # reference to another reddit user check
            if "/r" in w:
                has_r[i] = 1

            # quote check
            if "\"" in w or "\'" in w:
                has_quote[i] = 1

            # link check
            if "http" in w:
                has_link[i] = 1

    # misspelled counter
    for i in range(dataset.shape[0]):
        new = [re.sub(r"^\W+|\W+$","", word) for word in dataset.iloc[i]["text"]]
        misspelled_words = spell.unknown(new)
        misspelled_feature[i] = len(misspelled_words)

    # computes the average number of letters per word
    for i in range(dataset.shape[0]):
        letters_per_word[i] = letters_per_word[i]/n_words[i]

    # computes the average punctuation marks per word
    for i in range(dataset.shape[0]):
        punctuation_count[i] = punctuation_count[i]/n_words[i]

    # Add feature columns to dataset

    # Misspelled words count
    dataset = dataset.assign(misspelled = misspelled_feature)

    # Swear words
    dataset = dataset.assign(s_words = s_words)

    # Question marks
    dataset = dataset.assign(has_question = qmarks.astype(int))

    # Avg letters per word
    dataset = dataset.assign(letters_per_word = letters_per_word)

    # Punctuation per word
    dataset = dataset.assign(punctuation_count = punctuation_count)

    # number of words in comment
    dataset = dataset.assign(nb_of_words = n_words)

    # emarks
    dataset = dataset.assign(emarks = emarks)

    # has_r
    dataset = dataset.assign(has_r = has_r)

    # has_quote
    dataset = dataset.assign(has_quote = has_quote)

    # has_link
    dataset = dataset.assign(has_link = has_link)

    # sqrt of children
    dataset = dataset.assign(sqrt_child = np.sqrt(dataset.iloc[:,0]))

    # Add bias term
    ones = np.ones((dataset.shape[0]))
    dataset = dataset.assign(bias = pd.Series(ones).values)

    # Drop text column
    dataset = dataset.drop(["text"], axis=1)

    # Move y value to the end and bias to the first column
    dataset = dataset[["bias"] + [c for c in dataset if c not in ["popularity_score","bias"]] + ["popularity_score"]]

    # returns the dataset and the most common words
    return (dataset)
