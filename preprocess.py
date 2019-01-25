import numpy as np
import pandas as pd
import re
from collections import Counter
from spellchecker import SpellChecker

def preprocess(dataset, nb_words):
    
    # Feature: Does the comment contain a question mark
    qmarks = np.zeros((dataset.shape[0]))
    # Feature: Normalized comment length (number of words)
    n_words = np.zeros((dataset.shape[0]))
    # Feature: Avg number of letters per words
    letters_per_word = np.zeros((dataset.shape[0]))
    # Feature: Number of punctuation signs per word (, . ! ? : ;)
    punctuation_count = np.zeros((dataset.shape[0]))
    punct = [',', '.', '!', '?', ':', ';']
    # Feature: Most common word count
    l = np.concatenate(dataset["text"])
    most_common_words = [word for word, word_count in Counter(l).most_common(nb_words)]
    zeros = np.zeros(shape = (dataset.shape[0], nb_words))
    word_count_features = pd.DataFrame(zeros, columns = most_common_words)    
    # Feature: Misspelled words
    misspelled_feature = np.zeros(dataset.shape[0])
    spell = SpellChecker()
    #Feature: Swear words
    swear_words = pd.read_csv("swearWords.csv")
    s_words = np.zeros(dataset.shape[0])
    
    
    # Iterate over comments
    for i in range(dataset.shape[0]):
        txt = dataset.iloc[i]["text"]
        # Iterate over words
        for w in txt:
            # most common words
            for target in most_common_words:
                if w == target:
                    word_count_features.iloc[i][target] += 1
            
            #swear words count
            for target in swear_words:
                if w == target:
                    s_words[i]+=1
            
            # punctuation count
            for x in punct:
                punctuation_count[i] += w.count(x)
                
            # question counter
            if "?" in w:
                qmarks[i] = 1
                
            # comment length
            n_words[i] += 1
            # number of letters
            letters_per_word[i] += len(w)
            
    # misspelled count   
    for i in range(dataset.shape[0]):
        new = [re.sub(r"^\W+|\W+$","", word) for word in dataset.iloc[i]["text"]]
        misspelled_words = spell.unknown(new)
        misspelled_feature[i] = len(misspelled_words)
    
    # Get average number of letters per word
    for i in range(dataset.shape[0]):
        letters_per_word[i] = letters_per_word[i]/n_words[i]
        
    # Get average punctuation marks per word
    for i in range(dataset.shape[0]):
        punctuation_count[i] = punctuation_count[i]/n_words[i]
        
    # Add feature columns 
    # Most common words
    dataset = pd.concat([dataset, word_count_features], axis=1)
    # Misspelled words count
    dataset = dataset.assign(misspelled=pd.Series(misspelled_feature).values)
    # Swear words
    dataset = dataset.assign(s_words=pd.Series(s_words).values)
    # Question marks
    dataset = dataset.assign(has_question=pd.Series(qmarks).values.astype(int))
    # Avg letters per word
    dataset = dataset.assign(letters_per_word=pd.Series(letters_per_word).values)
    # Punctuation per word
    dataset = dataset.assign(punctuation_count=pd.Series(punctuation_count).values)
    #Add bias term
    ones = np.ones((dataset.shape[0]))
    dataset = dataset.assign(bias = pd.Series(ones).values)
     #Drop text column
    dataset = dataset.drop(["text"], axis=1)
    #Move y value to the end
    dataset = dataset[["bias"] + [c for c in dataset if c not in ["popularity_score","bias"]] + ["popularity_score"]]
    return (dataset, most_common_words)