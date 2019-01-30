import pandas as pd
import matplotlib.pyplot as plt


def feature_exploration_plot(dataset):
    plt.figure(figsize=(12,12))

    # Plot the feature "Children"
    plt.subplot(3,4,1)
    plt.scatter(dataset.iloc[:,1],dataset.iloc[:,-1])
    plt.xticks([], [])
    plt.yticks([],[])
    plt.title('Number of Children')

    # Plot the extra features
    for i in range(2,13,1):
        plt.subplot(3,4,i)
        plt.scatter(dataset.iloc[:,-i],dataset.iloc[:,-1])
        title=["", "", "Sqrt of Children Feature","Has Link?", "Has Quote?",
        "Has Reference?", "Has Exclamation Mark?", "Number of Words",
        "Punctuation/Word", "Average Letteres/Word", "Is There A Question?",
        "Swear Words Count", "Misspelled Words Count"]
        plt.title(title[i])
        plt.xticks([], [])
        plt.yticks([],[])

    # Save the image
    plt.savefig('features.pdf')
