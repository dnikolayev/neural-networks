from __future__ import print_function
from builtins import range
import numpy as np

def display_nearest_words(word, model, k):
    # % Shows the k-nearest words to the query word.
    # % Inputs:
    # %   word: The query word as a string.
    # %   model: Model returned by the training script.
    # %   k: The number of nearest words to display.
    # % Example usage:
    # %   display_nearest_words('school', model, 10)

    word_embedding_weights = model['word_embedding_weights']
    vocab = list(model['vocab'])

    id = -1
    try:
        id =  vocab.index(word)
    except: 
        print("Word %s\ not in vocabulary.\n" % word)

    if id == -1 :
        exit()

    #% Compute distance to every other word.
    vocab_size = len(vocab)
    word_rep = word_embedding_weights[id, :]
    diff = word_embedding_weights - np.kron(np.ones((vocab_size,1)), word_rep)
    distance = np.sqrt(np.sum(np.multiply(diff, diff), axis=1))
    # #% Sort by distance.
    order = sorted(range(len(distance)), key=lambda k: distance[k])
    order = order[1:k+1] # The nearest word is the query word itself, skip that.
    for i in range(0,k):
        print("%s %.2f" % (vocab[order[i]], distance[order[i]]))