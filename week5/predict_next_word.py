from __future__ import print_function
from builtins import range
import numpy as np
from fprop import fprop

def predict_next_word(word1, word2, word3, model, k):
    '''
    % Predicts the next word.
    % Inputs:
    %   word1: The first word as a string.
    %   word2: The second word as a string.
    %   word3: The third word as a string.
    %   model: Model returned by the training script.
    %   k: The k most probable predictions are shown.
    % Example usage:
    %   predict_next_word('john', 'might', 'be', model, 3)
    %   predict_next_word('life', 'in', 'new', model, 3)
    '''

    word_embedding_weights = model['word_embedding_weights']
    vocab = list(model['vocab'])

    input = np.vstack(np.asarray([-1,-1,-1]))
    words = (word1, word2, word3) 
    for i,w in enumerate(words):
        if words[i] in vocab:
            input[i] = vocab.index(words[i])
        else:
            print("Word ''%s\'' not in vocabulary.\n" % (word1))
    
    
    [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(input, model['word_embedding_weights'], model['embed_to_hid_weights'], model['hid_to_output_weights'], model['hid_bias'], model['output_bias'])
    indices = sorted(range(len(output_layer_state)), key=lambda k: output_layer_state[k], reverse=True)
    
    for i in range(0,k):
        print("%s %s %s %s Prob: %.5f\n" % (word1, word2, word3, vocab[indices[i]], output_layer_state[i]))