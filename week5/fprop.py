import numpy as np
import math



def fprop(input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias):
    '''% This method forward propagates through a neural network.
    % Inputs:
    %   input_batch: The input data as a matrix of size numwords X batchsize where,
    %     numwords is the number of words, batchsize is the number of data points.
    %     So, if input_batch(i, j) = k then the ith word in data point j is word
    %     index k of the vocabulary.
    %
    %   word_embedding_weights: Word embedding as a matrix of size
    %     vocab_size X numhid1, where vocab_size is the size of the vocabulary
    %     numhid1 is the dimensionality of the embedding space.
    %
    %   embed_to_hid_weights: Weights between the word embedding layer and hidden
    %     layer as a matrix of soze numhid1*numwords X numhid2, numhid2 is the
    %     number of hidden units.
    %
    %   hid_to_output_weights: Weights between the hidden layer and output softmax
    %               unit as a matrix of size numhid2 X vocab_size
    %
    %   hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
    %
    %   output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
    %
    % Outputs:
    %   embedding_layer_state: State of units in the embedding layer as a matrix of
    %     size numhid1*numwords X batchsize
    %
    %   hidden_layer_state: State of units in the hidden layer as a matrix of size
    %     numhid2 X batchsize
    %
    %   output_layer_state: State of units in the output layer as a matrix of size
    %     vocab_size X batchsize
    %'''
    (numwords, batchsize) = input_batch.shape;

    (vocab_size, numhid1) = word_embedding_weights.shape;
    numhid2 = embed_to_hid_weights.shape[1] 

    #%% COMPUTE STATE OF WORD EMBEDDING LAYER.
    #% Look up the inputs word indices in the word_embedding_weights matrix.
    #function result transponated
    embedding_layer_state = word_embedding_weights[input_batch.reshape(1, -1).T].reshape(numhid1 * numwords, -1)

    
    #print(embedding_layer_state)
    #%% COMPUTE STATE OF HIDDEN LAYER.
    #% Compute inputs to hidden units.
    inputs_to_hidden_units = np.dot(embed_to_hid_weights.T, embedding_layer_state) + np.kron(np.ones((1, batchsize)), hid_bias)


    '''% Apply logistic activation function.
    % FILL IN CODE. Replace the line below by one of the options.
    '''
    hidden_layer_state = np.zeros((numhid2, batchsize));
    '''% Options '''
    # (a) hidden_layer_state = 1 ./ (1 + exp(inputs_to_hidden_units));
    # (b) hidden_layer_state = 1 ./ (1 - exp(-inputs_to_hidden_units));
    # (c) hidden_layer_state = 1 / (1 + np.exp(-inputs_to_hidden_units))
    # (d) hidden_layer_state = -1 ./ (1 + exp(-inputs_to_hidden_units));

    '''%% COMPUTE STATE OF OUTPUT LAYER.
    % Compute inputs to softmax.
    % FILL IN CODE. Replace the line below by one of the options.
    '''
    ####inputs_to_softmax = np.zeros((vocab_size, batchsize));
    #'''% Options
    #% (a) inputs_to_softmax = np.dot(hid_to_output_weights.T, hidden_layer_state) +  np.kron(np.ones((1,batchsize)), output_bias)
    '''% (b) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, batchsize, 1);
    % (c) inputs_to_softmax = hidden_layer_state * hid_to_output_weights' +  repmat(output_bias, 1, batchsize);
    % (d) inputs_to_softmax = hid_to_output_weights * hidden_layer_state +  repmat(output_bias, batchsize, 1);

    % Subtract maximum. 
    % Remember that adding or subtracting the same constant from each input to a
    % softmax unit does not affect the outputs. Here we are subtracting maximum to
    % make all inputs <= 0. This prevents overflows when computing their
    % exponents.
    '''
    inputs_to_softmax = inputs_to_softmax - np.kron(np.ones((vocab_size,1)), np.max(inputs_to_softmax))
    
    #% Compute exp.
    output_layer_state = np.exp(inputs_to_softmax);

    #% Normalize to get probability distribution.
    output_layer_state = output_layer_state / np.kron(np.ones((vocab_size,1)), output_layer_state.sum(axis=0))

    return [embedding_layer_state, hidden_layer_state, output_layer_state]
