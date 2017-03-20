from __future__ import print_function
from builtins import range
import time
import numpy as np
import math

from load_data import load_data
from fprop import fprop

#% This function trains a neural network language model.
def train(epochs):
    #% Inputs:
    #%   epochs: Number of epochs to run.
    #% Output:
    #%   model: A struct containing the learned weights and biases and vocabulary.

    start_time=time.time()

    #% SET HYPERPARAMETERS HERE.
    batchsize = 100  #% Mini-batch size.
    learning_rate = 0.1  #% Learning rate; default = 0.1.
    momentum = 0.9  #% Momentum; default = 0.9.
    numhid1 = 50  #% Dimensionality of embedding space; default = 50.
    numhid2 = 200  #% Number of units in hidden layer; default = 200.
    init_wt = 0.01  #% Standard deviation of the normal distribution
                     #% which is sampled to get the initial weights; default = 0.01

    #% VARIABLES FOR TRACKING TRAINING PROGRESS.
    show_training_CE_after = 100
    show_validation_CE_after = 1000

    #% LOAD DATA.
    [train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data(batchsize)

    [numwords, batchsize, numbatches] = np.shape(train_input)
    vocab_size = len(vocab)

    # % INITIALIZE WEIGHTS AND BIASES.
    word_embedding_weights = init_wt * np.random.standard_normal((vocab_size, numhid1))
    embed_to_hid_weights = init_wt * np.random.standard_normal((numwords * numhid1, numhid2))
    
    hid_to_output_weights = init_wt * np.random.standard_normal((numhid2, vocab_size))
    hid_bias = np.zeros((numhid2, 1))
    output_bias = np.zeros((vocab_size, 1))

    word_embedding_weights_delta = np.zeros((vocab_size, numhid1))
    word_embedding_weights_gradient = np.zeros((vocab_size, numhid1))
    embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2))
    hid_to_output_weights_delta = np.zeros((numhid2, vocab_size))
    hid_bias_delta = np.zeros((numhid2, 1));
    output_bias_delta = np.zeros((vocab_size, 1));
    expansion_matrix = np.identity(vocab_size);
    count = 0;
    tiny = math.exp(-30);

    # % TRAIN.
    for epoch in range(1, epochs+1):
        print("Epoch %d\n" % (epoch))
        this_chunk_CE = 0
        trainset_CE = 0
        #% LOOP OVER MINI-BATCHES.
        for m in range(0, numbatches): 
            input_batch = np.asarray(train_input[:, :, m])
            target_batch = np.asarray(train_target[:, :, m])

            '''% FORWARD PROPAGATE.
            % Compute the state of each layer in the network given the input batch
            % and all weights and biases '''

            [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
            
            '''% COMPUTE DERIVATIVE.
            %% Expand the target to a sparse 1-of-K vector.'''
            expanded_target_batch = np.vstack((expansion_matrix[:, [i for i in target_batch[0]]]))
            # if m>=400:
            #     print(m)
            #     print(expanded_target_batch)
            #     exit()

            #%% Compute derivative of cross-entropy loss function.
            error_deriv = output_layer_state - expanded_target_batch
            # print(output_layer_state.shape)
            # print(tiny)
            # print(np.log(output_layer_state + tiny)-np.log(output_layer_state))
            # print(-np.sum(np.dot(expanded_target_batch, np.log(output_layer_state + tiny).T)) / batchsize)
            # print(-np.sum(np.multiply(expanded_target_batch, np.log(output_layer_state + tiny))) / batchsize)
            # #print()
            # exit()
            #% MEASURE LOSS FUNCTION.
            CE = -np.sum(np.multiply(expanded_target_batch, np.log(output_layer_state + tiny))) / batchsize
            print(CE,end="\r")
            count =  count + 1
            this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count
            trainset_CE = trainset_CE + (CE - trainset_CE) / (m+1)
            print("'\rBatch %d Train CE %.3f" % (m+1, this_chunk_CE),end="\r")
            if np.mod(m+1, show_training_CE_after) == 0:
                print("\n", end="\r");
                count = 0;
                this_chunk_CE = 0;


    #     % BACK PROPAGATE.
    #     %% OUTPUT LAYER.
            hid_to_output_weights_gradient =  np.dot(hidden_layer_state, error_deriv.T)
            output_bias_gradient = np.vstack(error_deriv.sum(axis=1))
            back_propagated_deriv_1 = np.dot(hid_to_output_weights, error_deriv) * hidden_layer_state * (1 - hidden_layer_state)

    #     %% HIDDEN LAYER.
    #     % FILL IN CODE. Replace the line below by one of the options.     
            embed_to_hid_weights_gradient = np.zeros((numhid1 * numwords, numhid2))
    #     % Options:
    #     % (a) embed_to_hid_weights_gradient = back_propagated_deriv_1.T * embedding_layer_state
    #     % (b) embed_to_hid_weights_gradient = np.dot(embedding_layer_state, back_propagated_deriv_1.T)
    #     % (c) embed_to_hid_weights_gradient = back_propagated_deriv_1
    #     % (d) embed_to_hid_weights_gradient = embedding_layer_state

    #     % FILL IN CODE. Replace the line below by one of the options.
            hid_bias_gradient = np.zeros((numhid2, 1))
    #     % Options
    #     % (a) hid_bias_gradient = np.vstack(back_propagated_deriv_1.sum(axis=1))
    #     % (b) hid_bias_gradient = np.sum(back_propagated_deriv_1[0]);
    #     % (c) hid_bias_gradient = back_propagated_deriv_1
    #     % (d) hid_bias_gradient = back_propagated_deriv_1.T

    #     % FILL IN CODE. Replace the line below by one of the options.
            back_propagated_deriv_2 = np.zeros((numhid2, batchsize));
    #     % Options
    #     % (a) back_propagated_deriv_2 = np.dot(embed_to_hid_weights, back_propagated_deriv_1)
    #     % (b) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights;
    #     % (c) back_propagated_deriv_2 = back_propagated_deriv_1' * embed_to_hid_weights;
    #     % (d) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights';

            #word_embedding_weights_gradient[:] = 0
    #     %% EMBEDDING LAYER.
            for w in range(0,numwords):
               word_embedding_weights_gradient = word_embedding_weights_gradient + np.dot(expansion_matrix[:, input_batch[w, :]], back_propagated_deriv_2[w * numhid1 : (w+1) * numhid1, :].T)

        
    #     % UPDATE WEIGHTS AND BIASES.
            word_embedding_weights_delta = momentum * word_embedding_weights_delta + word_embedding_weights_gradient / batchsize
            word_embedding_weights = word_embedding_weights - learning_rate * word_embedding_weights_delta

            embed_to_hid_weights_delta = momentum * embed_to_hid_weights_delta + embed_to_hid_weights_gradient / batchsize
            embed_to_hid_weights = embed_to_hid_weights - learning_rate * embed_to_hid_weights_delta

            hid_to_output_weights_delta = momentum * hid_to_output_weights_delta + hid_to_output_weights_gradient / batchsize
            hid_to_output_weights = hid_to_output_weights - learning_rate * hid_to_output_weights_delta

            hid_bias_delta = momentum * hid_bias_delta + hid_bias_gradient / batchsize
            hid_bias = hid_bias - learning_rate * hid_bias_delta

            output_bias_delta = momentum * output_bias_delta + output_bias_gradient / batchsize
            output_bias = output_bias - learning_rate * output_bias_delta


        #% VALIDATE.
            if np.mod(m+1, show_validation_CE_after) == 0:
                print("\rRunning validation ...");
                [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(valid_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
                datasetsize = valid_input.shape[1]
                expanded_valid_target = expansion_matrix[:, valid_target]
                CE = -np.sum(np.multiply(expanded_valid_target, np.log(output_layer_state + tiny))) / datasetsize
                print(" Validation CE %.3f\n" % (CE))

        print("\rAverage Training CE %.3f\n" % (trainset_CE))

    print("Finished Training.\n")
    print("Final Training CE %.3f\n" % (trainset_CE))

    # % EVALUATE ON VALIDATION SET.
    print("\rRunning validation ...")
    
    [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(valid_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
    datasetsize = valid_input.shape[1]
    expanded_valid_target = expansion_matrix[:, valid_target]
    CE = -np.sum(np.multiply(expanded_valid_target, np.log(output_layer_state + tiny))) / datasetsize
    print("\rFinal Validation CE %.3f\n" % (CE))

    # % EVALUATE ON TEST SET.
    print("\rRunning test ...")
    [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(test_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
    datasetsize = test_input.shape[1]
    expanded_test_target = expansion_matrix[:, test_target];
    #   expanded_test_target .* log(output_layer_state + tiny))) / datasetsize;
    CE = -np.sum(np.multiply(expanded_test_target, np.log(output_layer_state + tiny))) / datasetsize
    print("\rFinal Test CE %.3f\n" % (CE));

    model = dict();
    model['word_embedding_weights'] = word_embedding_weights
    model['embed_to_hid_weights'] = embed_to_hid_weights
    model['hid_to_output_weights'] = hid_to_output_weights
    model['hid_bias'] = hid_bias
    model['output_bias'] = output_bias
    model['vocab'] = vocab

    end_time=time.time()
    print("Training took %.2f seconds\n" % (end_time - start_time))
    return model

train(1)