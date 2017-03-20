from matlab_fix import loadmat
import math
import numpy as np

def load_data(N):
# % This method loads the training, validation and test set.
# % It also divides the training set into mini-batches.
# % Inputs:
# %   N: Mini-batch size.
# % Outputs:
# %   train_input: An array of size D X N X M, where
# %                 D: number of input dimensions (in this case, 3).
# %                 N: size of each mini-batch (in this case, 100).
# %                 M: number of minibatches.
# %   train_target: An array of size 1 X N X M.
# %   valid_input: An array of size D X number of points in the validation set.
# %   test: An array of size D X number of points in the test set.
# %   vocab: Vocabulary containing index to word mapping.

#here we load the dataset
    data = loadmat('data.mat')['data']

    #small fix for the incoming array positions
    #originally arrays stored positions like in MathLab (1...)
    #Changed to Python positions by the most efficient way
    for key in ('trainData','validData','testData'):
        data[key] = data[key] - np.ones(data[key].shape, int)
  
            
    numdims = len(data['trainData'])
    D = int(numdims) - 1
    M = int(math.floor(len(data['trainData'][0]) / N));
    train_input = data['trainData'][0:D,0:N*M].reshape(D, N, M)
    train_target = data['trainData'][D,0:N*M].reshape(1, N, M) 
    valid_input = data['validData'][0:D,:]
    valid_target = data['validData'][D,:]
    test_input = data['testData'][0:D,:]
    test_target = data['testData'][D,:]

    return [train_input, train_target, valid_input, valid_target, test_input, test_target, data['vocab']]
