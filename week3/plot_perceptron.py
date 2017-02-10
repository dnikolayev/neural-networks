import pylab as pl
import numpy as np
import os

# %% Plots information about a perceptron classifier on a 2-dimensional dataset.
def plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history):
# %%
# % The top-left plot shows the dataset and the classification boundary given by
# % the weights of the perceptron. The negative examples are shown as circles
# % while the positive examples are shown as squares. If an example is colored
# % green then it means that the example has been correctly classified by the
# % provided weights. If it is colored red then it has been incorrectly classified.
# % The top-right plot shows the number of mistakes the perceptron algorithm has
# % made in each iteration so far.
# % The bottom-left plot shows the distance to some generously feasible weight
# % vector if one has been provided (note, there can be an infinite number of these).
# % Points that the classifier has made a mistake on are shown in red,
# % while points that are correctly classified are shown in green.
# % The goal is for all of the points to be green (if it is possible to do so).
# % Inputs:
# %   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
# %       num_neg_examples is the number of examples for the negative class.
# %   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
# %       num_pos_examples is the number of examples for the positive class.
# %   mistakes0 - A vector containing the indices of the datapoints from class 0 incorrectly
# %       classified by the perceptron. This is a subset of neg_examples.
# %   mistakes1 - A vector containing the indices of the datapoints from class 1 incorrectly
# %       classified by the perceptron. This is a subset of pos_examples.
# %   num_err_history - A vector containing the number of mistakes for each
# %       iteration of learning so far.
# %   w - A 3-dimensional vector corresponding to the current weights of the
# %       perceptron. The last element is the bias.
# %   w_dist_history - A vector containing the L2-distance to a generously
# %       feasible weight vector for each iteration of learning so far.
# %       Empty if one has not been provided.
# %%

    neg_correct_ind = np.array(list(set(range(len(neg_examples))) - set(mistakes0)),dtype=int)
    pos_correct_ind = np.array(list(set(range(len(pos_examples))) - set(mistakes1)),dtype=int)

    pl.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

    pl.subplot(2,2,1);
    if (len(neg_examples)):
        pl.plot(np.take(neg_examples[:,0], neg_correct_ind),np.take(neg_examples[:,1], neg_correct_ind),'og',markersize=20);
    if (len(pos_examples)):
        pl.plot(np.take(pos_examples[:,0], pos_correct_ind),np.take(pos_examples[:,1], pos_correct_ind),'sg',markersize=20);
    if (len(mistakes0) > 0):
        pl.plot(np.take(neg_examples[:,0], mistakes0),np.take(neg_examples[:,1], mistakes0),'or',markersize=20);
    if (len(mistakes1) > 0):
        pl.plot(np.take(pos_examples[:,0], mistakes1),np.take(pos_examples[:,1], mistakes1),'sr',markersize=20);
    pl.title('Classifier');
    
    # %In order to plot the decision line, we just need to get two points.
    pl.plot([-5,5],[(-w[-1]+5*w[0])/w[1],(-w[-1]-5*w[0])/w[1]],'k')
    pl.xlim([-1,1]);
    pl.ylim([-1,1]);

    pl.subplot(2,2,2);
    pl.plot(np.array(range(len(num_err_history))),num_err_history);
    pl.xlim([-1,max(15,len(num_err_history))]);
    pl.ylim([0,len(neg_examples)+len(pos_examples)+1]);
    pl.title('Number of errors');
    pl.xlabel('Iteration');
    pl.ylabel('Number of errors');

    pl.subplot(2,2,3);
    pl.plot(np.array(range(len(w_dist_history))),w_dist_history);
    pl.xlim([-1,max(15,len(num_err_history))]);
    pl.ylim([0,15]);
    pl.title('Distance')
    pl.xlabel('Iteration');
    pl.ylabel('Distance');
    if not os.path.exists('img'):
        os.makedirs('img')
    pl.savefig("img/%s.png" % (len(w_dist_history)-1))
    pl.close()