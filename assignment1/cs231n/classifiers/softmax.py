import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  for i in range(num_train):
    f = X[i].dot(W)
    shift_f = f - np.max(f)
    loss += -shift_f[y[i]] + np.log(np.sum(np.exp(shift_f)))
    for j in range(num_classes):
      common_part = np.exp(shift_f[j]) / np.sum(np.exp(shift_f)) * X[i]
      if j == y[i]:
        dW[:, j] += -X[i] + common_part
      else:
        dW[:, j] += common_part
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)
  dW /= num_train
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  f = X.dot(W)
  shift_f = f - np.max(f, axis = 1).reshape(-1, 1)
  loss = np.sum(np.log(np.sum(np.exp(shift_f), axis = 1)).reshape(-1, 1) - shift_f[range(num_train), y].reshape(-1,1)) 
  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)
    
  common_part = np.exp(shift_f) / np.sum(np.exp(shift_f), axis = 1).reshape(-1, 1)
  dW = X.T.dot(common_part)
  special_part = np.zeros((num_train, num_classes))
  special_part[range(num_train), y] = 1
  dW += - special_part.T.dot(X).T
  dW /= num_train
  dW += reg * W

  return loss, dW

