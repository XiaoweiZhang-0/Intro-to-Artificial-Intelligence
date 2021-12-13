import copy
import csv
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import random
# import seaborn

from collections import Counter

# data format: Nxd where N is # of cases and d is dimension
def set_seed(seed):  # For reproducibility, fix random seeds.
  random.seed(seed)
  np.random.seed(seed)

#dataset class
class MNISTDataset:

  def __init__(self, dataset, legalLabels):
    self.legalLabels = legalLabels
    if self.legalLabels == 10:
        self.d1 = 28
        self.d2 = 28
    else:
        self.d1 = 60
        self.d2 = 70
    newtrain = copy.deepcopy(dataset)
    trainlist = list(newtrain)
    N = len(trainlist)
    self.inputs = np.zeros((N,self.d1*self.d2))
    for x in range(N):
        # temp = []
        for y in range(0, self.d1):
        #   print("=====d1 is ", d1)
          for z in range(0, self.d2):
          # print(trainlist[x].get((y,z)))
            # temp.append(trainlist[x].get((y,z)))
        # datatrain.append((temp, trainlabels[x]))
            self.inputs[x][y*self.d2+z] = trainlist[x].get((y,z))
    self.labels = None
    # if split != 'test':
    #   self.labels, self.label_count = self.get_labels('{:s}labels_{:s}.npy'.format(datadir, split))
    #   assert self.labels.shape[0] == self.inputs.shape[0]
    # self.split = split

  def get_labels(self, labels):
    # label_matrix = np.load(filepath)  # (num_examples, num_labels)

    # labels = np.zeros((len(label_matrix), 1)).astype(int)  # (num_examples, 1)
    self.labels = np.array(labels)[:,np.newaxis]
    label_count = Counter()
    for i in range(len(self.labels)):
      label= labels[i]
      label_count[label] += 1

    return self.labels, label_count

  def num_examples(self):
    return self.inputs.shape[0]

  def dim(self):
    return self.inputs.shape[1]

  def generate_batch(self, batch_size):
    inds = list(range(self.num_examples()))  
    # if self.split == 'train':  # If train, shuffle example indices before generating
    random.shuffle(inds)    
    for i in range(0, len(inds), batch_size):
        inds_batch = inds[i: i + batch_size]
        # print("====inds_batch, line 72", self.labels)
        X = self.inputs[inds_batch, :]
        y = self.labels[inds_batch, :] if self.labels is not None else None
        yield X, y


# feature normalization
def normalize_features(X, mu=None, sigma=None):
  if mu is None or sigma is None: 
    mu = X.mean(0)
    sigma = X.std(0)
    sigma[sigma < 0.0001] = 1  # Avoid division by zero in case of degenerate features.

  # Normalize features and also add a bias feature.
  X_new = np.concatenate([np.ones((X.shape[0], 1)), (X - mu) / sigma], 1)

  return X_new, mu, sigma

def softmax(scores):  # (num_examples, num_labels)
  nonnegs = np.exp(scores - np.amax(scores, axis=1)[:, np.newaxis])  # Mitigate numerical overflow by subtracting max 
  return nonnegs / np.sum(nonnegs, axis=1)[:, np.newaxis]

def logsumexp(scores):  # (num_examples, num_labels)
  rowwise_max = np.amax(scores, axis=1)[:, np.newaxis] 
  return rowwise_max + np.log(np.sum(np.exp(scores - rowwise_max), axis=1)[:, np.newaxis])


class LinearClassifier:
    
  def __init__(self, inputs_train, num_labels, init_range=0.0):
    new_features, self.mu, self.sigma = normalize_features(inputs_train)  # Get means and standard devations
    self.dim = new_features.shape[1]
    if num_labels == 10:
        self.d1 = 28
        self.d2 = 28
    else:
        self.d1 = 60
        self.d2 = 70

    # Initialize parameters.
    # print("=========", num_labels) 
    self.W = np.random.uniform(-init_range, init_range, (self.dim, num_labels))

    # Initialize the gradient.
    self.W_grad = np.zeros((self.dim, num_labels))
                      
  def forward(self, X_raw, y=None, regularization_weight=0.):
    X = normalize_features(X_raw, self.mu, self.sigma)[0]
    scores = np.matmul(X, self.W)  # (batch_size, num_labels)        
    loss_sum = None
    if y is not None:  # We're given gold labels, we're training.

      # TODO: Compute the negative log probabilities here using logsumexp. The NumPy function take_along_axis will also be useful.
      negative_log_probs = -1*(np.take_along_axis(scores,y,axis=1) - logsumexp(scores))  # (batch_size,)

      squared_norm_W = np.linalg.norm(self.W[1:, :], 'fro') ** 2  # Don't regularize bias parameters
      loss_sum = np.sum(negative_log_probs) + regularization_weight * squared_norm_W
      self.accumulate_gradients(X, y, scores, regularization_weight)

    return loss_sum, scores
  
  def accumulate_gradients(self, X, y, scores, regularization_weight):
    batch_size, num_labels = scores.shape
    probs = softmax(scores)        

    # TODO: Compute the gradient of the average negative log probability wrt W
    G = np.zeros((batch_size,num_labels))
    G[np.arange(len(y)),y.T] = 1
    loss_grad = 1/batch_size * np.matmul(X.T,probs - G)

    # TODO: Compute the gradient of the regularization term (again, remember that bias parameters are not regularized).
    squared_norm_grad = self.W  # (dim, num_labels)
    
    self.W_grad += loss_grad + regularization_weight * squared_norm_grad
      
  def predict(self, X_raw):
    #   print("++++++X-raw",X_raw.shape)
      _, scores = self.forward(X_raw)
      preds = np.argmax(scores, axis=1)[:, np.newaxis]  # (batch_size, 1)
      return preds
  
  def classify(self, X_raw):
        newtrain = copy.deepcopy(X_raw)
        trainlist = list(newtrain)
        N = len(trainlist)
        # print("_____", self.d1)
        inputs = np.zeros((N,self.d1*self.d2))
        for x in range(N):
        # temp = []
            for y in range(0, self.d1):
                for z in range(0, self.d2):
          # print(trainlist[x].get((y,z)))
            # temp.append(trainlist[x].get((y,z)))
        # datatrain.append((temp, trainlabels[x]))
                    inputs[x][y*self.d2+z] = trainlist[x].get((y,z))
        # print("++++++inputs", inputs.shape)
        _, scores = self.forward(inputs)
        preds = np.argmax(scores, axis=1)[:, np.newaxis]  # (batch_size, 1)
        return preds


  def zero_grad(self):
    self.W_grad.fill(0.)
      
  def num_parameters(self):
    return self.W.size 



class SGDOptimizer:
    
  def __init__(self, model, learning_rate):
    self.model = model
    self.lr = learning_rate
      
  def step(self):
    self.model.W -= self.lr * self.model.W_grad
      
  def zero_grad(self):
    self.model.zero_grad()
      
  def modify_lr(self, learning_rate):
    self.lr = learning_rate   


def evaluate_accuracy(model, dataset_eval, batch_size_eval=64):
  num_correct = 0
  for X, y in dataset_eval.generate_batch(batch_size_eval):
    num_correct += np.sum(model.predict(X) == y)
  return num_correct / dataset_eval.num_examples() * 100.

def train(dataset_train, dataset_val, legalLabels, learning_rate=0.1, init_range=0., batch_size=16, regularization_weight=0., max_num_epochs=10, seed=42, loss_improvement=0.01, decay=2., tolerance=5, verbose=False):
  set_seed(seed)
#   print("================",dataset_train.inputs.shape)  
  model = LinearClassifier(dataset_train.inputs, legalLabels, init_range=init_range)  
  optimizer = SGDOptimizer(model, learning_rate)
  
  best_acc_val = float('-inf')
  best_W = None
  num_continuous_fails = 0
  loss_avg_before = None
  
  if verbose:
    print('Num parameters {:d}'.format(model.num_parameters()))
    print('Batch size {:d}, learning rate {:.5f}, regularization_weight {:.5f}'.format(batch_size, learning_rate, regularization_weight))
      
  for epoch in range(max_num_epochs):
    loss_total = 0.
    num_correct = 0    

    for X, y in dataset_train.generate_batch(batch_size):  # Shuffled in each epoch
      loss_sum, scores = model.forward(X, y, regularization_weight)
      loss_total += loss_sum
      preds = np.argmax(scores, axis=1)[:, np.newaxis]
      num_correct += np.sum(preds == y)

      optimizer.step()
      optimizer.zero_grad()

    loss_avg = loss_total / dataset_train.num_examples()
    acc_train = num_correct / dataset_train.num_examples() * 100. 
    acc_val = evaluate_accuracy(model, dataset_val)
  
    if acc_val > best_acc_val:
      num_continuous_fails = 0
      best_acc_val = acc_val
      best_W = copy.deepcopy(model.W)
    else:
      num_continuous_fails += 1
      if num_continuous_fails > tolerance:
        if verbose: 
            print('Early stopping')
        break
    
    if loss_avg_before is not None:
      if loss_avg_before - loss_avg < loss_improvement:  # Training loss has not improved sufficiently, decay the learning rate
        optimizer.modify_lr(optimizer.lr / decay)
        if verbose and decay != 1.0:
          print('Decaying learning rate to {:.5f}'.format(optimizer.lr))
    loss_avg_before = loss_avg 

    if verbose:
      print('End of epoch {:3d}:\t loss avg {:10.4f}\t acc train {:10.2f}\t acc val {:10.2f}'.format(
          epoch + 1, loss_avg, acc_train, acc_val)) 

  model.W = best_W
  if verbose:
    print('Best acc val: {:10.2f}'.format(best_acc_val))

  return model, best_acc_val, loss_avg, acc_train


def train_and_tune(datatrain, dataval, trainlabel, valabel, legalLabels):
    set_seed(42)
    legalLabels = legalLabels[-1] + 1
    dataset_train = MNISTDataset(datatrain, legalLabels)
    dataset_val = MNISTDataset(dataval, legalLabels)
    dataset_train.get_labels(trainlabel)
    dataset_val.get_labels(valabel)
    model_best = tune(dataset_train, dataset_val, legalLabels)
    return model_best


def tune(dataset_train, dataset_val, legalLabels):
    model_best = None
    best_acc_val = float('-inf')
    for batch_size in [16]:
        for learning_rate in [0.1]:
            for regularization_weight in [0, 0.0001, 0.001, 0.01, 1.0, 10.0]:
                model, acc_val, loss_avg, acc_train = train(dataset_train, dataset_val, legalLabels, learning_rate=learning_rate, batch_size=batch_size, regularization_weight=regularization_weight, max_num_epochs=60)
                print('Lambda {:10.4f}\t loss {:10.4f}\t acc train {:2.2f}\t acc val {:2.2f}'.format(regularization_weight, loss_avg, acc_train, acc_val))
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    model_best = copy.deepcopy(model)
    return model_best