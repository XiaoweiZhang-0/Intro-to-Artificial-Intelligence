import heapq
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import classificationMethod

from collections import deque
from itertools import accumulate, groupby


def set_seed(seed):  # For reproducibility, fix random seeds.
  random.seed(seed)
  np.random.seed(seed)


def gini_impurity(distribution):  
    # distribution: [p_1 ... p_L] such that p_l >= 0 and sum_{l=1}^L p_l = 1
    impurity = 0
    for p in distribution:
      impurity = impurity + p*(1-p)
    # print("impurity is ", impurity)
    return impurity  


def compute_split_loss(total1, total2, positive1, positive2):
    positive1_prob = positive1 / total1 if total1 > 0. else 0.5
    positive2_prob = positive2 / total2 if total2 > 0. else 0.5
    impurity1 = gini_impurity([1 - positive1_prob, positive1_prob])
    impurity2 = gini_impurity([1 - positive2_prob, positive2_prob])
    impurity = total1 * impurity1 + total2 * impurity2
    return impurity

def fit_stump(data, weights=None, indices=None):  # O(dN)
    """
    Computes the best split on a dataset of N (input, label) pairs according to Gini impurity where the label is either +1 or -1.
    Each example is weighted by some nonnegative weight value (1.0 if None).
    Only the examples included in the list of indices are considered (all if None). 
    """
    if weights is None:
      weights = np.ones(len(data))
    assert len(weights) == len(data)  
    assert (weights >= 0).all()
    
    if indices is None:
      indices = list(range(len(data)))

    feature_best = None
    threshold_best = None
    loss_best = float('inf')

    for feature in range(len(data[0][0])):
      key = lambda i: data[i][0][feature]

      # Sorting indices so that feature values are nondecreasing. 
      indices_sorted = sorted(indices, key=key)

      # Group indices with the same feature value for efficiency.
      groups = [list(group) for _, group in groupby(indices_sorted, key)]
      group_weights = [sum(weights[i] for i in group) for group in groups]
      group_weights_positive = [sum(weights[i] for i in group if data[i][1] == 1) for group in groups]

      # Precompute (1) total weight and (2) total positive label weight of every partition in O(N) time.  
      cumulative_weights = list(accumulate(group_weights))
      cumulative_weights_positive = list(accumulate(group_weights_positive))
      total = cumulative_weights[-1]  
      positive = cumulative_weights_positive[-1]

      # Loop over effective partitions.
      for group_num, (total1, positive1) in enumerate(zip(cumulative_weights[:-1], cumulative_weights_positive[:-1])):

        # TODO: Implement. You simply need to call compute_split_loss with right values.
        loss = compute_split_loss(total1, total-total1, positive1, positive-positive1)

        if loss < loss_best:
          loss_best = loss
          feature_best = feature
          current_feature_value = data[groups[group_num][0]][0][feature]
          next_feature_value = data[groups[group_num + 1][0]][0][feature]
          threshold_best = (current_feature_value + next_feature_value) / 2.        
      
    # May return (None, None, float('inf')) if no split can be found (e.g., has one feature group for every dimension).
    return feature_best, threshold_best, loss_best 

class Node:

    def __init__(self, parent):
      self.parent = parent
      self.child_left = None
      self.child_right = None
      self.feature = None  # Feature (i.e., dimension) to split on
      self.threshold = None  
      self.label = None
      self.leaf = False

class BinaryClassifier():

    def predict(self, x):  # Given a vector x, return either +1 or -1 
        raise NotImplementedError

    def predict_all(self, data_unlabeled):
      return [self.predict(x) for x in data_unlabeled]    

    def evaluate_accuracy(self, data):
      # print(data[0][0])
      # for (x,y) in data:
      #   if y!=self.predict(x):
      #     print("y is ", y)
      #     print("prediction is ", self.predict(x))
      num_correct = sum(y == self.predict(x) for (x, y) in data)
      return num_correct / len(data) * 100. 

class DecisionTree(BinaryClassifier):

    def __init__(self, data, weights=None, max_depth=10, min_split_size=1):
      if weights is None:
        weights = np.ones(len(data))  
      self.root = self.fit(data, weights, max_depth, min_split_size)

    def fit(self, data, weights, max_depth, min_split_size):
      root = Node(None)
      queue = deque()
      queue.append((list(range(len(data))), root, 1))
      while queue:
        indices, node, depth = queue.popleft()
        weight_total = sum(weights[i] for i in indices)
        weight_total_positive = sum(weights[i] for i in indices if data[i][1] == 1)
        node.label = 1 if weight_total_positive > weight_total / 2. else 0

        if depth >= max_depth or len(indices) < min_split_size:
          node.leaf = True 
          continue
        
        feature, threshold, loss = fit_stump(data, weights, indices)
        # TODO: Fit a stump on the data subset under the node (indicated by indices). 
        # This should give you 3 variables: feature, threshold, and loss.
        # raise 

        if loss == float('inf'):  # Could not find any split (e.g., pure).
          node.leaf = True 
          continue

        node.feature = feature
        node.threshold = threshold
        indices_left = [i for i in indices if data[i][0][feature] <= threshold]
        indices_right = [i for i in indices if data[i][0][feature] > threshold]
        node.child_left = Node(None)
        node.child_right = Node(None)
        queue.append((indices_left, node.child_left, depth + 1))
        queue.append((indices_right, node.child_right, depth + 1))
    
      return root    

    def predict(self, x):
      node = self.root
      while not node.leaf: 
        node = node.child_left if x[node.feature] <= node.threshold else node.child_right
      return node.label 

def tune_tree(data_train, data_val, verbose=False):
    tree_best = None
    acc_val_best = 0. 
    for max_depth in np.logspace(1, 5, num=6, base=2).astype(int):
      for min_split_size in np.logspace(0, 4, num=5, base=2).astype(int):
        tree = DecisionTree(data_train, max_depth=max_depth, min_split_size=min_split_size)
        # print(type(data_train[0][1]))
        # for i in data_train:
        #   print(i)
        acc_train = tree.evaluate_accuracy(data_train)
        acc_val = tree.evaluate_accuracy(data_val)
        print_string = 'max_depth={:d}   min_split_size={:d}   acc_train {:.2f}   acc_val {:.2f}'.format(max_depth, min_split_size, acc_train, acc_val)
        if acc_val > acc_val_best:
          acc_val_best = acc_val
          tree_best = tree
          print_string += ' <--------new best'
        if verbose:
          print(print_string)
    return tree_best, acc_val_best

# def train():
#         tree_best, acc_best = tune_tree(data_train, data_val, verbose=True)