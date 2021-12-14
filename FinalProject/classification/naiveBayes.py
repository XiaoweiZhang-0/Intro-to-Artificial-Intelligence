# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import collections
import util
import classificationMethod
import math
import copy
import numpy as np

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 0.077 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    # self.odds = None
    # self.count = None
    # self.prior = None

  # def occurrProb(self, ret):
  #   prob = dict(collections.Counter(ret))
  #   for k in prob.keys():
  #       prob[k] = prob[k] / float(len(ret))
  #   return prob
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))
    
    if (self.automaticTuning):
        kgrid = np.arange(0.001, 0.1, 0.002)
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    #training data Nxd where N is # of training cases and d is dimension
    # print("training data is ", len(trainingData))
    feature= trainingData[0].values() 
    "feature for first training case"
    # print("feature is ", feature) 
    self.prior = util.Counter()
    self.condProb = {}

    #calculate prior distribution
    for i in trainingLabels:
      self.prior[i] += 1
    self.prior.normalize()
    
    # acc_val_best = 0
    # best_smoothing = 0
    # for smoothing in kgrid:

    #   for i in self.legalLabels: #i for label
    #     self.condProb[i] = {}
    #     # print(trainingData[1])
    #     for j in trainingData[0]: #j for the key for the corresponding feature
    #       # print("j is ", j)
    #       self.condProb[i][j] = util.Counter()
    #       for k in feature: # k for corresponding value for jth feature in 1st case
    #                   # print("k is ", k)
    #           self.condProb[i][j][k] = smoothing
      
    #   for n, fea in enumerate(trainingData):
    #     # print(n)
    #     i = trainingLabels[n] # true label for the nth training case
    #     for j in fea:
    #       # print(j)
    #       self.condProb[i][j][fea[j]] += 1 
      
    #   # print(self.condProb)
    #   for i in self.legalLabels:
    #     for j in trainingData[0]:
    #       self.condProb[i][j].normalize() 
      
    #   validation = copy.deepcopy(validationData)
    #   guesses = self.classify(validation)
    #   # print(len(guesses))
    #   acc_val = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True) * 100.0 / len(validationLabels)
    #   print("val acc is ", acc_val)
    #   if acc_val > acc_val_best:
    #     acc_val_best = acc_val
    #     best_smoothing = smoothing

    # print("best_smoothing is ", best_smoothing)  
    for i in self.legalLabels: #i for label
      self.condProb[i] = {}
        # print(trainingData[1])
      for j in trainingData[0]: #j for the key for the corresponding feature
          # print("j is ", j)
          self.condProb[i][j] = util.Counter()
          for k in feature: # k for corresponding value for jth feature in 1st case
                      # print("k is ", k)
              self.condProb[i][j][k] = self.k
      
    for n, fea in enumerate(trainingData):
        # print(n)
        i = trainingLabels[n] # true label for the nth training case
        for j in fea:
          # print(j)
          self.condProb[i][j][fea[j]] += 1 
      
      # print(self.condProb)
    for i in self.legalLabels:
        for j in trainingData[0]:
          self.condProb[i][j].normalize()
    
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    # logP(w|ci)P(ci)
    for i in self.legalLabels:
      logJoint[i] = math.log(self.prior[i])
      for j in datum:
        # print("j is ", datum[j])
        logJoint[i] += math.log(self.condProb[i][j][datum[j]])
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds