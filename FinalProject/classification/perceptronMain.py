import perceptron
import mira
import samples
import sys
import util
import time

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
ITERATIONS = 3

def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def runClassifier():
  #Get Inputs
  dataset = input('Choose dataset (faces or digits): ')
  training_val = int(input('Enter amount of training data: '))
  testing_amount = int(input('Enter the amount of data to test: '))
  
  if dataset == "faces":
    featureFunction = basicFeatureExtractorFace
    legalLabels = range(2)
  
  if dataset == "digits":
    featureFunction = basicFeatureExtractorDigit
    legalLabels = range(10)

  classifier = perceptron.PerceptronClassifier(legalLabels, ITERATIONS)
  
  #Load data
  numTraining = training_val
  numTest = testing_amount

  if dataset == "faces":
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  
  if dataset == "digits":
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


  #Extract features
  print("Extracting features...")
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)

  #Conduct training
  print("Training...")
  test_begintime = time.time()
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  test_endtime = time.time()
  test_duration = test_endtime - test_begintime
  
  #Conduct testing
  print("Testing...")
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))

  #Training Time
  print("Training time: " + str(test_duration) + " seconds.")

if __name__ == '__main__':
  runClassifier()