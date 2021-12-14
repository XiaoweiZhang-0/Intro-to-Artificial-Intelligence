import matplotlib.pyplot as plt

trainPercetage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # training data percentage

# perceptron for digits
# time = [4.196268082, 8.464776993, 12.73568797, 16.91051579, 21.11814904, 25.38081312, 29.57767105, 33.99771786, 38.23259902, 42.25947809]
# validationAccuracy = [73.50, 77.50, 79.30, 77.00, 82.10, 82.60, 83.70, 81.70, 81.10, 82.00]
# testAccuracy = [68.60, 74.00, 76.50, 75.30, 78.90, 80.30, 80.20, 77.10, 80.10, 78.30]

# perceptron for faces
# time = [0.4639101028, 0.9323031902, 1.383772135, 1.823302746, 2.260907888, 2.689082861, 3.152004004, 3.552192688, 3.995384216, 4.417536974]
# validationAccuracy = [70.00, 84.70, 92.00, 96.00, 96.70, 99.30, 96.70, 94.70, 98.70, 98.00]
# testAccuracy = [60.70, 74.00, 75.30, 82.70, 80.00, 86.70, 81.30, 81.30, 84.00, 84.70]

# naive Bayes for digits
# time = [0.6341879368, 0.7790851593, 0.9930491447, 1.089653015, 1.25755024, 1.419877052, 1.591797829, 1.731026173, 1.897032976, 2.093582869]
# validationAccuracy = [74.8, 78.5, 79.5, 79.7, 80.9, 80.6, 81.1, 81.4, 81.9, 81.7]
# testAccuracy = [69.6, 72.9, 74.6, 74.9, 76.4, 76.4, 76.5, 77.4, 76.8, 77.4]

# naive Bayes for faces
# time = [2.709470987, 2.812730312, 2.863176823, 3.037161827, 3.035547972, 3.065768003, 3.175688028, 3.258857012, 3.296796799, 3.442404032]
# validationAccuracy = [64, 85.3, 96, 98, 97.3, 96, 96.7, 96, 94.7, 94]
# testAccuracy = [53.3, 82, 86, 86.7, 84.7, 86.7, 86.7, 88, 88.7, 88]

# log regression for digits

# log regression for faces
time = [0.9111790657, 1.257606268, 1.533348083, 1.682288885, 1.84480381, 1.851242065, 2.054958105, 2.483216047, 2.511002064, 2.666673899]
validationAccuracy = [82.7, 92, 98, 100, 100, 100, 100,	100, 100, 100]
testAccuracy = [82, 84, 86, 88.7, 76.7, 86, 84, 85.3, 80.7, 89.3]

# accuracy plot
plt.title("Log Regression Algorithm for Faces")
plt.xlabel("Used Training Data (%)")
plt.ylabel("Accuracy (%)")
plt.plot(trainPercetage, validationAccuracy,label="Validation Accuracy")
plt.plot(trainPercetage, testAccuracy, label="Test Accuracy")
plt.xticks(trainPercetage,trainPercetage)
plt.legend()
plt.show()

# time plot
# plt.title("Log Regression Algorithm for Faces")
# plt.xlabel("Used Training Data (%)")
# plt.ylabel("Training Time (sec)")
# plt.plot(trainPercetage, time)
# plt.xticks(trainPercetage,trainPercetage)
# plt.legend()
# plt.show()