import matplotlib.pyplot as plt

trainPercetage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # training data percentage

# perceptron for digits
# time = [4.196268082, 8.464776993, 12.73568797, 16.91051579, 21.11814904, 25.38081312, 29.57767105, 33.99771786, 38.23259902, 42.25947809]
# validationAccuracy = [73.50, 77.50, 79.30, 77.00, 82.10, 82.60, 83.70, 81.70, 81.10, 82.00]
# testAccuracy = [68.60, 74.00, 76.50, 75.30, 78.90, 80.30, 80.20, 77.10, 80.10, 78.30]

# perceptron for faces
time = [0.4639101028, 0.9323031902, 1.383772135, 1.823302746, 2.260907888, 2.689082861, 3.152004004, 3.552192688, 3.995384216, 4.417536974]
validationAccuracy = [70.00, 84.70, 92.00, 96.00, 96.70, 99.30, 96.70, 94.70, 98.70, 98.00]
testAccuracy = [60.70, 74.00, 75.30, 82.70, 80.00, 86.70, 81.30, 81.30, 84.00, 84.70]

# accuracy plot
# plt.title("Perceptron Algorithm for Faces")
# plt.xlabel("Used Training Data (%)")
# plt.ylabel("Accuracy (%)")
# plt.plot(trainPercetage, validationAccuracy,label="Validation Accuracy")
# plt.plot(trainPercetage, testAccuracy, label="Test Accuracy")
# plt.xticks(trainPercetage,trainPercetage)
# plt.legend()
# plt.show()

# time plot
plt.title("Perceptron Algorithm for Faces")
plt.xlabel("Used Training Data (%)")
plt.ylabel("Training Time (sec)")
plt.plot(trainPercetage, time)
plt.xticks(trainPercetage,trainPercetage)
plt.legend()
plt.show()