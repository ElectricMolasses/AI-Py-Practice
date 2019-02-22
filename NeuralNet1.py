import numpy as np

def sigmoid(s):
    # activation function
    return 1/(1 + np.exp(-s))

class Neural_Network(object):
    def __init__(self):
        #params
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
            #(3x2) weight matrix, input to first hidden.
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
            #(3x1) weight matrix to move from hidden to output.
            
    def forward(self, X):
        #propogate forwards
        self.z = np.dot(X, self.W1) #dot product to push current to next layer
        self.z2 = sigmoid(self.z) #applies activation function
        self.z3 = np.dot(self.z2, self.W2)
            #dot product of hidden layer and second set of 3x1 weights
        o = sigmoid(self.z3) #Final activation func application
        return o


# X = (Hours sleeping, Hours studying), y = test score

X = np.array(([2, 9], [1, 5], [3, 6]), dtype = float)
y = np.array(([92], [86], [89]), dtype = float)
print(X)

# scale down units
X = X/np.amax(X, axis = 0) #Converts each number to a decimal of itself/max on axis.
y = y/100 # max test score is 100, convert to decimal/percent.

print(X)

# We're going to pass two points of information into 3 hidden layers, and then straight
# into an output.

# We also need an activation function which is going to make the output less linear.
# For the sake of simplicity on the first attempt, I'm just going to use a sigmoid function.
# Later, I'm going to try creating my own functions for specific tasks.

NN = Neural_Network()

o = NN.forward(X)

print "Predicted Output: \n" + str(o)
print "Actual Output: \n" + str(y)