import numpy as np

class Network(object):
    
    def  __init__(self, sizes):
        #define Hyperparameters
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        
    def forward(self, X):
        #return output when input is X
        a = X
        self.Z = []
        self.A = [X]
        for w in self.weights:
            zi = np.dot(a, w)
            self.Z.append(zi)
            a = self.sigmoid(zi)
            self.A.append(a)
        return self.A[-1]
    
    def back(self, X, y):
        #backpropagates error to weights from inputs X and expected output O
        #derivaive of sigmoid(output sum[sigmoid**-1(forward)])
        dj = self.costFunctionPrime(X, y)
        scalar = 3
        for i in range(0, len(dj)):
            self.weights[i] = self.weights[i] - scalar*dj[i]  
        cost2 = self.costFunction(X, y)
        return cost2
        
    
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        s = [0.5*(y-yH)**2 for y, yH in zip(y, self.yHat)]
        return sum(s)
    
    def costDerivative(self, oa, y):
        return y-oa
    
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        dJdWx = [[0]]*(self.numLayers-1)
        #first delta
        d = np.multiply(self.costDerivative(y, self.yHat), self.sigmoidPrime(self.Z[-1]))
        dJdWx[-1] = np.dot(self.A[-2].T, d)
        #propagation
        for l in range(2, self.numLayers):
            zi = self.Z[-l]
            sp = self.sigmoidPrime(zi)
            d = np.multiply(np.dot(d, self.weights[-l+1].T), sp)
            dJdWx[-l] = np.dot(self.A[-l-1].T, d)
        return dJdWx
