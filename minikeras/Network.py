import numpy as np

def ReLu(x):
    return np.maximum(x, 0.0)

def ReLuprime(x):
    return np.where(x > 0, 1.0, 0.0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidprime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class DenseNetwork(object):
    """ Basic neural network with fully connected layers
    # Arguments
        layers: a list of integers such that layers[l] = neurons in layer l
        sigma: activation function
        sigmaprime: derivative of activation function
    # References
        - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
    """
    def __init__(self, layers, sigma=ReLu, sigmaprime=ReLuprime, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.L = len(layers)
        self.sigma = sigma
        self.sigmaprime = sigmaprime
        self.weights = []
        self.biases = []
        for l in range(1, self.L):
            self.weights.append(np.random.rand(layers[l], layers[l-1]))
            self.biases.append(np.random.rand(layers[l], 1))

    def propagate(self, x):
        activation = x
        for w, b in zip(self.weights, self.biases):
            activation = self.sigma(np.dot(w, activation) + b)
        return activation
    
    def evaluate(self, x):
        a = self.propagate(x)
        return np.argmax(a)

    def backpropagate(self, x, y):
        batch_size = x.shape[1]
        delCW = [np.zeros(w.shape) for w in self.weights]
        delCB = [np.zeros(b.shape) for b in self.biases]

        Z = []
        A = [x]
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigma(z)
            A.append(a)
            Z.append(z)
        
        delta = (a - y) * self.sigmaprime(z) 
        delCW[-1] = np.dot(delta, A[-2].T) / batch_size
        delCB[-1] = delta.sum(axis=1, keepdims=True) / batch_size

        for l in range(2, self.L):
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmaprime(Z[-l]) / batch_size
            delCW[-l] = np.dot(delta, A[-l-1].T) / batch_size
            delCB[-l] = delta.sum(axis=1, keepdims=True) / batch_size
        
        return delCW, delCB
    
    def SGD(self, x, y):
        lr = self.learning_rate
        delCW, delCB = self.backpropagate(x, y)
        self.weights = [w - dw * lr for w, dw in zip(self.weights, delCW)]
        self.biases = [b - db * lr for b, db in zip(self.biases, delCB)]


if __name__ == '__main__':
    '''
        Basic test showing network learning the function
        f(x) = 2*x
    '''

    net = DenseNetwork([2, 50, 2])
    batch_size = 100
    print '---before training---'
    xrand = np.array([[3]
                    ,[5]])
    print 'f({}) = \n {}'.format(xrand, net.propagate(xrand))

    print '---training...---'
    num_epochs = 10000
    for epoch in xrange(num_epochs):
        x = np.random.rand(2, batch_size)
        y = 2*x
        net.SGD(x,y)

    print '---after training---'
    for _ in xrange(1):
        x = np.array([[3], [5]])
        print 'outputs: {}'.format(net.propagate(x))




