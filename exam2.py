import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """前向传播"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """随机梯度下降"""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """使用后向传播算法进行参数更新.mini_batch是一个元组(x, y)的列表、eta是学习速率"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回一个元组(nabla_b, nabla_w)代表目标函数的梯度."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传播
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """返回分类正确的个数"""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    """sigmoid函数的导数"""
    return sigmoid(z)*(1-sigmoid(z))
def vectorized_result(j,nclass):
    """离散数据进行one-hot"""
    e = np.zeros((nclass, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':

    #使用MINIST手写数据集
    ftrain = open('train-images.idx3-ubyte', 'rb').read()
    flabel = open('train-labels.idx1-ubyte', 'rb').read()
    ftest = open('t10k-images.idx3-ubyte', 'rb').read()
    ftest_label = open('t10k-labels.idx1-ubyte', 'rb').read()

    f_test = []
    f_test_label = []
    f_train = []
    f_label = []
    for item in open('train-images.idx3-ubyte', 'rb').read():
        f_train.append(item)

    for item in open('train-labels.idx1-ubyte', 'rb').read():
        f_label.append(item)

    for item in open('t10k-images.idx3-ubyte', 'rb').read():
        f_test.append(item)

    for item in open('t10k-labels.idx1-ubyte', 'rb').read():
        f_test_label.append(item)

    train_data = []
    test_data = []


    for k in range(60000):
        train_data.append(
            (np.array(f_train[16 + k * 784:16 + k * 784 + 784]).reshape(784, 1), vectorized_result(f_label[8 + k], 10)))

    for k in range(10000):
        test_data.append((np.array(f_test[16 + k * 784:16 + k * 784 + 784]).reshape(784, 1), f_test_label[8 + k]))

    net = Network(sizes=[784, 30 , 30 , 10])
    net.SGD(training_data=train_data, epochs=300, mini_batch_size=10, eta=0.1, test_data=test_data)