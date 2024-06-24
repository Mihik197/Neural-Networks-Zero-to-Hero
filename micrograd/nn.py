import random
from micrograd.engine import Value

# implementation of neurons in a pytorch like way

class Neuron:
    def __init__(self, nin):  # nin: number of inputs to the neuron
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # creates random list of weights
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # (w * x) + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)  # sum takes in an optional 2nd parameter 'start' which by default is 0. we change it to the bias so we don't have to add it separately
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):  # nout: number of neurons in that layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]  # a one line list comprehension for the below code
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params

class MLP:
    def __init__(self, nin, nouts):  # nouts: is a list of nouts, i.e. a list of the number of neurons in each layer
        sz = [nin] + nouts  # combine them into one list since the nin (inputs) is in a way the first layer. refer to pic above
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]  # creates layers, by iterating over consecutive pairs, using the one layer as the number of inputs, and other as the number of neurons for that layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
