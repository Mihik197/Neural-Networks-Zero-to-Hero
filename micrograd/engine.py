import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # the gradient. the derivative of the output, with respect to the input(s). i.e. the change in output with change in input. we keep the default value when starting out at 0
        self._backward = lambda: None  # by default doesn't do anything, empty function. for leaf nodes
        self._prev = set(_children) # children is the set of nodes on which the operation is performed
        self._op = _op  # op is the operation
        self.label = label  # to help us track variables along with their values

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # if we want to do something like say, b = a + 1, (where 1 is an int so it doesn't have data attribute) this checks if its an instance of Value type otherwise it wraps it in a Value object
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad # we use '+=' instead of '=' because of the multivariable chain rule, in which gradients add up. this is in the case when one variable is used more than once.
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad  # exercise
        out._backward = _backward

        return out
    
    def __truediv__(self, other):  # self / other
        return self * other**-1
    
    def __sub__(self, other):  # self - other
        return self + (-other)
    
    def log(self):
        x = self.data
        out = Value(math.log(x), (self, ), 'log')

        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):  # other * self   eg: 1 + a, which would give an error if using __mul__
        return self * other
    
    def __rsub__(self, other):
        return self - other
    
    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):  # -self
        return self * -1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # exercise
        out._backward = _backward

        return out

    def backward(self):
        # topological sort i.e. DFS
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # going back and getting the gradient by applying the chain rule
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()