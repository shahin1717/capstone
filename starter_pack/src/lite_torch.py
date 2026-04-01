import numpy as np

"""
How to use it:
    # Create two numbers
    a = Tensor([10])
    b = Tensor([5])
    
    # Do some math
    c = a + b
    
    # To calculate the gradients (gradients)
    c.backward()
    
    # To check the gradient of a variable 'a'
    print(a.grad)
"""

class Tensor:
    def __init__(self, data, children=(), dtype=np.float32):
        self.dtype = dtype
        self.data = np.array(data, dtype=dtype)
        self.grad = np.zeros_like(data, dtype=np.float32)

        self.partial_diff = lambda: None
        self.children = children

    def sum(self, axis=0):
        output = Tensor(np.sum(self.data, axis=axis, keepdim=True), children=(self,))

        def partial_diff():
            self.grad += np.broadcast_to(output.grad, self.data.shape)
        
        output.partial_diff = partial_diff
        return output
    
    def __add__(self, other):
        output = Tensor(self.data + other.data, children=(self, other), dtype=self.dtype)

        def partial_diff():
            def broadcast_grad(grad, original_shape):
                for index, (grad_dim, original_dim) in enumerate(zip(grad.shape, original_shape)):
                    if original_dim == 1:
                        grad = grad.sum(axis=index, keepdims=True)
                
                return grad
            
            self.grad += broadcast_grad(output.grad, self.data.shape)
            other.grad += broadcast_grad(output.grad, other.data.shape)
        
        output.partial_diff = partial_diff
        return output
    
    def __sub__(self, other):
        output = Tensor(self.data - other.data, children=(self, other), dtype=self.dtype)

        def partial_diff():
            self.grad += output.grad
            other.grad -= output.grad
        
        output.partial_diff = partial_diff
        return output
    
    def __mul__(self, other):
        output = Tensor(self.data * other.data, children=(self, other), dtype=self.dtype)

        def partial_diff():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        
        output.partial_diff = partial_diff
        return output

    def __matmul__(self, other):
        output = Tensor(self.data @ other.data, children=(self, other), dtype=self.dtype)

        def partial_diff():
            self.grad += output.grad @ other.data.T
            other.grad += self.data.T @ output.grad

        output.partial_diff = partial_diff
        return output
    
    @property
    def T(self):
        output = Tensor(self.data.T, children=(self,))

        def partial_diff():
            self.grad += output.grad.T
        
        output.partial_diff = partial_diff
        return output
    
    def log(self):
        output = Tensor(np.log(self.data + 1e-8), children=(self,))

        def partial_diff():
            self.grad += output.grad / (self.data + 1e-8)
        
        output.partial_diff = partial_diff
        return output
    
    def tanh(self):
        output = Tensor(np.tanh(self.data + 1e-8), children=(self,))

        def partial_diff():
            self.grad += output.grad * (1 - np.tanh(self.data)**2)
        
        output.partial_diff = partial_diff
        return output
    
    def exp(self):
        output = Tensor(np.exp(self.data), children=(self,))

        def partial_diff():
            self.grad += output.grad * np.exp(self.data)
        
        output.partial_diff = partial_diff
        return output

    def softmax(self, axis=-1):
        numerator = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        denominator = np.sum(numerator, axis=axis, keepdims=True)
        output = Tensor(numerator / denominator, children=(self, ))

        def partial_diff():
            s = output.data
            g = output.grad
            sum_g_s = np.sum(g * s, axis=axis, keepdims=True)
            self.grad += s * (g - sum_g_s)
        
        output.partial_diff = partial_diff
        return output
    
    def crossentropy(self, other, axis=-1):
            log_probs = self.log()
            res = (other * log_probs)
            negate_tensor = Tensor((np.ones_like(res) * -1))
            res = res * negate_tensor
            output = Tensor(np.sum(res.data, axis=axis), children=(self, other))

            def partial_diff():
                grad = output.grad[..., np.newaxis] if output.grad.ndim == 1 else output.grad
                self.grad += grad * -(other.data / (self.data + 1e-8))
                other.grad += grad * -np.log(self.data + 1e-8)
            
            output.partial_diff = partial_diff
            return output
    
    def shape(self):
        return self.data.shape

    def backward(self):
        graph = []
        visited = set()
        added_to_graph = set()
        stack = [self]

        while stack:
            node = stack[-1]
            
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    if child not in visited:
                        stack.append(child)
            else:
                node = stack.pop()
                if node not in added_to_graph:
                    graph.append(node)
                    added_to_graph.add(node)
        
        self.grad = np.ones_like(self.grad)
        
        for node in reversed(graph):
            node.partial_diff()


    def __repr__(self,):
        return f" Tensor : {self.data}, tensor_shape : ({self.data.shape})"