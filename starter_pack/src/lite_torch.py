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
    """
    @brief This is a simple implementation of a tensor class with automatic differentiation capabilities.
    The Tensor class supports basic operations such as addition, subtraction, multiplication, matrix multiplication, 
    transposition, logarithm, hyperbolic tangent, exponential, softmax, and cross-entropy loss. 
    Each operation creates a new Tensor object that keeps track of its children and the function to compute the partial derivatives for backpropagation. 
    The backward method computes the gradients for all tensors in the computational graph by traversing it in reverse order.

    @param data: The data to be stored in the tensor, can be a list or a numpy array.   
    @param children: A tuple of child tensors that were used to compute this tensor, used for backpropagation.
    @param dtype: The data type of the tensor, default is np.float32.
    """
    def __init__(self, data, children=(), dtype=np.float32):
        """
        @brief Initializes a Tensor object with the given data, children, and data type.
        @param data: The data to be stored in the tensor, can be a list or a numpy array.
        @param children: A tuple of child tensors that were used to compute this tensor, used for backpropagation.
        @param dtype: The data type of the tensor, default is np.float32.
        """
    
        self.dtype = dtype
        self.data = np.array(data, dtype=dtype)
        self.grad = np.zeros_like(data, dtype=np.float32)

        self.partial_diff = lambda: None
        self.children = children

    def sum(self, axis=0):
        """
        @brief Computes the sum of the tensor along a specified axis and returns a new Tensor object.
        @param axis: The axis along which to compute the sum, default is 0. (0 means summing over rows, 1 means summing over columns.)
        @return A new Tensor object containing the sum of the original tensor along the specified axis.
        """
        output = Tensor(np.sum(self.data, axis=axis, keepdim=True), children=(self,))

        def partial_diff():
            self.grad += np.broadcast_to(output.grad, self.data.shape)
        
        output.partial_diff = partial_diff
        return output
    
    def __add__(self, other):
        """
        @brief Adds two Tensor objects element-wise and returns a new Tensor object containing the result.
        @param other: Another Tensor object to be added to the current tensor.
        @return A new Tensor object containing the element-wise sum of the two tensors.
        """
        
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
        """
        @brief Subtracts one Tensor object from another element-wise and returns a new Tensor object containing the result.
        @param other: Another Tensor object to be subtracted from the current tensor.
        @return A new Tensor object containing the element-wise difference of the two tensors.
        """
        output = Tensor(self.data - other.data, children=(self, other), dtype=self.dtype)

        def partial_diff():
            self.grad += output.grad
            other.grad -= output.grad
        
        output.partial_diff = partial_diff
        return output
    
    def __mul__(self, other):
        """
        @brief Multiplies two Tensor objects element-wise and returns a new Tensor object containing the result.
        @param other: Another Tensor object to be multiplied with the current tensor.
        @return A new Tensor object containing the element-wise product of the two tensors.
        """
        output = Tensor(self.data * other.data, children=(self, other), dtype=self.dtype)

        def partial_diff():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        
        output.partial_diff = partial_diff
        return output

    def __matmul__(self, other):
        """
        @brief Performs matrix multiplication between two Tensor objects and returns a new Tensor object containing the result.
        @param other: Another Tensor object to be matrix-multiplied with the current tensor
        @return A new Tensor object containing the result of the matrix multiplication.
        """
        output = Tensor(self.data @ other.data, children=(self, other), dtype=self.dtype)

        def partial_diff():
            self.grad += output.grad @ other.data.T
            other.grad += self.data.T @ output.grad

        output.partial_diff = partial_diff
        return output
    
    @property
    def T(self):
        """
        @return A new Tensor object that is the transpose of the current tensor.
        """
        output = Tensor(self.data.T, children=(self,))

        def partial_diff():
            self.grad += output.grad.T
        
        output.partial_diff = partial_diff
        return output
    
    def log(self):
        """
        @brief Computes the natural logarithm of the current tensor element-wise and returns a new Tensor object containing the result.
        @return A new Tensor object containing the element-wise natural logarithm of the current tensor.
        """
        output = Tensor(np.log(self.data + 1e-8), children=(self,))

        def partial_diff():
            self.grad += output.grad / (self.data + 1e-8)
        
        output.partial_diff = partial_diff
        return output
    
    def tanh(self):
        """
        @brief Computes the hyperbolic tangent of the current tensor element-wise and returns a new Tensor object containing the result.
        @return A new Tensor object containing the element-wise hyperbolic tangent of the current tensor
        """
        output = Tensor(np.tanh(self.data + 1e-8), children=(self,))

        def partial_diff():
            self.grad += output.grad * (1 - np.tanh(self.data)**2)
        
        output.partial_diff = partial_diff
        return output
    
    def exp(self):
        """
        @brief Computes the exponential of the current tensor element-wise and returns a new Tensor object containing the result.
        @return A new Tensor object containing the element-wise exponential of the current tensor.
        """
        output = Tensor(np.exp(self.data), children=(self,))

        def partial_diff():
            self.grad += output.grad * np.exp(self.data)
        
        output.partial_diff = partial_diff
        return output

    def softmax(self, axis=-1):
        """
        @brief Computes the softmax of the current tensor along a specified axis and returns a new Tensor object containing the result.
        @param axis: The axis along which to compute the softmax, default is -1 (the last axis).
        @return A new Tensor object containing the softmax of the current tensor along the specified axis.
        """
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
        """
        @brief Computes the cross-entropy loss between the current tensor (predicted probabilities) and another tensor (true labels) along a specified axis, and returns a new Tensor object containing the result.
        @param other: Another Tensor object containing the true labels (one-hot encoded) to be compared with the current tensor (predicted probabilities).
        @param axis: The axis along which to compute the cross-entropy loss, default is -1 (the last axis).
        @return A new Tensor object containing the cross-entropy loss between the current tensor and the other tensor along the specified axis.
        """
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
        """
        @return A tuple representing the shape of the tensor's data.
        """
        
        return self.data.shape

    def backward(self):
        """
        @brief Computes the gradients for all tensors in the computational graph by traversing it in reverse order, starting from the current tensor.
        The method first constructs a list of tensors in the computational graph using a depth-first search approach, and then iterates through the list in reverse order to call the partial_diff function for each tensor, which computes the gradients based on the chain rule of differentiation.
        """
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