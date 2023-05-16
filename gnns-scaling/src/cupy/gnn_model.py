from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import utils

class Context:
    """Context class for storing data in a layer."""
    def __init__(self) -> None:
        self._ctx_data: Dict[str, Any] = {}

    def __getattr__(self, __name: str) -> Any:
        if __name == "_ctx_data":
            return super().__getattr__(__name)
        elif __name in self._ctx_data:
            return self._ctx_data[__name]
        else:
            raise AttributeError(f"Context does not have data {__name}.")

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "_ctx_data":
            super().__setattr__(__name, __value)
        elif __value is None and not __name in self._ctx_data:
            raise AttributeError(f"Attempting to free unset data {__name}.")
        else:
            self._ctx_data[__name] = __value
class Parameter:
    """Parameter in a layer."""
    def __init__(self, data: np.ndarray, cache_grad: bool = True) -> None:
        self._param_data: np.ndarray = data.copy()
        self._param_grad: np.ndarray = np.zeros_like(data) if cache_grad else None
    
    def zero_grad(self):
        """Set the gradient to zero."""
        if self._param_grad is not None:
            self._param_grad.fill(0)

    def accumulate_grad(self, grad: np.ndarray):
        """Accumulate the gradient."""
        if self._param_grad is not None:
            self._param_grad += grad

class Parameters:
    """Parameters class for storing parameters in a layer."""
    def __init__(self) -> None:
        self._params: Dict[str, Parameter] = {}

    def __getattr__(self, __name: str) -> Parameter:
        if __name == "_params":
            return super().__getattr__(__name)
        elif __name in self._params:
            return self._params[__name]
        else:
            raise AttributeError(f"No parameter named: {__name}.")

    def __setattr__(self, __name: str, __value: Parameter) -> None:
        if __name == "_params":
            super().__setattr__(__name, __value)
        else:
            self._params[__name] = __value

    def zero_grad(self):
        """Set the gradient of all parameters to zero."""
        for param in self._params.values():
            param.zero_grad()



class GnnLayer:
    def __init__(self, in_channel: int, out_channel: int, use_gpu: bool) -> None:
        """Initialize a GNN layer."""
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ctx = Context()
        self.use_gpu = use_gpu
        # If cache_data is True, then the layer will cache the intermediate variables in self.ctx for backward compute.
        self.cache_data = False
        self.parameters = Parameters()

        # If is_first_layer is True, then backward pass will not compute gradient of input matrix.
        self.is_first_layer = False

    def init_parameters(self, rng, dtype, cache_grad: bool = True):
        """Initialize the parameters of the layer."""
        raise NotImplementedError()

    def force_set_parameters(self, cache_grad: bool = True, **kwargs):
        """Force set parameters of the layer.
        
            :param cache_grad: Whether to cache the gradient of the parameters.
            :param kwargs: The parameters to set. Key is the parameter name, value is the parameter data in numpy array.
        """
        for key, value in kwargs.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Parameter {key} is not a numpy array.")
            self.parameters.__setattr__(key, Parameter(value.copy(), cache_grad))
    
    def __getattr__(self, __name: str) -> Any:
        """Access parameter data by layer.parameter_name."""
        if __name in self.parameters._params:
            return self.parameters._params[__name]._param_data
        else:
            raise AttributeError(f"Layer does not have parameter {__name}.")

    def forward(self, A, input):
        """Forward pass of the layer.
        
            :param A: The adjacency matrix.
            :param input: The input matrix from previous layer.
            :return: The output matrix.
        """

        if self.use_gpu:
            return self.forward_gpu(A, input)
        else:
            return self.forward_cpu(A, input)

    def forward_gpu(self, A, input):
        """Forward pass of the layer on GPU.
        
            :param A: The adjacency matrix.
            :param input: The input matrix from previous layer.
            :return: The output matrix.
        """

        raise NotImplementedError()

    def forward_cpu(self, A, input):
        """Forward pass of the layer on CPU.
        
            :param A: The adjacency matrix.
            :param input: The input matrix from previous layer.
            :return: The output matrix.
        """

        raise NotImplementedError()

    def backward(self, A, grad_out):
        """Backward pass of the layer.
        
            :param A: The adjacency matrix.
            :param grad_out: The gradient from next layer.
            :return: The gradient of the input matrix.
        """

        if self.use_gpu:
            return self.backward_gpu(A, grad_out)
        else:
            return self.backward_cpu(A, grad_out)

    def backward_gpu(self, A, grad_out):
        """Backward pass of the layer on GPU.
        
            :param A: The adjacency matrix.
            :param grad_out: The gradient from next layer.
            :return: The gradient of the input matrix.
        """

        raise NotImplementedError()

    def backward_cpu(self, A, grad_out):
        """Backward pass of the layer on CPU.
        
            :param A: The adjacency matrix.
            :param grad_out: The gradient from next layer.
            :return: The gradient of the input matrix.
        """

        raise NotImplementedError()


class GnnModel:
    def __init__(self, layers: List[GnnLayer], inference_only=False) -> None:
        """Initialize a GNN model."""
        self.layers = layers
        self.num_layers = len(layers)
        self.inference_only = inference_only

        self.set_cache(inference_only)
        self.layers[0].is_first_layer = True

    def init_parameters(self, rng, dtype):
        """Initialize the parameters of the model."""
        for layer in self.layers:
            layer.init_parameters(rng, dtype, cache_grad=not self.inference_only)

    def set_cache(self, inference_only: bool):
        """ Set the cache_data flag for each layer.
            If inference_only is True, then the model will not cache 
            the intermediate variables in layer.ctx for backward compute.
        """
        for layer in self.layers:
            layer.cache_data = not inference_only

    def forward(self, A, input):
        """ Forward pass of the model.
            For each layer, invokes the forward method of each layer in the model,
            and then invokes the redistribute_between_layers_forward method."""
        for layer in self.layers:
            input = layer.forward(A, input)
            if layer != self.layers[-1]:
                input = self.redistribute_between_layers_forward(input)
        return input

    def backward(self, A, grad_out):
        """ Backward pass of the model.
            For each layer, invokes the backward method of each layer in the model,
            and then invokes the redistribute_between_layers_backward method."""
        for layer in reversed(self.layers):
            grad_out = layer.backward(A, grad_out)
            if layer != self.layers[0]:
                grad_out = self.redistribute_between_layers_backward(grad_out)

    def redistribute_between_layers_forward(self, out):
        """Redistribute the output of each layer to the next layer."""
        raise NotImplementedError()

    def redistribute_between_layers_backward(self, grad_out):
        """Redistribute the gradient of each layer to the previous layer."""
        raise NotImplementedError()

    def redistribute_forward_output(self, out):
        """Redistribute the output of the last layer to loss function."""
        # Redistribute the output of the last layer to loss function
        raise NotImplementedError()

    def redistribute_loss_grad(self, grad_out):
        """Redistribute the gradient of loss function to the last layer."""
        # Redistribute the gradient of loss function to the last layer
        raise NotImplementedError()


class Loss:
    """Sum of Squared Error loss function."""
    def __init__(self, model: GnnModel, A) -> None:
        self.model = model
        self.A = A

    def backward(self, input, target):
        """ Invoke the redistribute_forward_output method to redistribute
            the output of the last layer to loss function.
            Invoke the backward method of the model.
            Invoke the redistribute_loss_grad method to redistribute
            the gradient of loss function to the last layer."""
        out = self.model.redistribute_forward_output(input)
        grad = out - target
        grad = self.model.redistribute_loss_grad(grad)
        self.model.backward(self.A, grad)

class Optimizer:
    """Base class of optimizer."""
    def __init__(self, model: GnnModel, lr: float) -> None:
        """Initialize an optimizer."""
        self.model = model
        self.lr = lr

    def step(self):
        """Update the parameters of the model."""
        for layer in self.model.layers:
            for param in layer.parameters._params.values():
                param._param_data -= self.lr * param._param_grad
