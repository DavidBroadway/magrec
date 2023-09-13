
import torch
import abc


class SquaredError(torch.nn.Module):
    """
    Implements the sum of squared errors in space dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Computes the squared error of the input.

        Parameters
        ----------
        x : torch.tensor
            The values for which the squared error should be computed.
        """
        return torch.sum(torch.square(x), dim=1)
    
    
class Condition(abc.ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
    
    @abc.abstractmethod
    def sample(self):
        pass
    
    @abc.abstractmethod
    def loss(self):
        pass
    
    

class TPCondition(torch.nn.Module):
    """
    A general condition which should be optimized or tracked.

    Parameters
    -------
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.
    track_gradients : bool
        Whether to track input gradients or not. Helps to avoid tracking the
        gradients during validation. If a condition is applied during training,
        the gradients will always be tracked.
    """

    def __init__(self, name=None, weight=1.0, requires_grad=True):
        super().__init__()
        self.name = name
        self.weight = weight
        self.requires_grad = requires_grad

    @abc.abstractmethod
    def forward(self, device='cpu', iteration=None):
        """
        The forward run performed by this condition. Every
        derived condition should implement it. Since it is 
        a torch.nn.Module, .forward() is called automatically
        by __call__().

        Returns
        -------
        torch.Tensor : the loss which should be minimized or monitored during training
        """
        raise NotImplementedError

    # TODO: REMOVE until "RETURN"
    def _setup_data_functions(self, data_functions, sampler):
        for fun in data_functions:
            data_functions[fun] = UserFunction(data_functions[fun])
        if isinstance(sampler, StaticSampler):
            # functions can be evaluated once
            for fun in data_functions:
                points = sampler.sample_points()
                data_fun_points = data_functions[fun](points)
                #self.register_buffer(fun, data_fun_points)
                data_functions[fun] = UserFunction(data_fun_points)
        return data_functions

    def _move_static_data(self, device):
        pass


class PINNCondition(TPCondition):

    def __init__(self, module, sampler, residual_fn, 
                 error_fn=SquaredError(), reduce_fn=torch.mean, name='pinncondition', **kwargs):
        super().__init__(**kwargs)
        """
        
        Parameters
        ----------
        residual_fn : callable
            A user-defined function that computes the residual (unreduced loss) from
            inputs and outputs of the model, e.g. by using utils.differentialoperators
            and/or domain.normal, it computes a deviation from the physical model by sample. 
        error_fn : callable
            Function that will be applied to the output of the residual_fn to compute
            the unreduced loss. Should reduce only along the 2nd (i.e. space-)axis.    
        reduce_fn : callable
            Function that will be applied to reduce the loss to a scalar. Defaults to
            torch.mean.

        
        """
        self.module = module
        self.sampler = sampler
        self.residual_fn = residual_fn
        self.error_fn = error_fn
        self.reduce_fn = reduce_fn
        self.name = name
        
    def forward(self, device='cpu', iteration=None):
        x = self.sampler.sample_points(device=device)
        x = x.requires_grad_(True)
        y = self.module(x)
        # Here the order of the arguments is defined
        # in the defition of the residual function. 
        # It would be nice to remove this interdependency
        # using UserFunction(). 
        residuals = self.residual_fn(y, x)
        unreduced_loss = self.error_fn(residuals)
        return self.reduce_fn(unreduced_loss)