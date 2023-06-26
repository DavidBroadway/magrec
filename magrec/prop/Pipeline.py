import fnmatch
import inspect
import json
import warnings
import weakref
from collections import OrderedDict, defaultdict
from functools import partial, wraps

import numpy as np
import skorch
import torch
# the way it is now is a hard dependence on IPython, which prevents the use of magrec in a non-interactive environment
# TODO: make this optional and make `display` for flexible
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
# anticipate the change of the API in scikit-learn which moves Pipeline and FeatureUnion to sklearn.compose
from sklearn.utils import Bunch
from skorch import History
from skorch.callbacks import Callback, EpochTimer, PassthroughScoring, PrintLog
from skorch.utils import open_file_like

from magrec.misc.plot import plot_n_components
from magrec.prop.Fourier import FourierTransform2d

from magrec.prop.Propagator import (AxisProjectionPropagator,
                                    CurrentPropagator2d)


def identity(X):
    """The identity function."""
    return X


class Step(object):
    def __init__(self):
        self.enabled = True
        self.visual = False
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
    
    def __call__(self, X, y=None, **params):
        return self.fit(X, y, **params).transform(X, y, **params)


class Pipe(object):
    def __init__(self, steps, *, memory=None, verbose=False):
        # to allow for unnamed steps, see _name_estimators from sklearn.pipeline, 
        # for now we require names for each step
        self.steps: OrderedDict = OrderedDict(steps)
        self.memory = memory
        self.verbose = verbose
        self.fitted = False

    def add_step(self, step_name, step, after=None, before=None):
        """Add a named step `step` to the pipeline"""
        if after is not None or before is not None:
            raise NotImplementedError("Step position specification is not implemented yet. "
                                      "Use `add_step` method consequentially in order steps need "
                                      "to be executed.")
        self.steps.update((step_name, step))
        return self
    
    def _iter(self, filter=()):
        """
        Generate (idx, name, trans) tuples from self.steps. Omit those in filter tuple.

        Args:
            filters (tuple):  tuple of strings to filter out of the iteration, accepts glob patterns
        """
        filter_fns = list(
            f if callable(f) else partial(fnmatch.fnmatch, pat=f) for f in filter
        )

        for idx, (name, step) in enumerate(self.steps.items()):

            # skip if any of the filters is matched
            skip = any(
                filter_fn(name) or filter_fn(step.__class__.__name__.lower()) or filter_fn(str(step))
                for filter_fn in filter_fns
            )

            if skip:
                continue
            else:
                yield idx, name, step

    def fit(self, X, y=None, **fit_params):
        for idx, step_name, step in self._iter(filter=("passthrough", "plot*")):
            params = self.get_step_params(step_name, **fit_params)
            X = step.fit(X, y, **params).transform(X, visual=False)

        self.fitted = True
        return self
    
    def __call__(self, X, y=None, **params):
        return self.fit(X, y, **params).transform(X, y, **params)

    def get_step_params(self, step_name, **fit_params):
        """Get parameters for a step.

        By default, the parameters are passed to the step's constructor.
        """
        return fit_params.get(step_name, {})

    def check_params(self, **params):
        params_steps = {name: {} for name, step in self.steps if step is not None}
        for pname, pval in params.items():
            if "__" not in pname:
                raise ValueError(
                    "Pipe does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipe.fit(X, y, projection__theta"
                    "=theta)`.".format(pname)
                )
            step, param = pname.split("__", 1)
            params_steps[step][param] = pval
        return params_steps

    def transform(self, X, y=None, **transform_params):
        filters = ("passthrough",)
        visual = transform_params.get("visual", True)
        if not visual:
            filters += ("plot*",)

        for idx, step_name, step in self._iter(filter=filters):
            X = step.transform(X, y, **transform_params)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        for idx, step_name, step in self._iter(filter=("passthrough*", "plot*")):
            X = step.fit_transform(X, y, **fit_params)
        return X

    def propagate(self, X, y=None, **params):
        if not self.fitted:
            self.fit(X, y)
        return self.transform(X, y, **params)

    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(
                self.steps[ind], memory=self.memory, verbose=self.verbose
            )
        try:
            est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est
    
    def __setitem__(self, ind, value):
        self.steps[ind] = value

    @property
    def named_steps(self):
        """Access the steps by name.

        Read-only attribute to access any step by given name.
        Keys are steps names and values are the steps objects."""
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.steps))


class Union(Pipe):
    def __init__(self, transformer_list, dim, *, n_jobs=None, verbose=False):
        super().__init__(transformer_list)
        self.dim = dim
        self.n_jobs = n_jobs

    def _iter(self, filter=()):
        """
        Generate (idx, name, trans) tuples from self.steps. Replace 'passthrough' with identity transformation.
        """
        for idx, step_name, step in super()._iter(filter=filter):
            if step == "passthrough":
                step = Function(func=identity)
            yield idx, step_name, step

    def transform(self, X, y=None, **fit_params):
        Xs = tuple(step.transform(X, y, **fit_params) for _, _, step in self._iter())
        try:
            X = torch.cat(Xs, dim=self.dim)
        except IndexError:
            X = torch.stack(Xs, dim=self.dim)
        return X

    def fit(self, X, y=None, **fit_params):
        for idx, step_name, step in self._iter(filter=("plot", "passthrough")):
            X = step.fit_transform(X, y)
        return self


class Multiplexer(Step):
    """
    Multiplexer allows to create multiple outputs from a single input and then apply `func` to them.
    By default `func` is Identity, so the output is a tuple of outputs from each step.
    """

    def __init__(self, transformer_list, func, *, n_jobs=None, verbose=False):
        super().__init__(transformer_list, verbose=verbose)
        self.func = func
        self.n_jobs = n_jobs

    def _iter(self, filter=()):
        """
        Generate (idx, name, trans) tuples from self.steps. Replace 'passthrough' with identity transformation.
        """
        for idx, step_name, step in super()._iter(filter=filter):
            if step == "passthrough":
                step = Function(func=identity)
            yield idx, step_name, step

    def transform(self, X, y=None, **fit_params):
        Xs = tuple(step.transform(X, y, **fit_params) for _, _, step in self._iter())
        X = torch.cat(Xs, dim=self.dim)
        return X

    def fit(self, X, y=None, **fit_params):
        for idx, step_name, step in self._iter(filter=("plot", "passthrough")):
            X = step.fit_transform(X, y)
        return self


class Function(Step):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return self.func(X)


class FourierDivergence2d(Step):
    """Calculates divergence of a 2d signal in Fourier space.

    Implements the formula:

    .. math::
        F[∇•J(x,y)] = i k_x F[J_x] + i k_y F[J_y]

    where F is Fourier transform, k_x and k_y are spatial frequencies.
    Note the sign, which is due to the defintion of the Fourier transform
    adopted here in :class:`FourierTransform2d`.

    """

    def __init__(self, keep_dim=False):
        super().__init__()
        self.real_signal = False
        self.keep_dim = keep_dim
        # dimensions at which divergence is computed
        self.dim = (-2, -1)
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(real_signal={self.real_signal})"

    def fit(self, X, y=None, **fit_params):
        if len(X.shape) < 3:
            raise ValueError(
                "X must have number of dimensions (..., components, height, width) > 3, "
                "got {}".format(len(X.shape))
            )

        self.ft = FourierTransform2d(
            grid_shape=X.shape, dx=1, dy=1, real_signal=self.real_signal, type="linear"
        )
        return self

    def transform(self, X, y=None, **fit_params):
        Xf = self.ft.forward(X, dim=self.dim)
        Yf = (
            1j * Xf[..., 0, :, :] * self.ft.kx_vector[:, None]
            + 1j * Xf[..., 1, :, :] * self.ft.ky_vector[None, :]
        )
        Y = self.ft.backward(Yf, dim=self.dim).real

        # return the dimension along which the divergence was computed
        if self.keep_dim:
            return Y.unsqueeze(-3)

        return Y
    
class FourierCurl3d(Step):
    """Calculates curl of a 3d signal in Fourier space.

    Implements the formula:

    .. math::

                         ┌─                            ─┐    
                         │ i k_y F[g_z] - i k_z F[g_y]  │    
                         │                              │
        F[∇ × g(x,y)] =  │ i k_z F[g_x] - i k_x F[g_z]  │
                         │                              │
                         │ i k_x F[g_y] - i k_y F[g_x]  │
                         └─                            ─┘
        
    where F is Fourier transform, k_x and k_y, k_z are spatial frequencies.
    Note the sign, which is due to the defintion of the Fourier transform
    adopted here in :class:`FourierTransform2d`. 
    
    This component can be used to construct a field J such that ∇·J = 0, by
    letting J = ∇ × g for a function g. 
    
    To constrain J to 2d, set the third component to zero by toggling `force_2d = True`.

    """

    def __init__(self, force_2d=False):
        super().__init__()
        self.real_signal = False
        self.force_2d = force_2d
        # dimensions at which divergence is computed
        self.dim = (-3, -2, -1)
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(real_signal={self.real_signal})"

    def fit(self, X, y=None, **fit_params):
        if len(X.shape) < 3:
            raise ValueError(
                "X must have number of dimensions (..., components, height, width) > 3, "
                "got {}".format(len(X.shape))
            )

        self.ft = FourierTransform2d(
            grid_shape=X.shape, dx=1, dy=1, real_signal=self.real_signal, type="linear"
        )
        return self

    def transform(self, X, y=None, **fit_params):
        Xf = self.ft.forward(X, dim=self.dim)
        Yf = (
            1j * Xf[..., 0, :, :] * self.ft.kx_vector[:, None]
            + 1j * Xf[..., 1, :, :] * self.ft.ky_vector[None, :]
        )
        Y = self.ft.backward(Yf, dim=self.dim).real

        # return the dimension along which the divergence was computed
        if self.keep_dim:
            return Y.unsqueeze(-3)

        return Y


class FourierZeroDivergenceConstraint2d(Step):
    """The component that enforces zero divergence in Fourier space. It takes as an input one
    component of the divergence-less field and computes the second component such that the
    divergence of the resultant field is everywhere zero.
    """

    def __init__(self, adjust_input=False, k00=0.0):
        super().__init__()
        # TODO: Implement real signal option, the issue occurs when X is of odd shape,
        # then ft.backward(ft.forward(X)) does not produce the same shape as X
        self.real_signal = False
        self.k00 = k00
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(real_signal={self.real_signal})"

    def fit(self, X, y=None, **fit_params):
        self.ft = FourierTransform2d(
            grid_shape=X.shape, dx=1, dy=1, real_signal=self.real_signal
        )

        # preprare the matrix M such that M * Xf = Yf (* is the bitwise multiplication operator)
        kx_vector = self.ft.kx_vector
        ky_vector = self.ft.ky_vector
        # set to 1 components of k_y to avoid division by zero
        ky_vector[0] = 1
        M = -kx_vector[:, None] / ky_vector[None, :]

        # all is good now except the [:, 0] components of Yf is not determined properly
        self.M = M
        return self

    def transform(self, X, y=None, **transform_params):
        Xf = self.ft.forward(X, dim=(-2, -1))
        M = self.M
        Yf = M * Xf
        # set k_y = 0 components to zero
        # TODO: Implement general boundary conditions to handle this:
        # Idea is to pass a function f(x) = f(x, y0) that sets a divergence-less field component at y0
        # Since k_y = 0, this function is the same for all y in the field component distribution.
        # Then calculate ft(f(x)) and set Yf[..., :, 0] = ft(f(x))
        Yf[..., :, 0] = 0.0
        Yf[..., 0, 0] = self.k00
        Y = self.ft.backward(Yf, dim=(-2, -1)).real
        return Y


class PlotResults(Step):
    def __init__(self, **plot_kwargs):
        self.plot_kwargs = plot_kwargs
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        # copy plot_kwargs to avoid changing the original dict
        plot_kwargs = self.plot_kwargs.copy()

        # update plot_kwargs with the params passed in transform_params by using
        # only existing keys in plot_kwargs
        for key in plot_kwargs.keys():
            if key in transform_params:
                plot_kwargs[key] = transform_params[key]

        p = plot_n_components(X, **self.plot_kwargs, show=False)
        display(p)
        return X


class CurrentLayerToField(Step):
    def __init__(self, dx, dy, height, layer_thickness):
        self.dx = dx
        self.dy = dy
        self.height = height
        self.layer_thickness = layer_thickness
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(dx={self.dx}, dy={self.dy}, height={self.height}, layer_thickness={self.layer_thickness})"

    def fit(self, X, y=None, **fit_params):
        # TODO: Possibly there needs to be a mandatory padding with zeros, so that the shape of the field is expnaded. This is to avoid
        # artifacts due to the periodicity of the Fourier methods. The result is exact only if the source itself is periodic with the
        # period of the source shape. Otherwise padding with zeros is enough, but only in the case when no currents pass the source boundary.
        source_shape = X.shape[-2:]
        self.propagator = CurrentPropagator2d(
            source_shape,
            self.dx,
            self.dy,
            self.height,
            self.layer_thickness,
            real_signal=False,
        )
        return self

    def transform(self, X, y=None, **transform_params):
        return self.propagator(X).real


class Projection(Step):
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi
        self.proj = AxisProjectionPropagator(theta=self.theta, phi=self.phi)
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(theta={self.theta}, phi={self.phi})"

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        return self.proj(X).unsqueeze(-3)


# TODO: Implement padding with up_to argument
# the current pad_n is more logically called mult_of
# TODO: torch.tensor_split seems very suitable for the task, it returns
# a view of the original tensor split along indices or in n parts.
class Padder(Step):
    def __init__(
        self, mult_of=None, up_to=None, position="center", bc=0, dims=(-2, -1)
    ):
        """Pads the input so that the shape of the output is larger according to `pad_n` argument by expanding the input
        with the `bc` boundary condition rules.

        Args:
            pad_n (int or tuple[int]): The scale of the padding.
                If int, then the padding is the same in all dimensions.
                If tuple, then the padding is different in each dimension.
                Follows the specification of torch.nn.functional.pad: output
                is expanded with `pad_n[i] * input.shape[i]` in each direction i:
                left, right, top, bottom

            bc (int, str, or callable, optional): The boundary condition.
                Defaults to 0. If boundary condition is a string, then it must be
                one of the following: 'reflect' or  'replicate'.

            dims (tuple[int], optional): The dimensions to pad. Defaults to (-2, -1).

            up_to (int or tuple[int, int]): The size of the output. If int, then the
                output is the same in each dimension. If tuple, then pads the input to the
                corresponding size in each dimension separately.

            position (str, optional): position that determines where the original input is placed in the output.
                - 'center' (default) - the original input is centered in the output, or is by 1 index
                shifted to the begginging of the dims if the input shape is odd.
                - 'corner' - the original input is placed in the corner of the output at index 0 along
                each dimension.
        """
        if isinstance(mult_of, str):
            raise NotImplementedError(
                "String specification of padding is not implemented yet."
            )
        elif isinstance(mult_of, int):
            # if int, then pad the same in all dimensions (* len(dims)) and in all directions (* 2)
            mult_of = (mult_of,) * len(dims) * 2

        if isinstance(up_to, int):
            # if int, then pad the same in all dimensions (* len(dims)) and in all directions (* 2)
            up_to = (up_to,) * len(dims) * 2

        if up_to is None and mult_of is None:
            raise ValueError("Either `mult_of` or `up_to` must be specified.")

        self.mult_of = mult_of
        self.up_to = up_to
        self.pad_func = torch.nn.functional.pad
        self.dims = dims
        pass

    def __repr__(self):
        if self.mult_of is not None:
            return f"{self.__class__.__name__}(mult_of={self.mult_of})"
        return f"{self.__class__.__name__}(pad_width={self.up_to})"

    def fit(self, X, y=None, **fit_params):
        self.padding = self.get_pad(
            shape=X.shape, up_to=self.up_to, mult_of=self.mult_of, dims=self.dims
        )

        self.original_shape = X.shape
        self.padded_shape, self.original_slices = Padder.get_padded_shape_and_slices(
            self.original_shape, pad=self.padding, dims=self.dims
        )

        self.X_ = torch.zeros(self.padded_shape, dtype=X.dtype, device=X.device)
        return self

    @staticmethod
    def get_pad(shape, up_to=None, mult_of=None, dims=(-2, -1), order="physical"):
        """Calculate the padding needed in `dims` to realize the strategy:
        either pad until the size is `pad_to` or until the size is a `mult_of` multiple
        of the original `shape`.

        Args:
            shape:   (tuple[int]) The shape of the array to be padded.
            up_to:   (int, tuple[int, int]) The size of the output. If int, then the
                     the size is the same in each dimension. If tuple, then pads the
                     input to the corresponding size in each dimension separately.
            mult_of: (int, tuple[int, int]) The size of the output. If int, then pads the input with that
                     multiple of size in each dimension. If tuple, then pads the input
                     with the corresponding multiple of size in each dimension separately.
            dims:    (tuple[int], optional) The dimensions to pad. Defaults to (-2, -1).
            order:   (str, "physical" or "torch") Order of the paddings in the tuple. "torch"
                     padding follows PyTorch convention where ordering begins from the last dimension,
                     physical ordering begins follows the ordering in the dims argument.

        """
        if up_to is not None:
            pad = tuple()
            for i, dim in enumerate(dims):
                # calculate how much padding on both sides is needed to get the desired shape
                a, reminder = divmod(up_to[i] - shape[dim], 2)
                pad += (
                    a,
                    a + reminder,
                )
        elif mult_of is not None:
            pad = tuple(
                # calculate how much padding on both sides is needed to get the desired shape
                shape[dims[torch.div(i, 2, rounding_mode="floor")]] * n
                for i, n in enumerate(mult_of)
            )
        return pad

    @staticmethod
    def get_slice_into_original(dims, original_slices):
        """Construct a slice into the unpadded portion of the array by putting the original slices
        into the corresponding positions specified in `dims`."""

        if all(d < 0 for d in dims):
            negative = True
        elif all(d >= 0 for d in dims):
            negative = False
        else:
            raise ValueError(
                "Dimensions specification must be consistent, i.e. all negative or all positive, "
                "to specify the index of dimensions either from the beginning, or from the end."
            )

        n_dims = max(dims) - min(dims) + 1  # expected number of dimensions
        # e.g. if dims = (-2, -1), then n_dims = 2, if dims = (-3, -1), then n_dims = 3,
        # meaning that there is at least 3 dimensions expected in the padded array

        sl = [slice(None)] * n_dims
        for i, dim in enumerate(dims):
            sl[dim] = original_slices[i]

        # handle not-padded dimensions at the beginning of the end of the arra
        if negative:
            sl = [...] + sl  # append ellipsis at the beginning
        else:
            sl = sl + [...]  # append ellipsis at the end

        return sl

    def transform(self, X, y=None, **params):
        """Pad X with prepared values."""
        # copy the array and detach it from the graph so that autograd does not propagate
        # through it twice on the .backward() call
        Y = self.X_.detach().clone()
        # use the slice to asssign the original array to the values inside the padded array
        sl = Padder.get_slice_into_original(
            dims=self.dims, original_slices=self.original_slices
        )
        Y[sl] = X
        return Y

    def inverse_transform(self, X, y=None, **params):
        """Unpad the tensor by taking the slice into original values at the
        dimensions where padding was performed.

        `X` can change size from the first invocation of self.transform, only
        `self.dims` dimensions are expected to match the originally padded
        dimensions. Thus we slice into `X` by taking all the same except the
        `original_slices` at self.dims.
        """
        sl = self.get_slice_into_original(
            dims=self.dims, original_slices=self.original_slices
        )
        return X[sl]

    def test_transform_inverse_transform():
        shape = (45, 66)
        p = Padder(up_to=shape, dims=(-2, -1))
        x = torch.rand(1, 1, 10, 22)
        p.fit(x)
        assert shape == p.transform(x).shape
        assert torch.all(x == p.inverse_transform(p.transform(x)))

    @staticmethod
    def get_padded_shape_and_slices(original_shape, pad, dims=(-2, -1)):
        """Returns the shape of the padded array, after the original shape is padded with `pad` in the dimensions `dims` such that each
        dimension is expanded by the pair of numbers in `pad` corresponding to that dimension.
        """
        if len(pad) != len(dims) * 2:
            raise ValueError(
                "The length of the `pad` argument must be twice that of the `dims` to specify all padding dimensions and directions."
            )

        original_slice = tuple()
        padded_shape = list(original_shape)

        # Iterate through the dimensions to pad and extract the amount padded and boundaries of the original array in the padded array
        for i, d in enumerate(dims):
            a = pad[2 * i]
            b = pad[2 * i + 1]
            sh = original_shape[d] + a + b
            sl = slice(a, -b)
            original_slice += (sl,)
            padded_shape[d] = sh

        return padded_shape, original_slice


class Sandwich(Pipe):
    def __init__(self, bread, steps, *, memory=None, verbose=False):
        self.bread = bread
        self.pipe = Pipe(steps=steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        self.bread.fit(X, y, **fit_params)
        self.pipe.fit(self.bread.transform(X), y, **fit_params)
        return self

    def transform(self, X, y=None, **transform_params):
        return self.bread.inverse_transform(
            self.pipe.transform(self.bread.transform(X), y, **transform_params)
        )


class GaussianFilter(Step):
    def __init__(
        self, sigma, order=0, mode="reflect", cval=0.0, truncate=4.0, radius=None
    ):
        super().__init__()

        import scipy

        self.gaussian_filter = scipy.ndimage.gaussian_filter1d

        self.sigma = sigma
        self.order = order
        self.mode = mode
        self.cval = cval
        self.truncate = truncate
        self.radius = radius

    def fit(self, X, y=None, **fit_params):
        # Overwrite the default values of the parameters with the ones provided in fit_params
        for key, value in fit_params.items():
            setattr(self, key, value)

        return self

    def transform(self, X, y=None, **transform_params):
        res = self.gaussian_filter(
            X,
            axis=-1,
            sigma=self.sigma,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            truncate=self.truncate,
        )
        res = self.gaussian_filter(
            res,
            axis=-2,
            sigma=self.sigma,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            truncate=self.truncate,
        )
        return torch.tensor(res, device=X.device)


class HannFilter(Step):
    def __init__(self, real_signal=True):
        super().__init__()
        self.real_signal = real_signal

    def fit(self, X, y=None, **fit_params):
        # Proper definition of the Hann filter does not depend on dx and dy, for it cancels out in the
        # filter defintion by multiplying the k_vector by dx. This is how it should be, and equivalent call
        # without the additional parameter would be to hardcode dx = 1, dy = 1, in the units of image pixels.
        # However it raises a question of what to do when image is padded or modified?
        self.ft = FourierTransform2d(
            grid_shape=X.shape, dx=1, dy=1, real_signal=self.real_signal
        )
        if self.real_signal:
            self.filter = (
                0.5
                * (1 + np.cos(self.ft.kx_vector / 2))[:, None]
                * 0.5
                * (1 + np.cos(self.ft.ky_vector / 4))[None, :]
            )
        else:
            self.filter = (
                0.5
                * (1 + np.cos(self.ft.kx_vector / 2))[:, None]
                * 0.5
                * (1 + np.cos(self.ft.ky_vector / 2))[None, :]
            )
        return self

    def transform(self, X, y=None):
        return self.ft.backward(
            self.ft.forward(X, dim=(-2, -1)) * self.filter, dim=(-2, -1)
        ).real

    def show_filter(self, centered_zero_frequency=False):
        """
        Plots the filter in Fourier space.

        Args:
            centered (bool):    whether to center zero frequency in the middle of the plot. By default,
                                the zero frequency is at the index [0, 0] of the filter. If centered, the
                                zero frequency is at the index [N//2, N//2] where N is the size of the filter.
                                The standard behavior is to have the zero frequency at the index [0, 0], positive
                                frequencies to be at [i, j] and negative frequencies to be at [-i, -j] where
                                i, j ∈ [1, N//2].
        """
        import matplotlib.pyplot as plt

        from magrec.misc.plot import plot_n_components

        filter = self.filter
        kx_vector = self.ft.kx_vector
        ky_vector = self.ft.ky_vector

        if centered_zero_frequency:
            if self.real_signal:
                filter = np.fft.fftshift(filter, axes=(-2,))
            else:
                filter = np.fft.fftshift(filter, axes=(-2, -1))
                ky_vector = np.fft.fftshift(ky_vector)

            kx_vector = np.fft.fftshift(kx_vector)

        fig = plot_n_components(
            filter, show_coordinate_system=False, climits=(0, 1), labels="no_labels"
        )
        ax: plt.Axes = fig.axes[0]
        ax.set_xlabel(r"$k_x$ (radians per unit length)")
        ax.set_ylabel(r"$k_y$ (radians per unit length)")
        ax.set_title("Hann filter")

        plt.show(fig)
        plt.close()
        xticks_locs = ax.get_xticks()
        yticks_locs = ax.get_yticks()

        # set tick labels to correspond to the kx, ky values
        ax.set_xticklabels(["{:.2f}".format(l.item()) for l in kx_vector[xticks_locs]])
        ax.set_yticklabels(["{:.2f}".format(l.item()) for l in ky_vector[yticks_locs]])

        return fig


# how we want to use it:
# J = recon.propagate(B)


def params_for(prefix, kwargs):
    """Extract parameters that belong to a given sklearn module prefix from
    ``kwargs``. This is useful to obtain parameters that belong to a
    submodule.

    Examples
    --------
    >>> kwargs = {'encoder__a': 3, 'encoder__b': 4, 'decoder__a': 5}
    >>> params_for('encoder', kwargs)
    {'a': 3, 'b': 4}

    """
    if not prefix.endswith("__"):
        prefix += "__"
    return {
        key[len(prefix) :]: val for key, val in kwargs.items() if key.startswith(prefix)
    }


# from sklearn.base
def get_param_names(cls):
    """Get parameter names for the estimator"""
    # fetch the constructor or the original constructor before
    # deprecation wrapping if any
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    if init is object.__init__:
        # No explicit constructor to introspect
        return []

    # introspect the constructor arguments to find the model parameters
    # to represent
    init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    parameters = [
        p
        for p in init_signature.parameters.values()
        if p.name != "self" and p.kind != p.VAR_KEYWORD
    ]
    for p in parameters:
        if p.kind == p.VAR_POSITIONAL:
            raise RuntimeError(
                "scikit-learn estimators should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s with constructor %s doesn't "
                " follow this convention." % (cls, init_signature)
            )
    # Extract and sort argument names excluding 'self'
    return sorted([p.name for p in parameters])


def replace_keys_with_aliases(d: dict, aliases: dict) -> dict:
    """Replaces keys in a dictionary `d` with aliases in `aliases` if they exist."""
    result = d.__class__()
    for key, value in d.items():
        long_key = aliases.get(key, key)  # get long key by key, else just `key`
        result[long_key] = value

    return result


class PipelineEstimator(BaseEstimator):
    def __init__(
        self,
        net: torch.nn.Module,
        model: Pipe,
        pipe: Pipe,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        max_epochs=10,
        optimizer=torch.optim.SGD,
        criterion=torch.nn.MSELoss,
        callbacks=None,
        verbose=1,
    ):
        self.net: None | torch.nn.Module = net
        self.n_channels_in: int = n_channels_in
        self.n_channels_out: int = n_channels_out

        self.initialized: bool = False
        self.max_epochs: int = max_epochs
        self.optimizer: torch.optim.optimizer.Optimizer = optimizer
        self.lr = 0.01
        self.criterion: torch.nn.modules.loss._Loss = criterion

        self.verbose = verbose

        self.model: Pipe = model
        self.pipe: Pipe = pipe

        self.default_callbacks = OrderedDict(
            [
                ("epoch_timer", EpochTimer()),
                (
                    "train_loss",
                    PassthroughScoring(
                        name="train_loss",
                        on_train=True,
                    ),
                ),
                (
                    "valid_loss",
                    PassthroughScoring(
                        name="valid_loss",
                    ),
                ),
                ("print_log", PrintLogEvery(every=10)),
            ]
        )

        self.callbacks = callbacks
        self.history = History()

        self.param_name_aliases = {
            # "alias": "param name"
            "lr": "optimizer__lr",
            "print_log_every": "callbacks__print_log__every",
            "n_channels_in": "net__n_channels_in",
            "n_channels_out": "net__n_channels_out",
        }

    def propagate(self, B, epochs=None, **params):
        """Propagate X along the reconstruction pipeline.

        For example, if X is a magnetic field B, then the current density J is

        J = self.propagate(B)
        """

        if params:
            params = replace_keys_with_aliases(params, self.param_name_aliases)
            self.initialize(**params)

        epochs = epochs if epochs is not None else self.max_epochs
        optimizer = self.optimizer

        B = self.pipe.propagate(B, visual=False)
        B_target = B

        self.net.train(True)

        # TODO: For today,
        # 1. check ZeroDivergenceConstraint2d if there is a problem in math
        # 2. check if Fourier method k = 0 is handled properly
        # 3. add an additional loss function that enforces continuity of the current density

        try:
            self.notify("on_train_begin")
            for _ in range(epochs):
                # start history for the current epoch
                self.history.new_epoch()
                self.notify("on_epoch_begin")

                self.history.record("epoch", len(self.history))
                self.history.new_batch()
                self.notify("on_batch_begin")

                optimizer.zero_grad()
                J = self.net(B)
                B_pred = self.model.propagate(J, visual=False)
                # B_pred = narrow_to_target(B_pred)

                field_loss = self.criterion(B_pred, B_target)
                # div_J = FourierDivergence2d()(J)
                # div_loss = self.criterion(div_J * 1e2, torch.zeros_like(div_J))

                loss = field_loss

                loss.backward()
                optimizer.step()

                step = {
                    "loss": loss,
                    "field_loss": field_loss,
                    "y_pred": B_pred,
                    "training": True,
                }

                self.history.record_batch("train_loss", step["loss"].item())
                # self.history.record("field_loss", step["field_loss"].item())
                self.history.record_batch("train_batch_size", 1)
                self.notify("on_batch_end", **step)
                self.notify("on_epoch_end")
        except KeyboardInterrupt:
            pass

        self.net.train(False)
        # obtain J again in non-train mode, useful when epochs == 0
        # when you'd like to obtain the reconstructed J
        J = self.net(B)
        self.notify("on_train_end")

        res = self.model.propagate(J, visual=True)
        return res

    # TODO: Make consistent usage of unrolled params. It is not clear if a method
    # takes a dics of params or just a big list of params. The idea is to have
    # a list of parameters accepted only in function that are exposed to the user.
    def initialize(self, **params):
        self.net = self.initialize_component("net", **params)
        self.optimizer = self.initialize_component("optimizer", **params)
        self.criterion = self.initialize_component("criterion", **params)

        # callbacks need special treatment because they are OrderedDict
        self.callbacks = self.initialize_callbacks(**params)

        self.initialized = True
        return self

    # a quick fix to make skorch happy
    @property
    def optimizer_(self):
        return self.optimizer

    def notify(self, method_name, **cb_kwargs):
        """Call the callback method specified in `method_name` with
        parameters specified in `cb_kwargs`.

        Method names can be one of:
        * on_train_begin
        * on_train_end
        * on_epoch_begin
        * on_epoch_end
        * on_batch_begin
        * on_batch_end

        """
        for _, cb in self.callbacks.items():
            getattr(cb, method_name)(self, **cb_kwargs)

    def initialize_component(self, component_name, **params):
        """Initialize the component with the given parameters.

        Parameters
        ----------
        **params
          Parameters passed to the component.

        Returns
        -------
        self

        """
        component = getattr(self, component_name, None)

        if component is None:
            raise ValueError(
                "Cannot initialize {} when it is None.".format(component_name)
            )

        # normalize parameters with their aliases
        params = self.get_params_for(
            component_name,
            replace_keys_with_aliases(self.__dict__, self.param_name_aliases)
            | replace_keys_with_aliases(params, self.param_name_aliases),
        )

        if isinstance(component, type):  # component is a class, initialize it
            # handle specifal cases such as that optimizer needs params
            # on initialization
            if component_name == "optimizer":
                # update default parameters with passed params
                params = {
                    "params": self.net.parameters(),  # optimizer params
                    "lr": self.lr,
                } | params

            setattr(self, component_name, component(**params))
            return getattr(self, component_name)
        else:  # component is already initialized, update its parameters
            self.set_params(component, **params)

        return component

    # TODO: A lot of code below can be reused for pipe, model, etc.
    # where unique names are used.
    def initialize_callbacks(self, **params):
        """Initialize callbacks and update their parameters.

        This method is called at the beginning of :meth:`.propagate` and in
        :meth:`.initialize`.

        """
        callbacks_grouped_by_name = OrderedDict()
        callbacks_ = OrderedDict()
        print_logs = OrderedDict()
        names_set_by_user = set()

        # adapted from skorch.NeuralNet.initialize_callbacks
        class Dummy:
            # We cannot use None as dummy value since None is a
            # legitimate value to be set.
            pass

        # check if callbacks have name or assign one
        if self.callbacks is None:
            items = self.default_callbacks.items()
        elif isinstance(self.callbacks, OrderedDict):
            # merge existing callbacks with default callbacks
            items = (self.default_callbacks | self.callbacks).items()
        elif isinstance(self.callbacks, list):
            items = list(self.default_callbacks.items()) + self.callbacks
        elif isinstance(self.callbacks, tuple):
            items = tuple(self.default_callbacks.items()) + self.callbacks
        else:
            raise ValueError(
                "callbacks must be an instance of "
                "OrderedDict, list, or tuple, but got {}".format(type(self.callbacks))
            )
        for item in items:
            if isinstance(item, (tuple, list)):
                name, cb = item
                names_set_by_user.add(name)
            else:
                cb = item
                if isinstance(cb, type):  # uninitialized:
                    name = cb.__name__
                else:
                    name = cb.__class__.__name__
            # add print logs to a set which will be added as a last group in the ordered dict
            if isinstance(cb, PrintLog) or (cb == PrintLog):
                print_logs[name] = print_logs.get(name, []) + [cb]
                continue

            # add to a dict of callbacks grouped by name
            callbacks_grouped_by_name[name] = callbacks_grouped_by_name.get(
                name, []
            ) + [cb]

        # add print logs to the end of the ordered dict
        callbacks_grouped_by_name.update(print_logs)

        # make names in callbacks dict unique
        for name, cbs in callbacks_grouped_by_name.items():
            if len(cbs) > 1 and name in names_set_by_user:
                raise ValueError(
                    "Found duplicate user-set callback name "
                    "'{}'. Use unique names to correct this.".format(name)
                )

            for i, cb in enumerate(cbs):
                if len(cbs) > 1:
                    unique_name = "{}_{}".format(name, i + 1)
                    if unique_name in callbacks_:
                        raise ValueError(
                            "Assigning new callback name failed "
                            "since new name '{}' exists already.".format(unique_name)
                        )
                else:
                    unique_name = name
                # add to a dict of callbacks with unique names
                callbacks_[unique_name] = cb

        # prepare a list of initialized callbacks with parameters
        callbacks = OrderedDict()

        # check if callback itself is changed
        for name, cb in callbacks_.items():
            param_callback = getattr(self, "callbacks__{}".format(name), Dummy)
            if param_callback is not Dummy:  # callback itself was set
                cb = param_callback

            # below: check for callback params
            # don't set a parameter for non-existing callback
            cb_params = self.get_params_for("callbacks__{}".format(name), params)
            if (cb is None) and params:
                raise ValueError(
                    "Trying to set a parameter for callback {} "
                    "which does not exist.".format(name)
                )
            if cb is None:
                continue

            if isinstance(cb, type):  # uninitialized:
                cb = cb(**cb_params)
            else:
                cb.set_params(**cb_params)

            cb.initialize()
            callbacks.update({name: cb})

        return callbacks

    def get_params_for(self, prefix, params):
        """Collect and return init parameters for an attribute (`prefix`).

        Attributes could be, for instance, pytorch modules, criteria,
        or data loaders. Use the returned arguments to initialize the given
        attribute like this:

        .. code:: python

            # inside initialize_module method
            kwargs = self.get_params_for('module')
            self.module_ = self.module(**kwargs)

        Proceed analogously for the criterion etc.

        The reason to use this method is so that it's possible to
        change the init parameters with :meth:`.set_params`, which
        in turn makes grid search and other similar things work.

        Note that in general, as a user, you never have to deal with
        this method because :meth:`.initialize_module` etc. are
        already taking care of this. You only need to deal with this
        if you override :meth:`.initialize_module` (or similar
        methods) because you have some custom code that requires it.

        Parameters
        ----------
        prefix : str
          The name of the attribute whose arguments should be
          returned. E.g. for the module, it should be ``'module'``.

        Returns
        -------
        kwargs : dict
          Keyword arguments to be used as init parameters.

        """
        return params_for(prefix, params)

    # NOTE: Can be moved to a base class for all steps in pipe
    def set_params(self, component, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        component : str, component instance, or 'all'
        **params  : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        if isinstance(component, OrderedDict):
            valid_params = component.keys()
        elif isinstance(component, torch.optim.Optimizer):
            valid_params = component.defaults.keys()
        else:
            valid_params = self.get_params(component, deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            # check if key is valid, ie if component has a parameter with this name
            if key not in valid_params:
                local_valid_params = list(valid_params.keys())
                raise ValueError(
                    f"Invalid parameter {key!r} for {component}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )
            # if parameters is nested like component__param,
            if delim:
                # add to nested_params
                nested_params[key][sub_key] = value
            else:
                if isinstance(component, OrderedDict):
                    component[key] = value
                elif isinstance(component, torch.optim.Optimizer):
                    # TODO: Optimizer parameters are harder than they look.
                    # See line 1774 of skorch.net module, there they handle
                    # optimizer_params with patterns.
                    # NOTE: Works assuming there's only one group
                    if len(component.param_groups) > 1:
                        raise RuntimeError(
                            "Optimizer with more than one group is not supported"
                        )
                    component.param_groups[0][key] = value
                else:
                    setattr(component, key, value)

        for key, sub_params in nested_params.items():
            if isinstance(component, OrderedDict):
                sub_component = component[key]
            else:
                sub_component = getattr(component, key)

            self.set_params(sub_component, **sub_params)

        return self

    def get_params(self, component, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in get_param_names(component):
            try:
                value = getattr(component, key)
            except AttributeError:
                continue

            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def plot_loss(self, epochs: int | slice=None):
        fig = plt.figure(clear=True)
        ax = fig.subplots()
        
        if isinstance(epochs, int):
            epochs = slice(epochs)
        elif isinstance(epochs, tuple):
            if len(epochs) == 2:
                epochs = slice(min(epochs), max(epochs))
        elif epochs is None:
            epochs = slice(None)

        # Get loss statistics from self.history

        # First determine what loss is tracked
        # Assume all loss types are indicated at step 0
        loss_keys = []  # list of loss keys to extract
        for key in self.history[-2].keys():  # -2 is a dirty hack b/c I haven't 
            if key.endswith("_loss"):        # figured out structure of history yet
                loss_keys.append(key)        # -1 can be empty sometimes, 0 as well
                                             # -2 should work if number of epochs > 2
        for loss_key in loss_keys:
            loss = self.history[:, loss_key][epochs]
            ax.plot(loss, label=loss_key)

        # Set y-axis of the ax to be log scale
        ax.set_yscale("log", base=10)

        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best")
        ax.set_title("Error function evolution")


class HistoryRecorder(Callback):
    # Note: history recorder can be a callback when the issue with the order is solved,
    # currently net.history needs to be updated before any of the callbacks are called.
    # That would require this callback to be strictly first in the list of callbacks,
    # but this is painful to implement. A better way, I think, is to keep the history as
    # a list of steps and then have a separate callback that would be responsible for
    # adding these steps into the history.

    def on_batch_begin(self, net, batch=None, training=None, **kwargs):
        net.history.new_batch()
        pass

    def on_epoch_begin(self, net, **kwargs):
        net.history.new_epoch()
        self.history.record("epoch", len(net.history))
        pass

    def on_batch_end(self, net, **kwargs):
        net.history.record_batch()
        pass

    def on_epoch_end(self, net, **kwargs):
        net.history.record_epoch()
        pass


class PrintLogEvery(PrintLog):
    """Prints the log every `n` epochs."""

    def __init__(
        self,
        every=1,
        keys_ignored=None,
        sink=print,
        tablefmt="simple",
        floatfmt=".4f",
        stralign="right",
    ):
        self.every = every
        super().__init__(keys_ignored, sink, tablefmt, floatfmt, stralign)

    def on_epoch_end(self, net, **kwargs):
        if net.history[-1, "epoch"] % self.every == 0:
            super().on_epoch_end(net, **kwargs)
            pass
        elif self.first_iteration_:
            super().on_epoch_end(net, **kwargs)
        else:
            pass

    def on_train_end(self, net, **kwargs):
        super().on_train_end(net, **kwargs)
        self.first_iteration_ = True
        pass


class AdjustLROnLoss(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, loss_vals, lr_vals, last_epoch=-1, verbose=False, **kwargs
    ):
        if len(loss_vals) + 1 != len(lr_vals):
            raise ValueError(
                "Number of loss value points + 1 and learning rates must be the same"
            )
        self.loss_vals = loss_vals
        self.lr_vals = lr_vals
        self.optimizer = optimizer

        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def step(self, epoch, metrics):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        # convert `metrics` to float, in case it's a zero-dim Tensor
        metrics = float(metrics)
        for bound, lr in zip(self.loss_vals, self.lr_vals[:-1]):
            if metrics > bound:
                values = [lr] * len(self.optimizer.param_groups)
                break

        # if we didn't find a bound, use the last lr
        if metrics <= self.loss_vals[-1]:
            values = [lr] * len(self.optimizer.param_groups)

        self.set_lr(epoch, values)

    def set_lr(self, epoch, values):
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    "Epoch {}: reducing learning rate"
                    " of group {} to {:.4e}.".format(epoch_str, i, lr)
                )


class LRScheduler(skorch.callbacks.LRScheduler):
    def on_epoch_end(self, net, **kwargs):
        """Overwrite `on_epoch_end` to support AdjustLROnLoss scheduler."""
        if self.step_every != "epoch":
            return
        if isinstance(self.lr_scheduler_, AdjustLROnLoss):
            score = net.history[:, "train_loss"][-1]
            epoch = net.history[-1, "epoch"]
            self.lr_scheduler_.step(epoch, score)

            if self.event_name is not None and hasattr(
                self.lr_scheduler_, "get_last_lr"
            ):
                net.history.record(self.event_name, self.lr_scheduler_.get_last_lr()[0])

        # ReduceLROnPlateau does not expose the current lr so it can't be recorded
        elif isinstance(
            self.lr_scheduler_, skorch.callbacks.lr_scheduler.ReduceLROnPlateau
        ):
            if callable(self.monitor):
                score = self.monitor(net)
            else:
                try:
                    score = net.history[-1, self.monitor]
                except KeyError as e:
                    raise ValueError(
                        f"'{self.monitor}' was not found in history. A "
                        f"Scoring callback with name='{self.monitor}' "
                        "should be placed before the LRScheduler callback"
                    ) from e

            self.lr_scheduler_.step(score)
        else:
            if self.event_name is not None and hasattr(
                self.lr_scheduler_, "get_last_lr"
            ):
                net.history.record(self.event_name, self.lr_scheduler_.get_last_lr()[0])
            self.lr_scheduler_.step()


class PipelineEstimatorWithDivergence(PipelineEstimator):
    def propagate(self, B, epochs=None, **params):
        """Propagate X along the reconstruction pipeline.

        For example, if X is a magnetic field B, then the current density J is

        J = self.propagate(B)
        """

        if params:
            params = replace_keys_with_aliases(params, self.param_name_aliases)
            self.initialize(**params)

        epochs = epochs if epochs is not None else self.max_epochs
        optimizer = self.optimizer

        B = self.pipe.propagate(B, visual=False)
        B_target = B

        self.net.train(True)

        div = FourierDivergence2d()
        alpha = 1  # weight for the divergence loss
        beta = 1   # weight for the field loss
        threshold = 1e-2  # threshold below which divergence is taken into account

        try:
            self.notify("on_train_begin")
            for _ in range(epochs):
                # start history for the current epoch
                self.history.new_epoch()
                self.notify("on_epoch_begin")

                self.history.record("epoch", len(self.history))
                self.history.new_batch()
                self.notify("on_batch_begin")

                optimizer.zero_grad()
                J = self.net(B)
                B_pred = self.model.propagate(J, visual=False)

                field_loss = beta * self.criterion(B_pred, B_target)
                div_J = div(J)
                div_J_target = torch.zeros_like(div_J)
                div_J_target[20:-20, 0] = 100
                div_J_target[20:-20, -1] = -100
                div_loss = alpha * self.criterion(div_J, torch.zeros_like(div_J))

                if field_loss < threshold:
                    loss = field_loss + div_loss
                else:
                    loss = field_loss

                loss.backward()
                optimizer.step()

                step = {
                    "loss": loss,
                    "field_loss": field_loss,
                    "div_loss": div_loss,
                    "y_pred": B_pred,
                    "training": True,
                }
                
                if loss is torch.nan:
                    raise RuntimeWarning("Loss is NaN at epoch {}".format(len(self.history)))
                    break

                self.history.record_batch("train_loss", step["loss"].item())
                self.history.record("field_loss", step["field_loss"].item())
                self.history.record("div_loss", step["div_loss"].item())
                self.history.record_batch("train_batch_size", 1)
                self.notify("on_batch_end", **step)
                self.notify("on_epoch_end")
        except KeyboardInterrupt:
            pass

        self.net.train(False)
        # obtain J again in non-train mode, useful when epochs == 0
        # when you'd like to obtain the reconstructed J
        J = self.net(B)
        self.notify("on_train_end")

        res = self.model.propagate(J, visual=True)
        return res