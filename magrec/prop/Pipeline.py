from collections import OrderedDict
import torch

# anticipate the change of the API in scikit-learn which moves Pipeline and FeatureUnion to sklearn.compose
from sklearn.utils import Bunch
import skorch

from sklearn.pipeline import Pipeline

import numpy as np

import fnmatch
from functools import partial


from magrec.prop.Propagator import AxisProjectionPropagator, CurrentPropagator2d
from magrec.prop.Fourier import FourierTransform2d
from magrec.misc.plot import plot_n_components

# the way it is now is a hard dependence on IPython, which prevents the use of magrec in a non-interactive environment
# TODO: make this optional and make `display` for flexible
from IPython.display import display


def identity(X):
    """The identity function."""
    return X


# ReviewTS
# You could use Python's ABC library here to declare Step as an abstract base class. Then users implementing their own
# steps would get immediate feedback when they forget to implement a required method. See https://docs.python.org/3/library/abc.html
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


# ReviewTS
# From my understanding, the functionality of `sklearn.Pipeline` is is very much suited for your usecase, so I like that you are using it as
# inspiration. However, I have a hard time understanding the benefit of this bespoke Pipe implementation compared to using `sklearn.Pipeline`
# directly (which you already import at the top of this file, but I guess decided against using later?).
#
# The way I see it, the main difference of this implementation to sklearn's variant is that you have special handling for visualization steps.
# As discussed earlier, I see a few potential issues with this design:
# * Semantically, you have different instances of the same entity (a step) do different things. Most steps are transformators for data,
#   but those that are called Plot* are not transforming anything but just creating plots. I think this makes it harder for users to reason
#   about the usage contract of this library. It would be better if this code would use the same semantics for a Step as sklean itself.
# * There is no easy way to extract plots. Instead, they are always immediately displayed. Ideally, it should be just as easy to
#   create and save a plot without displaying it in a notebook
# * The special handling is tied to the classname of the specific step, which will confuse users. Controlling logic via the name of some
#   element is not a common pattern and would need to be explicitly documented somewhere.
#
# Also, you manually reimplement a partial variant of `sklearn.Pipeline` here, which bears the risk of introducing subtle bugs, if one of the copied methods
# required some missing functionality of the original implementation in some cases. This implementation also misses some convencience
# features of sklearn's original variant.
#
# As an alternative, I would suggest:
# * Subclass sklearn.Pipeline to benefit from it's more mature implementation. Only override/add the functionalities that are not
#   provided by sklearn
# * Instead of adding plotting as steps, have a different interface for specifying which steps to plot during transformation. This could also be used for
#   other side-effects to be performed during transformation. Somehing like
#
#   ```
#   class Pipe(sklearn.Pipeline):
#       def add_transform_side_effect(step_name: str, effect: Callable)
#   ```
#
#   and then add a hook inside the transform() method where all added side effects are executed after the respective step. A side-effect
#   could then be everything from plotting data directly to saving it to disk or even triggering a different execution thread.
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
        # ReviewTS
        # This function is not really required I think (the user can just create a new Pipe object with the steps they need instead
        # of modifying an existing one). Therefore, I would suggest simply removing it to reduce the code's surface area.

        # ReviewTS
        # If `after` and `before` always have to be None, you could just remove these parameters alltogether to clean up the
        # user interface.
        if after is not None or before is not None:
            raise NotImplementedError("Step position specification is not implemented yet. "
                                      "Use `add_step` method consequentially in order steps need "
                                      "to be executed.")
        self.steps.update((step_name, step))
        # ReviewTS
        # If we were to subclass `sklearn.Pipeline`, we could call `self._validate_steps()` here to make sure that the
        # newly added step implements the required methods
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
        # ReviewTS
        # `sklearn`'s implementation checks if the fit_params are valid beforehand, which reduces the probability of a
        # user encountering weird errors down the line they might not know how to fix easily
        for idx, step_name, step in self._iter(filter=("passthrough", "plot*")):
            params = self.get_step_params(step_name, **fit_params)
            X = step.fit(X, y, **params).transform(X, visual=False)

        self.fitted = True
        return self

    # ReviewTS
    # I suggest having this functionality in an explicitly named method instead to make it immediately clear to users
    # what happens if they call it.
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
        # ReviewTS:
        # Like mentioned above, I would argue for reworking the way visualization is done in this pipeline.
        # As an aside, the `visual` parameter applies to the transform() function of the whole Pipeline (if it is set to false, disable
        # _all_ plot* steps). Therefore, it should rather be a named parameter to this function instead of a part of `transform_params` (which
        # are the parameters forwarded to the step transform() methods)
        visual = transform_params.get("visual", True)
        if not visual:
            filters += ("plot*",)

        for idx, step_name, step in self._iter(filter=filters):
            X = step.transform(X, y, **transform_params)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # ReviewTS
        # Different to the tranform() method, here the plot* steps are always disabled, which is a little weird. As a user, I would
        # expect the interface of `transform` and `fit_transform` to be basically identical.
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


# ReviewTS
# As mentioned for `Pipe`, I think it would be greatly advantageous if you could leverage the existing `sklearn.FeatureUnion`
# implementation, either by subclassing or having it as a member of this class, for most of the same reasons as above.
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
        # ReviewTS
        # The filter argument should be "plot*" to catch all classes starting with Plot*. This is a good example of how
        # easy it is to introduce subtle bugs when relying on string matching for code branching.
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
        from magrec.misc.plot import plot_n_components
        import matplotlib.pyplot as plt

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
