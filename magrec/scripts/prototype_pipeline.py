from collections import OrderedDict, defaultdict
from functools import wraps
import inspect
import json
import warnings
import weakref


import skorch
from skorch import History
from skorch.utils import open_file_like
from skorch.callbacks import PrintLog, PassthroughScoring, EpochTimer, Callback

from sklearn.base import BaseEstimator

import torch
import numpy as np
from magrec.prop.Pipeline import (
    CurrentLayerToField,
    FourierZeroDivergenceConstraint2d,
    FourierDivergence2d,
    Function,
    Pipe,
    Projection,
    Union,
)


# write a highly specific code for our specific task, then generalize it

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


class Prototype(BaseEstimator):
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
