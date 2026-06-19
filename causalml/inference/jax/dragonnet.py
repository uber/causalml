"""
JAX/flax.nnx implementation of DragonNet [1].

DragonNet adapts neural network design and training to improve treatment effect
estimation via propensity-score sufficiency and targeted regularization.

References:
    [1] C. Shi, D. Blei, V. Veitch (2019).
        Adapting Neural Networks for the Estimation of Treatment Effects.
        https://arxiv.org/pdf/1906.02120.pdf
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.model_selection import train_test_split

from causalml.inference.jax.utils import (
    dragonnet_loss_binarycross,
    make_tarreg_loss,
)
from causalml.inference.meta.utils import convert_pd_to_np


class EpsilonLayer(nnx.Module):
    """Learned scalar epsilon broadcast to batch shape.

    A single trainable parameter that is broadcast to match the batch
    dimension, enabling targeted regularization in DragonNet.
    """

    def __init__(self, rngs: nnx.Rngs):
        """Initializes EpsilonLayer with a random scalar parameter.

        Args:
            rngs: flax.nnx RNG container.
        """
        self.epsilon = nnx.Param(jax.random.normal(rngs.params(), shape=(1, 1)))

    def __call__(self, inputs):
        """Broadcasts epsilon to (batch, 1).

        Args:
            inputs: Array of shape (n, *) used only to determine batch size.

        Returns:
            Array of shape (n, 1) filled with the learned epsilon value.
        """
        return self.epsilon[...] * jnp.ones_like(inputs)[:, 0:1]


class DragonNetModule(nnx.Module):
    """flax.nnx DragonNet architecture.

    Three shared representation layers, a propensity head, two outcome heads
    (y0 and y1), and a learned epsilon scalar. Output is the concatenation
    [y0, y1, propensity, epsilon] with shape (n, 4).
    """

    def __init__(
        self, input_dim: int, neurons_per_layer: int, reg_l2: float, rngs: nnx.Rngs
    ):
        """Initializes all layers of DragonNet.

        Args:
            input_dim: Number of input features.
            neurons_per_layer: Width of shared representation layers.
            reg_l2: L2 regularization coefficient applied to outcome head kernels.
            rngs: flax.nnx RNG container.
        """
        half = neurons_per_layer // 2

        # Shared representation
        self.repr1 = nnx.Linear(input_dim, neurons_per_layer, rngs=rngs)
        self.repr2 = nnx.Linear(neurons_per_layer, neurons_per_layer, rngs=rngs)
        self.repr3 = nnx.Linear(neurons_per_layer, neurons_per_layer, rngs=rngs)

        # Propensity head
        self.t_head = nnx.Linear(neurons_per_layer, 1, rngs=rngs)

        # Outcome head y0
        self.y0_h1 = nnx.Linear(neurons_per_layer, half, rngs=rngs)
        self.y0_h2 = nnx.Linear(half, half, rngs=rngs)
        self.y0_out = nnx.Linear(half, 1, rngs=rngs)

        # Outcome head y1
        self.y1_h1 = nnx.Linear(neurons_per_layer, half, rngs=rngs)
        self.y1_h2 = nnx.Linear(half, half, rngs=rngs)
        self.y1_out = nnx.Linear(half, 1, rngs=rngs)

        # Epsilon
        self.epsilon_layer = EpsilonLayer(rngs=rngs)

        self.reg_l2 = reg_l2

    def __call__(self, x):
        """Forward pass returning (y0, y1, propensity, epsilon) concatenated.

        Args:
            x: Input array of shape (n, input_dim).

        Returns:
            Array of shape (n, 4): [y0, y1, propensity, epsilon].
        """
        # Shared representation
        z = jax.nn.elu(self.repr1(x))
        z = jax.nn.elu(self.repr2(z))
        z = jax.nn.elu(self.repr3(z))

        # Propensity
        t_pred = jax.nn.sigmoid(self.t_head(z))

        # Outcome y0
        y0 = jax.nn.elu(self.y0_h1(z))
        y0 = jax.nn.elu(self.y0_h2(y0))
        y0 = self.y0_out(y0)

        # Outcome y1
        y1 = jax.nn.elu(self.y1_h1(z))
        y1 = jax.nn.elu(self.y1_h2(y1))
        y1 = self.y1_out(y1)

        eps = self.epsilon_layer(t_pred)

        return jnp.concatenate([y0, y1, t_pred, eps], axis=1)

    def l2_penalty(self):
        """Computes L2 regularization over outcome-head kernels.

        Returns:
            Scalar L2 penalty value.
        """
        kernels = [
            self.y0_h1.kernel[...],
            self.y0_h2.kernel[...],
            self.y0_out.kernel[...],
            self.y1_h1.kernel[...],
            self.y1_h2.kernel[...],
            self.y1_out.kernel[...],
        ]
        return self.reg_l2 * sum(jnp.sum(k**2) for k in kernels)


def _reduce_lr_on_plateau(lr, factor, no_improve_count, patience):
    """Returns updated lr if plateau patience is exceeded.

    Args:
        lr: Current learning rate.
        factor: Multiplicative reduction factor.
        no_improve_count: Number of epochs without improvement.
        patience: Number of epochs to wait before reducing.

    Returns:
        Updated learning rate.
    """
    if no_improve_count > 0 and no_improve_count % patience == 0:
        return lr * factor
    return lr


def _make_train_step(loss_fn):
    """Returns a jit-compiled train step for a given loss function.

    Args:
        loss_fn: Callable(concat_true, concat_pred) -> scalar.

    Returns:
        Compiled train step function.
    """

    @nnx.jit
    def train_step(model, optimizer, X_batch, y_batch):
        def _loss(model):
            preds = model(X_batch)
            return loss_fn(y_batch, preds) + model.l2_penalty()

        loss_val, grads = nnx.value_and_grad(_loss)(model)
        optimizer.update(model, grads)
        return loss_val

    return train_step


def _compute_val_loss(model, loss_fn, X_val, y_val, batch_size):
    """Evaluates loss over the validation set in mini-batches.

    Args:
        model: DragonNetModule instance.
        loss_fn: Loss function.
        X_val: Validation features array.
        y_val: Validation targets array.
        batch_size: Mini-batch size.

    Returns:
        Mean validation loss across all batches.
    """
    n = X_val.shape[0]
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        xb = jnp.array(X_val[start : start + batch_size])
        yb = jnp.array(y_val[start : start + batch_size])
        preds = model(xb)
        total_loss += float(loss_fn(yb, preds) + model.l2_penalty())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _run_training_loop(
    model,
    optimizer,
    train_step_fn,
    loss_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    early_stop_patience,
    reduce_lr_patience,
    reduce_lr_factor,
    verbose,
    phase_name,
):
    """Generic training loop with early stopping and LR reduction on plateau.

    Args:
        model: DragonNetModule instance (mutated in-place).
        optimizer: optax optimizer wrapped with nnx.Optimizer.
        train_step_fn: Compiled train step function.
        loss_fn: Loss function for validation evaluation.
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        early_stop_patience: Stop if val_loss does not improve for this many epochs.
        reduce_lr_patience: Reduce LR after this many epochs without improvement.
        reduce_lr_factor: Multiplicative LR reduction factor.
        verbose: Whether to print epoch summaries.
        phase_name: Label printed in verbose output.

    Returns:
        None; model and optimizer are updated in-place.
    """
    best_val_loss = float("inf")
    no_improve = 0
    n_train = X_train.shape[0]
    rng = np.random.default_rng(0)
    current_lr = (
        optimizer.opt_state[0].hyperparams["learning_rate"]
        if hasattr(optimizer.opt_state[0], "hyperparams")
        else None
    )

    for epoch in range(epochs):
        idx = rng.permutation(n_train)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        for start in range(0, n_train, batch_size):
            xb = jnp.array(X_shuf[start : start + batch_size])
            yb = jnp.array(y_shuf[start : start + batch_size])
            train_step_fn(model, optimizer, xb, yb)

        val_loss = _compute_val_loss(model, loss_fn, X_val, y_val, batch_size)

        if verbose:
            print(f"[{phase_name}] epoch {epoch + 1}/{epochs}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            if verbose:
                print(f"[{phase_name}] early stopping at epoch {epoch + 1}")
            break


class DragonNet:
    """JAX/flax.nnx DragonNet for treatment effect estimation.

    Ports the TF DragonNet to JAX using flax.nnx, exposing an identical
    sklearn-style API. Two-phase training: Adam warm-up followed by SGD
    with Nesterov momentum.

    Args:
        neurons_per_layer: Width of shared representation layers.
        targeted_reg: Whether to apply targeted regularization.
        ratio: Weight for the targeted regularization loss component.
        val_split: Fraction of training data reserved for validation.
        batch_size: Mini-batch size.
        epochs: Maximum SGD epochs.
        learning_rate: SGD learning rate.
        momentum: SGD Nesterov momentum.
        reg_l2: L2 regularization coefficient for outcome-head kernels.
        use_adam: Whether to run an Adam warm-up phase before SGD.
        adam_epochs: Maximum Adam epochs.
        adam_learning_rate: Adam learning rate.
        loss_func: Base loss function; defaults to dragonnet_loss_binarycross.
        verbose: Whether to print epoch summaries.
        seed: Random seed for parameter initialization and data shuffling.
    """

    def __init__(
        self,
        neurons_per_layer=200,
        targeted_reg=True,
        ratio=1.0,
        val_split=0.2,
        batch_size=64,
        epochs=100,
        learning_rate=1e-5,
        momentum=0.9,
        reg_l2=0.01,
        use_adam=True,
        adam_epochs=30,
        adam_learning_rate=1e-3,
        loss_func=dragonnet_loss_binarycross,
        verbose=True,
        seed=0,
    ):
        self.neurons_per_layer = neurons_per_layer
        self.targeted_reg = targeted_reg
        self.ratio = ratio
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reg_l2 = reg_l2
        self.use_adam = use_adam
        self.adam_epochs = adam_epochs
        self.adam_learning_rate = adam_learning_rate
        self.loss_func = loss_func
        self.verbose = verbose
        self.seed = seed
        self._model = None

    def _build_model(self, input_dim):
        rngs = nnx.Rngs(self.seed)
        return DragonNetModule(
            input_dim=input_dim,
            neurons_per_layer=self.neurons_per_layer,
            reg_l2=self.reg_l2,
            rngs=rngs,
        )

    def fit(self, X, treatment, y, p=None):
        """Fits the DragonNet model.

        Args:
            X: Feature matrix of shape (n, p). Accepts np.ndarray or pd.DataFrame.
            treatment: Binary treatment vector of shape (n,).
            y: Outcome vector of shape (n,).
            p: Ignored (kept for API compatibility with meta-learners).
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        targets = np.hstack((y.reshape(-1, 1), treatment.reshape(-1, 1))).astype(
            np.float32
        )
        X = X.astype(np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X, targets, test_size=self.val_split, random_state=self.seed
        )

        self._model = self._build_model(X.shape[1])

        loss_fn = (
            make_tarreg_loss(ratio=self.ratio, dragonnet_loss=self.loss_func)
            if self.targeted_reg
            else self.loss_func
        )
        train_step_fn = _make_train_step(loss_fn)

        if self.use_adam:
            adam_tx = optax.adam(self.adam_learning_rate)
            adam_opt = nnx.Optimizer(self._model, adam_tx, wrt=nnx.Param)
            _run_training_loop(
                model=self._model,
                optimizer=adam_opt,
                train_step_fn=train_step_fn,
                loss_fn=loss_fn,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=self.adam_epochs,
                batch_size=self.batch_size,
                early_stop_patience=2,
                reduce_lr_patience=5,
                reduce_lr_factor=0.5,
                verbose=self.verbose,
                phase_name="Adam",
            )

        sgd_tx = optax.sgd(self.learning_rate, momentum=self.momentum, nesterov=True)
        sgd_opt = nnx.Optimizer(self._model, sgd_tx, wrt=nnx.Param)
        _run_training_loop(
            model=self._model,
            optimizer=sgd_opt,
            train_step_fn=train_step_fn,
            loss_fn=loss_fn,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            early_stop_patience=40,
            reduce_lr_patience=5,
            reduce_lr_factor=0.5,
            verbose=self.verbose,
            phase_name="SGD",
        )

    def predict(self, X, treatment=None, y=None, p=None):
        """Runs forward pass on fitted DragonNet.

        Args:
            X: Feature matrix of shape (n, p).
            treatment: Ignored (API compatibility).
            y: Ignored (API compatibility).
            p: Ignored (API compatibility).

        Returns:
            np.ndarray of shape (n, 4): columns are (y0, y1, propensity, epsilon).
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float32)
        return np.array(self._model(jnp.array(X)))

    def predict_propensity(self, X):
        """Predicts individual propensity scores.

        Args:
            X: Feature matrix of shape (n, p).

        Returns:
            np.ndarray of shape (n,) with propensity scores.
        """
        return self.predict(X)[:, 2]

    def predict_tau(self, X):
        """Predicts individual treatment effects (ITE).

        Args:
            X: Feature matrix of shape (n, p).

        Returns:
            np.ndarray of shape (n, 1) with treatment effect estimates.
        """
        preds = self.predict(X)
        return (preds[:, 1] - preds[:, 0]).reshape(-1, 1)

    def fit_predict(self, X, treatment, y, p=None, return_components=False):
        """Fits the model and returns treatment effect estimates.

        Args:
            X: Feature matrix of shape (n, p).
            treatment: Binary treatment vector of shape (n,).
            y: Outcome vector of shape (n,).
            p: Ignored (API compatibility).
            return_components: Ignored for now; always returns ITEs.

        Returns:
            np.ndarray of shape (n, 1) with treatment effect estimates.
        """
        self.fit(X, treatment, y)
        return self.predict_tau(X)

    def save(self, path):
        """Saves model parameters to an orbax checkpoint directory.

        Args:
            path: Directory path where the checkpoint will be written.
        """
        import orbax.checkpoint as ocp

        path = str(path)
        state = nnx.state(self._model)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path, state)
        checkpointer.wait_until_finished()

    def load(self, path, input_dim):
        """Restores model parameters from an orbax checkpoint.

        Args:
            path: Directory path of a previously saved checkpoint.
            input_dim: Number of input features (needed to reconstruct the model).
        """
        import orbax.checkpoint as ocp

        path = str(path)
        self._model = self._build_model(input_dim)
        state = nnx.state(self._model)
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(path, state)
        nnx.update(self._model, restored)
