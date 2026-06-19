"""Loss functions for the JAX DragonNet implementation."""

import jax.numpy as jnp


def binary_classification_loss(concat_true, concat_pred):
    """Binary cross-entropy loss on the treatment (propensity) head.

    Args:
        concat_true: Array of shape (n, 2) where each row is (y, treatment).
        concat_pred: Array of shape (n, 4) where each row is
            (y0, y1, propensity, epsilon).

    Returns:
        Scalar binary cross-entropy loss.
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = -jnp.sum(
        t_true * jnp.log(t_pred + 1e-7) + (1.0 - t_true) * jnp.log(1.0 - t_pred + 1e-7)
    )
    return losst


def regression_loss(concat_true, concat_pred):
    """Weighted MSE loss on the two outcome heads.

    Args:
        concat_true: Array of shape (n, 2) where each row is (y, treatment).
        concat_pred: Array of shape (n, 4) where each row is
            (y0, y1, propensity, epsilon).

    Returns:
        Scalar regression loss.
    """
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    loss0 = jnp.sum((1.0 - t_true) * jnp.square(y_true - y0_pred))
    loss1 = jnp.sum(t_true * jnp.square(y_true - y1_pred))
    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    """Combined regression and binary classification loss.

    Args:
        concat_true: Array of shape (n, 2) where each row is (y, treatment).
        concat_pred: Array of shape (n, 4) where each row is
            (y0, y1, propensity, epsilon).

    Returns:
        Scalar combined loss.
    """
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(
        concat_true, concat_pred
    )


def make_tarreg_loss(ratio=1.0, dragonnet_loss=dragonnet_loss_binarycross):
    """Wrap a DragonNet loss with targeted regularization.

    Args:
        ratio: Weight assigned to the targeted regularization component.
        dragonnet_loss: Base loss function to augment.

    Returns:
        Loss function with targeted regularization.
    """

    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]
        epsilons = concat_pred[:, 3]

        t_pred = (t_pred + 0.01) / 1.02
        y_pred = t_true * y1_pred + (1.0 - t_true) * y0_pred
        h = t_true / t_pred - (1.0 - t_true) / (1.0 - t_pred)
        y_pert = y_pred + epsilons * h
        targeted_regularization = jnp.sum(jnp.square(y_true - y_pert))

        return vanilla_loss + ratio * targeted_regularization

    return tarreg_ATE_unbounded_domain_loss
