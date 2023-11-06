import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import binary_accuracy


def binary_classification_loss(concat_true, concat_pred):
    """
    Implements a classification (binary cross-entropy) loss function for DragonNet architecture.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): binary cross-entropy loss
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    """
    Implements a regression (squared error) loss function for DragonNet architecture.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): aggregated regression loss
    """
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1.0 - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    """
    Implements regression + classification loss in one wrapper function.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): aggregated regression + classification loss
    """
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(
        concat_true, concat_pred
    )


def treatment_accuracy(concat_true, concat_pred):
    """
    Returns keras' binary_accuracy between treatment and prediction of propensity.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): binary accuracy
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    """
    Tracks the mean absolute value of epsilon.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): mean absolute value of epsilon
    """
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


def make_tarreg_loss(ratio=1.0, dragonnet_loss=dragonnet_loss_binarycross):
    """
    Given a specified loss function, returns the same loss function with targeted regularization.

    Args:
        ratio (float): weight assigned to the targeted regularization loss component
        dragonnet_loss (function): a loss function
    Returns:
        (function): loss function with targeted regularization, weighted by specified ratio
    """

    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        """
        Returns the loss function (specified in outer function) with targeted regularization.
        """
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


class EpsilonLayer(Layer):
    """
    Custom keras layer to allow epsilon to be learned during training process.
    """

    def __init__(self):
        """
        Inherits keras' Layer object.
        """
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        """
        Creates a trainable weight variable for this layer.
        """
        self.epsilon = self.add_weight(
            name="epsilon", shape=[1, 1], initializer="RandomNormal", trainable=True
        )
        super(EpsilonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]
