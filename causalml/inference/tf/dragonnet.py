"""
This module implements the Dragonnet [1], which adapts the design and training of neural networks to improve
the quality of treatment effect estimates. The authors propose two adaptations:

- A new architecture, the Dragonnet, that exploits the sufficiency of the propensity score for estimation adjustment.
- A regularization procedure, targeted regularization, that induces a bias towards models that have non-parametrically
  optimal asymptotic properties ‘out-of-the-box’. Studies on benchmark datasets for causal inference show these
  adaptations outperform existing methods. Code is available at github.com/claudiashi57/dragonnet

**References**

[1] C. Shi, D. Blei, V. Veitch (2019).
    | Adapting Neural Networks for the Estimation of Treatment Effects.
    | https://arxiv.org/pdf/1906.02120.pdf
    | https://github.com/claudiashi57/dragonnet
"""

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

from causalml.inference.tf.utils import (
    dragonnet_loss_binarycross,
    EpsilonLayer,
    regression_loss,
    binary_classification_loss,
    treatment_accuracy,
    track_epsilon,
    make_tarreg_loss,
)
from causalml.inference.meta.utils import convert_pd_to_np


class DragonNet:
    def __init__(
        self,
        neurons_per_layer=200,
        targeted_reg=True,
        ratio=1.0,
        val_split=0.2,
        batch_size=64,
        epochs=30,
        learning_rate=1e-3,
        reg_l2=0.01,
        loss_func=dragonnet_loss_binarycross,
        verbose=True,
    ):
        """
        Initializes a Dragonnet.
        """
        self.neurons_per_layer = neurons_per_layer
        self.targeted_reg = targeted_reg
        self.ratio = ratio
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.reg_l2 = reg_l2
        self.verbose = verbose

    def make_dragonnet(self, input_dim):
        """
        Neural net predictive model. The dragon has three heads.

        Args:
            input_dim (int): number of rows in input
        Returns:
            model (keras.models.Model): DragonNet model
        """
        inputs = Input(shape=(input_dim,), name="input")

        # representation
        x = Dense(
            units=self.neurons_per_layer,
            activation="elu",
            kernel_initializer="RandomNormal",
        )(inputs)
        x = Dense(
            units=self.neurons_per_layer,
            activation="elu",
            kernel_initializer="RandomNormal",
        )(x)
        x = Dense(
            units=self.neurons_per_layer,
            activation="elu",
            kernel_initializer="RandomNormal",
        )(x)

        t_predictions = Dense(units=1, activation="sigmoid")(x)

        # HYPOTHESIS
        y0_hidden = Dense(
            units=int(self.neurons_per_layer / 2),
            activation="elu",
            kernel_regularizer=l2(self.reg_l2),
        )(x)
        y1_hidden = Dense(
            units=int(self.neurons_per_layer / 2),
            activation="elu",
            kernel_regularizer=l2(self.reg_l2),
        )(x)

        # second layer
        y0_hidden = Dense(
            units=int(self.neurons_per_layer / 2),
            activation="elu",
            kernel_regularizer=l2(self.reg_l2),
        )(y0_hidden)
        y1_hidden = Dense(
            units=int(self.neurons_per_layer / 2),
            activation="elu",
            kernel_regularizer=l2(self.reg_l2),
        )(y1_hidden)

        # third
        y0_predictions = Dense(
            units=1,
            activation=None,
            kernel_regularizer=l2(self.reg_l2),
            name="y0_predictions",
        )(y0_hidden)
        y1_predictions = Dense(
            units=1,
            activation=None,
            kernel_regularizer=l2(self.reg_l2),
            name="y1_predictions",
        )(y1_hidden)

        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name="epsilon")
        concat_pred = Concatenate(1)(
            [y0_predictions, y1_predictions, t_predictions, epsilons]
        )
        model = Model(inputs=inputs, outputs=concat_pred)

        return model

    def fit(self, X, treatment, y, p=None):
        """
        Fits the DragonNet model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        y = np.hstack((y.reshape(-1, 1), treatment.reshape(-1, 1)))

        self.dragonnet = self.make_dragonnet(X.shape[1])

        metrics = [
            regression_loss,
            binary_classification_loss,
            treatment_accuracy,
            track_epsilon,
        ]

        if self.targeted_reg:
            loss = make_tarreg_loss(ratio=self.ratio, dragonnet_loss=self.loss_func)
        else:
            loss = self.loss_func

        self.dragonnet.compile(
            optimizer=Adam(lr=self.learning_rate), loss=loss, metrics=metrics
        )

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=self.verbose,
                mode="auto",
                min_delta=1e-8,
                cooldown=0,
                min_lr=0,
            ),
        ]

        self.dragonnet.fit(
            X,
            y,
            callbacks=adam_callbacks,
            validation_split=self.val_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=40, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=self.verbose,
                mode="auto",
                min_delta=0.0,
                cooldown=0,
                min_lr=0,
            ),
        ]

        sgd_lr = 1e-5
        momentum = 0.9
        self.dragonnet.compile(
            optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
            loss=loss,
            metrics=metrics,
        )
        self.dragonnet.fit(
            X,
            y,
            callbacks=sgd_callbacks,
            validation_split=self.val_split,
            epochs=300,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

    def predict(self, X, treatment=None, y=None, p=None):
        """
        Calls predict on fitted DragonNet.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.array): a 2D array with shape (X.shape[0], 4),
                where each row takes the form of (outcome do(t=0), outcome do(t=1), propensity, epsilon)
        """
        return self.dragonnet.predict(X)

    def predict_propensity(self, X):
        """
        Predicts the individual propensity scores.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.array): propensity score vector
        """
        preds = self.predict(X)
        return preds[:, 2]

    def predict_tau(self, X):
        """
        Predicts the individual treatment effect (tau / "ITE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.array): treatment effect vector
        """
        preds = self.predict(X)
        return (preds[:, 1] - preds[:, 0]).reshape(-1, 1)

    def fit_predict(self, X, treatment, y, p=None, return_components=False):
        """
        Fits the DragonNet model and then predicts.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            return_components (bool, optional): whether to return
        Returns:
            (np.array): predictions based on return_components flag
                if return_components=False (default), each row is treatment effect
                if return_components=True, each row is (outcome do(t=0), outcome do(t=1), propensity, epsilon)
        """
        self.fit(X, treatment, y)
        return self.predict_tau(X)
