"""
This module calls the CEVAE[1] function implemented by pyro team. CEVAE demonstrates a number of innovations including:

- A generative model for causal effect inference with hidden confounders;
- A model and guide with twin neural nets to allow imbalanced treatment; and
- A custom training loss that includes both ELBO terms and extra terms needed to train the guide to be able to answer
counterfactual queries.

Generative model for a causal model with latent confounder z and binary treatment w:
        z ~ p(z)      # latent confounder
        x ~ p(x|z)    # partial noisy observation of z
        w ~ p(w|z)    # treatment, whose application is biased by z
        y ~ p(y|t,z)  # outcome
Each of these distributions is defined by a neural network.  The y distribution is defined by a disjoint pair of neural
networks defining p(y|t=0,z) and p(y|t=1,z); this allows highly imbalanced treatment.

**References**

[1] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling (2017).
    | Causal Effect Inference with Deep Latent-Variable Models.
    | http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf
    | https://github.com/AMLab-Amsterdam/CEVAE
"""

import logging
import torch
from pyro.contrib.cevae import CEVAE
from causalml.inference.meta.utils import convert_pd_to_np

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


class CEVAE_MODEL(object):
    def __init__(self, outcome_dist="studentt", latent_dim=20, hidden_dim=20, num_epochs=50,
                 batch_size=2048, learning_rate=0.0005, learning_rate_decay=0.02, num_samples=1000):
        """
        Initializes CEVAE.

            Args:
                outcome_dist (str): outcome distribution as one of: "bernoulli" , "exponential", "laplace", "normal",
                                    and "studentt"
                latent_dim (int) : Dimension of the latent variable
                hidden_dim (int) : Dimension of hidden layers of fully connected networks
                num_epochs (int): Number of training epochs
                batch_size (int): Batch size
                learning_rate (int): Learning rate
                learning_rate_decay (float/int): Learning rate decay over all epochs; the per-step decay rate will
                                                 depend on batch size and number of epochs such that the initial
                                                 learning rate will be learning_rate and the
                                                 final learning rate will be learning_rate * learning_rate_decay
                num_sample (int) : number of samples to calculate ITE
        """
        self.outcome_dist = outcome_dist
        self.laten_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def fit(self, X, treatment, y):
        """
        Fits CEVAE.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        self.cevae = CEVAE(outcome_dist=self.outcome_dist,
                           feature_dim=X.shape[-1],
                           latent_dim=self.latent_dim,
                           hidden_dim=self.hidden_dim)

        self.cevae.fit(x=torch.tensor(X, dtype=torch.float),
                       t=torch.tensor(treatment, dtype=torch.float),
                       y=torch.tensor(y, dtype=torch.float),
                       num_epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       learning_rate=self.learning_rate,
                       learning_rate_decay=self.learning_rate_decay)

    def predict(self, X):
        """
        Calls predict on fitted DragonNet.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.ndarray): Predictions of treatment effects.
        """
        return self.cevae.ite(torch.tensor(X, dtype=torch.float),
                              num_samples=self.num_samples,
                              batch_size=self.batch_size).cpu().numpy()

    def fit_predict(self, X, treatment, y):
        """
        Fits the CEVAE model and then predicts.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        Returns:
            (np.ndarray): Predictions of treatment effects.
        """
        self.fit(X, treatment, y)
        return self.predict(X)
