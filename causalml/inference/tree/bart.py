import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as ss
import statsmodels.api as sm
from typing import List, Tuple, Dict

class Data:
    def __init__(self, 
                 X:np.ndarray, 
                 y:np.ndarray, 
                 treatment:np.ndarray=None, 
                 R:np.ndarray=None, 
                 index:np.ndarray=None, 
                 available_values_per_col:Dict[int, np.ndarray]=None, 
                 available_predictors_for_split:List[int]=None):
        self.X = X
        self.treatment = treatment
        self.y = y
        self.R = R
        self.index = index
        self.get_available_predictors_for_split()
    
    def get_available_predictors_for_split(self):
        self.available_predictors_for_split = []
        for i in range(self.X.shape[1]):
            if self.X[:, i].size > 1:
                self.available_predictors_for_split.append(i)
                
#         if self.treatment is not None:
#             self.available_predictors_for_split[-1] = discrete_unique_values(self.treatment)
            
class Node:
    def __init__(self, 
                 type_:str = 'terminal', 
                 left:'Node' = None, 
                 right:'Node' = None, 
                 feature:int = None, 
                 threshold:float = 0., 
                 value:float = 0., 
                 depth:int = 0, 
                 parent:'Node' = None, 
                 data:Data = None):
        self.type_ = type_
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold 
        self.value = value
        self.depth = depth
        self.parent = parent
        self.data = data
        if data:
            self.nobs = self.data.X.shape[0]
      
    def apply_rule(self, x:np.ndarray) -> 'Node':
        if x[self.feature] < self.threshold:
            return self.left
        else:
            return self.right
    
    def predict(self) -> float:
        return self.value
        
class Tree:
    def __init__(self, root: Node):
        self.root = root
        self.nodes = []
        self.leaves = []
        self.leaves_parents = []
        self.update()
        self.predictors = list(range(self.root.data.X.shape[1]))
        self.root.predictors_used = []
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            node = self.root
            while node.left:
                node = node.apply_rule(x=x)
            predictions.append(node.predict())
        return np.array(predictions).reshape(-1, 1)
    
    def update_residuals(self, residuals:np.ndarray) -> None:
        self.root.data.R = residuals
        self.traverse(self.root, type_='update_residuals')
        
    def traverse(self, root:Node, type_:str = 'update_leaves'):
        if type_ == 'update_leaves':
            if not root.left and not root.right:
                self.leaves.append(root)
                return
        elif type_ == 'get_second_gen_nodes':
            if not root.left and not root.right:
                return
            if root.left.type_ == 'terminal' and root.right.type_ == 'terminal':
                self.second_gen_internal_nodes.append(root)
                return
        elif type_ == 'get_leaves_parents':
            if not root.left and not root.right:
                return
            if root.left.type_ == 'terminal' and root.right.type_ == 'terminal':
                self.leaves_parents.append(root)
                return
        elif type_ == 'update_residuals':
            if not root.left and not root.right:
                return
            if root.left.type_ == 'terminal' and root.right.type_ == 'terminal':
                root.data.R = self.root.data.R[root.data.index]
                return
        self.traverse(root.left, type_=type_)
        self.traverse(root.right, type_=type_)
        
    def get_leaves(self) -> List[Node]:
        self.leaves = []
        self.traverse(self.root, type_='update_leaves')
        return self.leaves

    def get_n_second_gen_internal_nodes(self, return_nodes:bool=False) -> int:
        if len(self.leaves) == 1:
            return 1
        self.second_gen_internal_nodes = []
        self.traverse(self.root, type_='get_second_gen_nodes')
        if return_nodes:
            return self.second_gen_internal_nodes
        else:
            return len(self.second_gen_internal_nodes)
    
    def get_leaves_parents(self) -> List[Node]:
        self.leaves_parents = []
        self.traverse(self.root, type_='get_leaves_parents')
        return self.leaves_parents
    
    def get_nodes(self) -> List[Node]:
        return self._get_nodes(self.root)
    
    def _get_nodes(self, root:Node) -> List[Node]:
        self.nodes = []
        self._get_nodes(root.left)
        if root.left and root.right:
            self.nodes.append(root)
        self._get_nodes(root.right)
        return self.nodes
    
    def update(self) -> None:
        self.leaves = []
        self.leaves_parents = []
        self.traverse(self.root, type_='update_leaves')
        self.traverse(self.root, type_='get_leaves_parents')

class Proposal:
    def __init__(self, 
                 type_,
                 tree, 
                 problem_type,
                 p_grow, 
                 p_prune, 
                 sigma, 
                 sigma_mu, 
                 alpha, 
                 beta):
        self.type_ = type_
        self.tree = tree
        self.problem_type = problem_type
        self.beta = beta
        self.alpha = alpha
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.sigma_mu = sigma_mu
        self.sigma = sigma
        if self.problem_type == 'uplift_modeling':
            self.predictors = self.tree.predictors[:-1] + [-1]
        else:
            self.predictors = self.tree.predictors
        if self.type_ == 'grow':
            self.b = len(tree.leaves)
            self.node_to_modify = np.random.choice(tree.leaves)
        else:
            self.b = len(tree.leaves) - 1
            self.node_to_modify = np.random.choice(tree.get_n_second_gen_internal_nodes(return_nodes=True))
        self.r = None
        # if root, choose treatment variable when problem type is uplift modeling
        if len(self.tree.leaves) == 1 and self.problem_type == 'uplift_modeling':
            # set split to be the treatment variable
            self.proposed_predictor = -1
            self.proposed_value = 0.5
            self.r = 1 # accept grow always
        else:
            self.proposed_predictor = np.random.choice(self.node_to_modify.data.available_predictors_for_split)
            self.proposed_value = np.random.normal(self.node_to_modify.data.X[:, self.proposed_predictor].mean(), self.node_to_modify.data.X[:, self.proposed_predictor].std())
            self.p_adj_eta = self.get_p_adj()
            self.nj_adj_eta = self.get_n_adj()
            self.w2 = tree.get_n_second_gen_internal_nodes()
        if self.type_ == 'grow':
            self.left_node, self.right_node = self.create_split()
        else:
            self.left_node, self.right_node = self.node_to_modify.left, self.node_to_modify.right

    def get_p_adj(self) -> int:
        return len(self.node_to_modify.data.available_predictors_for_split)

    def get_n_adj(self) -> int:
        return self.node_to_modify.data.X[:, self.proposed_predictor].size
    
    def get_transition_ratio(self) -> float:
        return (self.p_prune * self.b * self.p_adj_eta * self.nj_adj_eta) / (self.p_grow * self.w2)

    def get_node_weighted_averate(self, node:Node) -> float:
        return node.data.R.sum()**2 / (self.sigma**2 + node.nobs * self.sigma_mu**2)    

    def create_split(self) -> Tuple[Node]:
        X_data = self.node_to_modify.data.X
        y_data = self.node_to_modify.data.y
        R_data = self.node_to_modify.data.R
        t_data = self.node_to_modify.data.treatment
        i_data = self.node_to_modify.data.index
        mask_left = X_data[:, self.proposed_predictor] < self.proposed_value
        mask_right = X_data[:, self.proposed_predictor] >= self.proposed_value
        
        if (X_data[np.where(mask_left)].shape[0] <= 1 or X_data[np.where(mask_right)].shape[0] <= 1):
            self.r = 0 # reject always if data in leaf is not minimum size
            return None, None
            
        available_values_per_col = None
        available_predictors_for_split = None
        left_data = Data(X=X_data[np.where(mask_left)], 
                         y=y_data[np.where(mask_left)], 
                         R=R_data[np.where(mask_left)], 
                         index= i_data[np.where(mask_left)], 
                         treatment=t_data[np.where(mask_left)] if self.problem_type == 'uplift_modeling' else None,                      
                         available_values_per_col=available_values_per_col,
                         available_predictors_for_split=available_predictors_for_split)
        right_data = Data(X=X_data[np.where(mask_right)], 
                          y=y_data[np.where(mask_right)], 
                          R=R_data[np.where(mask_right)], 
                          index= i_data[np.where(mask_right)], 
                          treatment=t_data[np.where(mask_right)] if self.problem_type == 'uplift_modeling' else None,                      
                          available_values_per_col=available_values_per_col,
                          available_predictors_for_split=available_predictors_for_split)
        depth = self.node_to_modify.depth + 1
        return (
                Node(data=left_data, 
                     depth=depth, 
                     type_='terminal',
                     parent=self.node_to_modify, 
                    ), 
                Node(data=right_data, 
                     depth=depth, 
                     type_='terminal',
                     parent=self.node_to_modify,
                    )
        )

    def get_likelihood_ratio(self) -> float:
        nl = self.node_to_modify.nobs
        nll = self.left_node.nobs
        nlr = self.right_node.nobs
        sv = self.sigma**2 
        left_term = np.sqrt((sv * (sv + nl*sv)) / ( (sv + nll*self.sigma_mu**2) * (sv + nlr*self.sigma_mu**2)))
        nlwa = self.get_node_weighted_averate(node=self.left_node)
        nrwa = self.get_node_weighted_averate(node=self.right_node)
        nwa = self.get_node_weighted_averate(node=self.node_to_modify)
        likelihood_ratio = left_term * np.exp((self.sigma_mu**2 / (2*sv)) * (nlwa + nrwa - nwa))
        
        return likelihood_ratio
    
    def get_tree_structure_ratio(self) -> float:
        numerator = self.alpha * (1 - (self.alpha / (2 + self.node_to_modify.depth)**self.beta) )**2
        denominator = ((1 + self.node_to_modify.depth)**self.beta - self.alpha) * self.p_adj_eta * self.nj_adj_eta
        return numerator / denominator
    
    def compute_r(self) -> float:
        if self.r is None:
            tr = self.get_transition_ratio()
            lr = self.get_likelihood_ratio()
            tsr = self.get_tree_structure_ratio()
            if self.type_ == 'grow':
                r = tr * lr * tsr
            elif self.type_ == 'prune':
                r = 1/tr * 1/lr * 1/tsr
            return r
        else:
            return self.r
    
    def accept(self) -> Tree:
        if self.type_ == 'grow':
            self.node_to_modify.type_ = 'split'
            self.node_to_modify.left = self.left_node
            self.node_to_modify.right = self.right_node
            self.node_to_modify.feature = self.proposed_predictor
            self.node_to_modify.threshold = self.proposed_value
            self.tree.update()
        else:
            self.node_to_modify.type_ = 'terminal'
            self.node_to_modify.left = None
            self.node_to_modify.right = None
            self.tree.update()
        return self.tree

class BART:
    """A class that implements the logic for Bayesian Additive Regression Trees for both causal
    inference and classic ML settings.
    
    References:
    [1] Chipman et al. (2010) (https://arxiv.org/abs/0806.3286)
    [2] Hill (2011) (https://www.researchgate.net/publication/236588890_Bayesian_Nonparametric_Modeling_for_Causal_Inference)
    [3] Kapelner and Bleich (2014) (https://arxiv.org/abs/1312.2171)
    [4] Tan and Roy (2019) (https://arxiv.org/abs/1901.07504)
    """
    
    def __init__(self, v:int = 3, q:float = 0.9, k:int = 2, m:int = 200, alpha:float = 0.95, beta:float = 2) -> None:
        """Initialize BART.
        Args:
            v (optional): Parameter for sigma prior's alpha and beta calculation. Default = 3.
            q (optional): Parameter for sigma prior's definition. Default = 0.9.
            k (optional): Parameter initializing sigma_mu. Default = 2.
            m (optional): Number of trees. Default = 200.
            alpha (optional): Parameter used for calculating the probability of selecting a leaf node to be modified. 
                                Default = 0.95
            beta (optional): Parameter used for calculating the probability of selecting a leaf node to be modified. 
                                Default = 2
        """        
        self.v = v
        self.q = q
        self.k = k
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.available_mutations = ['grow', 'prune']
        self.available_mutations_prob = [0.5, 0.5]

    def sample_ig(self, alpha:float, beta:float) -> float:
        """A method to sample from the inverse gamma distribution.
        """
        return ss.invgamma(alpha, beta).rvs()
    
    def grow_prune(self, sigma:float, sigma_mu:float, tree:Tree) -> Tree:
        """A method to sample a new tree.
        """
        mutation_type = 'grow' if len(tree.leaves) == 1 else np.random.choice(self.available_mutations, p=self.available_mutations_prob)
        tree_copy = copy.deepcopy(tree)
        p = Proposal(tree=tree_copy,
                     type_=mutation_type,
                     problem_type=self.problem_type,
                     p_grow=self.available_mutations_prob[0], 
                     p_prune=self.available_mutations_prob[1],
                     sigma=sigma,
                     sigma_mu=sigma_mu,
                     alpha=self.alpha,
                     beta=self.beta)
        r = p.compute_r()
        unif_sample = np.random.uniform(0, 1)
        if unif_sample < r:
            tree = p.accept()
            return tree
        else:
            return tree

    def get_all_other_trees_predictions(self, X:np.ndarray, trees:List[Tree], j:int) -> np.ndarray:
        """Computes the predictions of all trees but j
        """
        predictions = np.zeros((X.shape[0], 1))
        for i in range(len(trees)):
            if i != j:
                predictions += trees[i].predict(X)
        return predictions

    def compute_residual(self, X:np.ndarray, y:np.ndarray, trees:List[Tree], j:int) -> np.ndarray:
        """Computes the residual Rj according to eq.12 from [1]
        """
        predictions = self.get_all_other_trees_predictions(X, trees, j)
        return y.reshape(-1, 1) - predictions
    
    def get_lambda(self, X:np.ndarray, y:np.ndarray) -> float:
        """Calculates the lambda parameter to be used for sigma's prior definition
        """
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        return results.resid.std().item()**2 * 0.9
        
    def sample_gd(self, mu:float, sigma:float) -> float:
        """Sample from gaussian distribution
        """
        return np.random.normal(loc=mu, scale=sigma)

    def sample_leaves(self, tree:Tree, sigma_mu:float, sigma:float) -> None:
        """Sample leaf values according to eq.16 from [1]
        """
        for leaf in tree.leaves:
            sms = sigma_mu**2
            ss = sigma**2 / leaf.nobs
            residuals_mean = leaf.data.R.sum() / leaf.nobs
            sms = 1. / (1. / sms + 1. / ss)
            pm = residuals_mean * (sms / (ss + sms))
            leaf.value = pm + (np.random.normal() * np.sqrt(sms / self.m))

    def rescale_y(self, transformed_y:np.ndarray) -> np.ndarray:
        """Transform response from [-0.5, 0.5] to original scale
        """
        return (transformed_y + 0.5) * (self.y_max - self.y_min) + self.y_min
    
    def sigmoid(self, x:np.ndarray) -> np.ndarray:
        """Calculates the lambda parameter to be used for sigma's prior definition
        """
        return 1 / (1 + np.exp(-x))
    
    def transform_y(self, y:np.ndarray) -> np.ndarray:
        """Transform response to a [-0.5, 0.5] interval
        """
        return ((y - self.y_min) / (self.y_max - self.y_min)) - 0.5
        
    def min_el(self, x:np.ndarray) -> np.ndarray:
        """Clip values greater than 0
        """
        return np.array(list(map(lambda z: min(z, 0), x.flatten()))).reshape(-1, 1)

    def max_el(self, x:np.ndarray) -> np.ndarray:
        """Clip values less than 0
        """
        return np.array(list(map(lambda z: max(z, 0), x.flatten()))).reshape(-1, 1)
    
    def fit(self, X:np.ndarray, y:np.ndarray, treatment:np.ndarray = None, max_iter:int = 100) -> None:
        """Fit the BART for either uplift modeling or classic ML setting.
        Args:
            X (np.array): a feature matrix
            y (np.array): an outcome vector
            treatment (np.array): a treatment vector
            max_ter (int): maximum number of iterations. Stopping criteria 
        """

        if treatment is not None:
            self.problem_type = 'uplift_modeling'
            if np.unique(treatment).size > 2:
                raise Exception('Number of treatments greater than 2. Currently this method only supports binary treatment.')
            self.treatments = np.unique(treatment).astype(int)
            X = np.concatenate([X, treatment], axis=1)
        else:
            self.problem_type = 'classic_ml'
            
        self.task = 'regression' if len(np.unique(y)) > 2 else 'classification'
        self.y_min = y.min()
        self.y_max = y.max()
        
        if self.task == 'regression':
            sigma_mu = 0.5 / (self.k * np.sqrt(self.m))
            transformed_y = self.transform_y(y=y)
            self.lambda_ = self.get_lambda(X, transformed_y)
            alpha_ig = self.v / 2 
            beta_ig = self.v * self.lambda_ / 2
            sigma = self.sample_ig(alpha=alpha_ig, beta=beta_ig)
        else:
            transformed_y = y
            sigma_mu = 3 / (self.k * np.sqrt(self.m))
            sigma = 1
            
        mean_value_per_tree = transformed_y.mean() / self.m
        self.sum_of_trees = transformed_y.mean()
        self.trees = [Tree(root=Node(value=mean_value_per_tree, 
                                     type_='terminal', 
                                     data=Data(X=X, 
                                               treatment=treatment,
                                               y=transformed_y, 
                                               R=transformed_y-mean_value_per_tree*(self.m-1),
                                               index=np.array(list(range(len(X)))))), 
                          ) for i in range(self.m)]
    
        for i in tqdm(range(max_iter), position=0, leave=True):
            if self.task == 'regression':
                response = transformed_y
            else:
                sample = np.random.normal(loc=self.sum_of_trees, scale=1, size=(len(transformed_y), 1))
                max_els = self.max_el(sample)
                min_els = self.min_el(sample)
                response = np.where(y.reshape(-1, 1) == 1, max_els, min_els) # Z in the paper

            for j in range(len(self.trees)):
                residuals = self.compute_residual(X=X, y=response, trees=self.trees, j=j) # opportunity to parallelize according to Pratola et al. (2013) (https://arxiv.org/abs/1309.1906)
                self.trees[j].update_residuals(residuals)
                new_tree = self.grow_prune(sigma=sigma, sigma_mu=sigma_mu, tree=self.trees[j])
                self.sample_leaves(tree=new_tree, sigma_mu=sigma_mu, sigma=sigma)
                self.trees[j] = new_tree

            y_pred = self.predict(X, train=True) # opportunity to parallelize according to Pratola et al. (2013) (https://arxiv.org/abs/1309.1906)
            self.sum_of_trees = y_pred
            if self.task == 'regression':
                squared_error = (transformed_y.reshape(-1, 1) - y_pred)**2
                p_alpha = (alpha_ig + len(y_pred)) / 2
                p_beta = (beta_ig + squared_error.sum()) / 2
                sigma = 1. / np.sqrt(np.random.gamma(p_alpha, 1./p_beta))

    def _predict(self, X:np.ndarray, rescale:bool) -> np.ndarray:
        """Private method for prediction that works equally for Classic ML and Uplift Modeling
        """
        # predict for each tree and aggregate
        predictions = np.zeros((X.shape[0], 1))
        for tree in self.trees:
            predictions += tree.predict(X)
            
        if self.task == 'classification':
            return self.sigmoid(predictions)
        
        if rescale:
            predictions = self.rescale_y(predictions)
        
        return predictions
    
    def predict(self, X:np.ndarray, train:bool = False, rescale:bool = False, full_output:bool = False) -> np.ndarray:
        """Predict treatment effects if the problem type is uplift modeling or one-dimensional response if 
        problem type is classic ml.
        Args:
            X (np.array): A feature matrix
            train (bool): Optional. To be used only by private methods. Distinguishes if prediction to be done for
                            each treatment in case of uplift modeling setting.
            rescale (bool): True if output wanted to be in the same scale as the original y.
            full_output (bool): True if the predictions for each treatment, alongside with the deltas with control
                                    as baseline. False if only deltas to be returned.
        Returns:
            (pd.DataFrame): Predictions of treatment effects if the problem type is uplift modeling or one-dimensional 
                                response if problem type is classic ml.
        """
        if self.problem_type == 'uplift_modeling' and not train:
            columns = []
            predictions = []
            for treatment in self.treatments:
                treatment_predictions = self._predict(np.concatenate([X, np.ones((len(X), 1)) * treatment], axis=1), rescale)
                predictions.append(treatment_predictions.flatten().tolist())
                columns.append(f'treatment_{treatment}' if treatment > 0 else 'control')
            
            df_res = pd.DataFrame(np.array(predictions).T, columns=columns)
            
            # From: https://github.com/uber/causalml/blob/c42e873061eb74ec9c3ca6ea991e113b886245ae/causalml/inference/tree/uplift.pyx
            df_res['recommended_treatment'] = df_res.apply(np.argmax, axis=1)

            # Calculate delta
            for treatment in self.treatments[1:]:
                df_res[f'delta_{treatment}'] = df_res[f'treatment_{treatment}'] - df_res['control']

            if full_output:
                return df_res
            else:
                return df_res[[col for col in df_res.columns if 'delta' in col]].values
        else:
            predictions = self._predict(X, rescale)
        return predictions