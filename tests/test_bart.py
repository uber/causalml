import numpy as np
import pandas as pd
from causalml.inference.tree import BART
from causalml.inference.tree.bart import Data, Node, Tree
from sklearn.model_selection import train_test_split
from causalml.metrics import get_cumgain

def test_Second_gen_internal_nodes(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()
    n9 = Node(data=Data(X=X, y=y), left=None, right=None, value=9, feature='f9', threshold=0.5, type_='terminal')
    n8 = Node(data=Data(X=X, y=y), left=None, right=None, value=8, feature='f8', threshold=0.5, type_='terminal')    
    n7 = Node(data=Data(X=X, y=y), left=None, right=None, value=7, feature='f7', threshold=0.5, type_='terminal')
    n6 = Node(data=Data(X=X, y=y), left=n8, right=n9, value=6, feature='f6', threshold=0.5, type_='split')
    n5 = Node(data=Data(X=X, y=y), left=None, right=None, value=5, feature='f5', threshold=0.5, type_='terminal')
    n4 = Node(data=Data(X=X, y=y), left=None, right=None, value=4, feature='f4', threshold=0.5, type_='terminal')
    n3 = Node(data=Data(X=X, y=y), left=n6, right=n7, value=3, feature='f3', threshold=0.5, type_='split')
    n2 = Node(data=Data(X=X, y=y), left=n4, right=n5, value=2, feature='f2', threshold=0.5, type_='split')
    n1 = Node(data=Data(X=X, y=y), left=n2, right=n3, value=1, feature='f1', threshold=0.5, type_='split')

    t = Tree(root=n1)
    assert t.get_n_second_gen_internal_nodes() == 2
    
def test_BART(generate_regression_data):
    y, X, w, tau, b, e = generate_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(np.concatenate([X, w.reshape(-1, 1), tau.reshape(-1, 1)], axis=1), 
                                                        y, 
                                                        stratify=pd.qcut(y, 10), 
                                                        test_size=0.25, 
                                                        random_state=42)
    tau_train = X_train[:, -1].reshape(-1, 1)
    w_train = X_train[:, -2].reshape(-1, 1)
    X_train = X_train[:, :-2]
    tau_test = X_test[:, -1].reshape(-1, 1)
    w_test = X_test[:, -2].reshape(-1, 1)
    X_test = X_test[:, :-2]
    
    bart = BART(m=35)
    bart.fit(X=X_train, treatment=w_train, y=y_train, max_iter=50)
    
    y_pred = bart.predict(X_test, rescale=True)
    auuc_metrics = pd.DataFrame({'cate_p':y_pred.flatten(), 
                                'tau':tau_test.flatten(), 
                                'w':w_test.flatten(), 
                                'y': y_test.flatten()})
    
    cumgain = get_cumgain(
        auuc_metrics, outcome_col='y', treatment_col='w', treatment_effect_col='tau'
    )

    assert cumgain['cate_p'].sum() > cumgain['Random'].sum()