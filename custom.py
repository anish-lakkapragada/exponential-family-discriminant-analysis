from models import EvalResult 
from typing import List
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score

def get_upper_diagonal(X: np.ndarray) -> np.ndarray: 
    d = X.shape[1]
    i,j = np.triu_indices(d)
    return X[:, i, j]        

def run_log_reg_wishart(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> List[EvalResult]: 
    X_train_changed, X_test_changed = get_upper_diagonal(X_train), get_upper_diagonal(X_test)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_changed, y_train)
    y_train_pred, y_test_pred = log_reg.predict(X_train_changed), log_reg.predict(X_test_changed)
    train_eval_result = EvalResult(auc_roc=roc_auc_score(y_train, y_train_pred), 
                                   acc=np.sum(np.round(y_train_pred) == y_train) / y_train.shape[0]) 
    
    test_eval_result = EvalResult(auc_roc=roc_auc_score(y_test, y_test_pred), 
                                  acc=np.sum(np.round(y_test_pred) == y_test) / y_test.shape[0]) 
    
    return [train_eval_result, test_eval_result]


def run_log_reg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> List[EvalResult]: 
    if len(X_train.shape) == 1: 
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_train_pred, y_test_pred = log_reg.predict(X_train), log_reg.predict(X_test)
    train_eval_result = EvalResult(auc_roc=roc_auc_score(y_train, y_train_pred), 
                                   acc=np.sum(np.round(y_train_pred) == y_train) / y_train.shape[0]) 
    
    test_eval_result = EvalResult(auc_roc=roc_auc_score(y_test, y_test_pred), 
                                  acc=np.sum(np.round(y_test_pred) == y_test) / y_test.shape[0]) 
    
    return [train_eval_result, test_eval_result]