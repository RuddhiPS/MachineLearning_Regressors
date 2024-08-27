from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    y_correct=(y_hat==y).sum()
    y_total=y.size
    accuracy=y_correct/y_total
    return accuracy

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    true_positive=((y_hat == cls) & (y == cls)).sum()  
    false_positive=((y_hat == cls) & (y != cls)).sum()  
    if (true_positive+false_positive)==0:
        precision=0
    else:   
        precision=true_positive/(true_positive+false_positive)
        return precision

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    true_positive=((y_hat == cls) & (y == cls)).sum()  
    false_negative=((y_hat != cls) & (y == cls)).sum()  
    if true_positive+false_negative==0:
        return 0
    recall=true_positive/(true_positive + false_negative)
    return recall

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    mse = ((y-y_hat)**2).mean()
    rmse=np.sqrt(mse)
    return rmse

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    mae=(abs(y-y_hat)).mean()
    return mae

