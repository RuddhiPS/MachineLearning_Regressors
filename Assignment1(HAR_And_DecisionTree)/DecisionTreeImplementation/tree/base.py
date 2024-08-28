import os
print(os.getcwd()) 

os.chdir('.')
print(os.getcwd()) 

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

class Tree_node:
    def __init__(self, attribute=None, split_value=None, left_child=None,right_child=None, value=None):
        
        self.attribute=attribute   
        self.split_value=split_value
        self.left_child=left_child  
        self.right_child=right_child
        self.value=value  #label in case of  leaf node.
        
    def is_leaf(self):
        if self.left_child==None and self.right_child==None:
            return True
        else:
            return False

class DecisionTree:
    
    criterion: Literal["entropy", "gini_index","mse"] 
    max_depth: int  # The maximum depth the tree can grow to
        
    def __init__(self, criterion: Literal["entropy", "gini_index","mse"], max_depth: int = 5):
        
        self.criterion=criterion
        self.max_depth=max_depth
        self.tree_=None 
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
                
        def build_tree(X: pd.DataFrame, y: pd.Series, depth: int):

            if len(np.unique(y)) == 1:
                return Tree_node(value=y.iloc[0])  
            if depth >= self.max_depth:
                return Tree_node(value=y.mode()[0])

            
            optimal_attr, optimal_split = opt_split_attribute(X, y, self.criterion, X.columns)
            
            if optimal_attr is None:
                return Tree_node(value=y.mode()[0])  

            X_left, X_right, y_left, y_right = split_data(X, y, optimal_attr, optimal_split)

            if X_left.empty or X_right.empty:          
                return Tree_node(value=y.mode()[0])  
           
            left_tree = build_tree(X_left, y_left, depth + 1)
            right_tree = build_tree(X_right, y_right, depth + 1)

            return Tree_node(
                attribute=optimal_attr,
                split_value=optimal_split,
                left_child=left_tree,
                right_child=right_tree
            )
    
        self.tree_ = build_tree(X, y, depth=0)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:

        def traverse_tree(node: Tree_node, sample: pd.Series) -> any:
            if node.is_leaf():
                return node.value 

            if sample[node.attribute] <= node.split_value:
                return traverse_tree(node.left_child, sample)  
            else:
                return traverse_tree(node.right_child, sample)  

        predictions = X.apply(lambda row: traverse_tree(self.tree_, row), axis=1)
        return predictions
    
    def plot(self) -> None:
        
        def print_tree(node: Tree_node, depth: int = 0)->None:
            if node==None:
                return
            spaces="    "*depth
            if node.is_leaf():
                print(f'{spaces}Value : {node.value}')
            else:
                print(f'{spaces}If {node.attribute} <= {node.split_value} ?')
                print_tree(node.left_child,depth+1)
                print(f'{spaces}Else :\n')
                print_tree(node.right_child,depth+1)
        if self.tree_ != None:
            print_tree(self.tree_)
            

