import pandas as pd
import numpy as np

### One Hot Encoding to convert categorical data to binary values 0's and 1's

def one_hot_encoding(X: pd.DataFrame, features) -> pd.DataFrame:
    
    df_encoded = X.copy()    

    for feature in features:
        if feature in X.columns:
            dummies = pd.get_dummies(df_encoded[feature], prefix=feature)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(feature, axis=1, inplace=True)
    
    return df_encoded

### Preprocessing

def preprocessing(X:pd.DataFrame,y:pd.Series):
    real_features = []
    discrete_features = []
    for col in X.columns:
        if check_ifreal(X[col]):
            real_features.append(col)
        else:
            discrete_features.append(col)
        
    X_processed = one_hot_encoding(X, discrete_features)
    
    if check_ifreal(y):
        y_processed=y
        print("Real output\n")

    else:
        y_processed=y
        print("\nDiscrete output\n")

    return X_processed,y_processed

### Chek if data has real output or discrete output

def check_ifreal(y: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(y):
        return True  
    else:
        return False
    
### Calculate Entropy in case of discrete output

def entropy(Y: pd.Series) -> float:
    y_i=Y.value_counts()
    y_count=len(Y)
    p_i=y_i/y_count
    p_i = p_i.replace(0, 1e-20)
    entropy = -np.sum(p_i*np.log2(p_i))
    return entropy

### Calculate gini_index

def gini_index(Y: pd.Series) -> float:
    y_i=Y.value_counts()
    y_count=len(Y)
    p_i=y_i/y_count
    gini_index = 1-np.sum(p_i**2)
    return gini_index

### Calculate mean squared error

def MSE(Y: pd.Series) -> float:
    avg=np.mean(Y)
    err=Y-avg
    sqr_err=(err)**2
    mean_sqr_err=np.mean(sqr_err)
    return mean_sqr_err

### Calculate information gain

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    
    info_gain=0
    
    if criterion=='entropy':
        parent_entr=entropy(Y)
    if criterion=='gini_index':
        parent_gini=gini_index(Y)
    if criterion=='mse':
        parent_mse=MSE(Y)
        
    weighted_criterion=0
    
    for attr_val in attr.unique():
        mask = attr == attr_val
        Y_attr_val = Y.loc[mask.index[mask]]
        weighted_Y_val=len(Y_attr_val)/len(Y)
        if criterion=='entropy':
            weighted_criterion+=weighted_Y_val*entropy(Y_attr_val)
        if criterion=='gini_index':
            weighted_criterion+=weighted_Y_val*gini_index(Y_attr_val)
        if criterion=='mse':
            weighted_criterion+=weighted_Y_val*MSE(Y_attr_val)
            
    if criterion=='entropy':
        info_gain=parent_entr-weighted_criterion
    if criterion=='gini_index':
        info_gain=parent_gini-weighted_criterion
    if criterion=='mse':
        info_gain=parent_mse-weighted_criterion
        
    return info_gain

### Find optimal split attribute to split about

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    opt_gain=-np.inf
    opt_attr=None
    opt_split=None
    opt_crit=np.inf
    
    
    for feature in features:
        
        X_feat=X[feature]
        
        if check_ifreal(X[feature]):
            total_gain=0
            wt_crit=0
            split_value=0
            sorted_indices = np.argsort(X_feat)
            sorted_X_feat = X_feat.iloc[sorted_indices]  
            sorted_y = y.iloc[sorted_indices]
            for i in range(len(sorted_y) - 1):
                split_value=(sorted_X_feat.iloc[i]+sorted_X_feat.iloc[i + 1])/2
                LHS=y[X_feat<=split_value]
                RHS=y[X_feat>split_value]
                if criterion=="entropy":
                    LHS_entr=entropy(LHS)
                    RHS_entr=entropy(RHS)
                    wt_crit=(len(LHS)/len(y))*LHS_entr+(len(RHS)/len(y))*RHS_entr
                if criterion=="gini_index":
                    LHS_gini=gini_index(LHS)
                    RHS_gini=gini_index(RHS)
                    wt_crit=(len(LHS)/len(y))*LHS_gini+(len(RHS)/len(y))*RHS_gini
                if criterion=="mse":
                    LHS_mse=MSE(LHS)
                    RHS_mse=MSE(RHS)
                    wt_crit=(len(LHS)/len(y))*LHS_mse+(len(RHS)/len(y))*RHS_mse
                if wt_crit<opt_crit:
                    opt_crit=wt_crit
                    opt_attr=feature
                    opt_split=split_value
                    
        else:
            total_gain=0
            unique_values = X[feature].unique()
            for value in unique_values:
                total_gain=information_gain(y,X_feat[X_feat==value],criterion)
                if total_gain>opt_gain:
                    opt_gain=total_gain
                    opt_attr=feature
                    opt_split=value
                    
    return opt_attr,opt_split

### Split data about particular value of a particular attribute

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    if isinstance(value, float): #for real input
        X_left = X[X[attribute] <= value]
        X_right = X[X[attribute] > value]
        y_left = y[X[attribute] <= value]
        y_right = y[X[attribute] > value]
        
    else:
        X_left = X[X[attribute] == value]
        X_right = X[X[attribute] != value]
        y_left = y[X[attribute] == value]
        y_right = y[X[attribute] != value]
        
   
    return X_left, X_right, y_left, y_right

