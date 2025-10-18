import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import random
from scipy.stats import norm
import pandas as pd


print('Q4')
url = "http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv"
df = pd.read_csv(url)
print(df.head())

print('Q4.a')
def train_test_split (df) :
    train = df.sample(frac=2/3, random_state=0)
    test = df.drop(train.index)
    xtrain = train.drop(columns=['MEDV'])
    xtest = test.drop(columns=['MEDV'])
    ytrain = train['MEDV']
    ytest = test['MEDV']

    return xtrain, xtest, ytrain, ytest

xtrain, xtest, ytrain, ytest = train_test_split(df)
def ones_basis (x,k=None) :
    return np.ones(len(x),1)
lin_reg_sol_k(xtrain, ytrain, basis_func=ones_basis)
train_ones = ones_basis(xtrain)
test_ones = ones_basis(xtest)
test_ones = np.ones(len(xtest))

