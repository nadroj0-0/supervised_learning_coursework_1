import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import random
from scipy.stats import norm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#  testing branching

# Training and Test Data Splitter
def train_test_split (df, i=0) :
    train = df.sample(frac=2/3, random_state=i)
    test = df.drop(train.index)
    xtrain = train.drop(columns=['MEDV'])
    xtest = test.drop(columns=['MEDV'])
    ytrain = train['MEDV']
    ytest = test['MEDV']
    return xtrain, xtest, ytrain, ytest


# Basis Functions
# Vector of ones ie constant basis
def ones_basis (x) :
    return np.ones((len(x), 1))

def single_attr_basis (x, att_col) :
    ones = np.ones((len(x), 1))
    att = np.asarray(x.iloc[:, att_col])
    att = att.reshape(-1, 1)
    return np.column_stack((ones, att))

def all_attr_basis (x) :
    noAtt = x.shape[1]
    theta = np.ones((len(x), 1))
    for i in range (noAtt) :
        theta = np.column_stack((theta, single_attr_basis(x,i)[:, 1:]))
    return theta

# Linear Regression Functions
# Solves least squares equation, y = θw, returns weights w
def least_sqr_sol (theta, y) :
    # return np.linalg.inv(theta.T@theta)@theta.T@y
    w, *_ = np.linalg.lstsq(theta, y, rcond=None)
    return w


# Fits a single model degree and returns weights, predictions and MSE
def lin_reg_sol(x,y, basis_func = ones_basis, **kwargs) :
    # theta = poly_basis(x,k)
    theta = basis_func(x, **kwargs)
    # print('Theta = ')
    # print(theta)
    w = least_sqr_sol(theta,y)
    y_hat = theta @ w
    mse = np.mean((y - y_hat)**2)
    return [w, y_hat, mse]


# Runs r reps of model fitting, each with different train/test splits, returns the average MSE across runs (train and test)
def r_sims(df, r, basis_func=ones_basis, **kwargs) :
    results=[]
    for i in range(r):
        xtrain, xtest, ytrain, ytest = train_test_split(df, i)
        lm = lin_reg_sol(xtrain, ytrain, basis_func, **kwargs)
        mse_train = lm[2]
        w = lm[0]
        theta_test = basis_func(xtest, **kwargs)
        y_hat_test = theta_test @ w
        mse_test = np.mean((ytest - y_hat_test) ** 2)
        mse = [mse_train, mse_test]
        results.append(mse)
    avgmse = [0,0]

    for i in range(2):
        total = 0
        for j in range(r):
            total += results[j][i]
        avgmse[i] = total / r

    avgmse_df = pd.DataFrame({'avg_mse_train': [avgmse[0]], 'avg_mse_test': [avgmse[1]]})
    return avgmse_df, w

# Plots
def plot_crossVal_mse(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sig = sorted(df['sig'].unique())
    gam = sorted(df['gam'].unique())
    mse = np.zeros((len(gam), len(sig)))
    print(df['avg_mse_test_fold'])
    for i, g in enumerate(gam) :
        for j, s in enumerate(sig) :
            mse[i,j] = df.loc[(df['sig'] == s) & (df['gam']== g), 'avg_mse_test_fold'].values[0]

    s, g = np.meshgrid(sig, gam)
    ax.plot_surface(np.log2(s), np.log2(g), np.log(mse),
                    cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel('log₂(σ)')
    ax.set_ylabel('log₂(γ)')
    ax.set_zlabel('ln(MSE)')
    ax.set_title('Cross-validation error surface')
    ax.view_init(elev=30, azim=-30)
    return fig, ax


# Main Questions
print('Q4')
url = "http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv"
df = pd.read_csv(url)
print(df.head())

print('Q4.a')
runs = 20
avgmse_df, w = r_sims(df, runs)
print('Naive Regression')
print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")
print(' ')

print('Q4.b')
runs = 20
for i in range(df.shape[1]-1):
    avgmse_df, w = r_sims(df, runs, basis_func=single_attr_basis, att_col=i)
    print(f'Linear Regression with single attribute {i} : {df.columns[i]}')
    print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")
    print(' ')

print('Q4.c')
runs = 20
print('Linear Regression with all attributes')
avgmse_df, w = r_sims(df, runs, basis_func=all_attr_basis)
print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")
print(' ')


print('Q5')
def norm_p(x, p) :
    if p > 0 :
        x=np.asarray(x)
        return np.sum(np.abs(x)**p)**(1/p)
    else:
        return 'Norm not defined : p <= 0'

def guass_kern (x1, x2, sig) :
    diff = np.array(x1) - np.array(x2)
    num = norm_p(diff, 2)**2
    Kij = np.exp(-num/(2*(sig**2)))
    return Kij

def full_kern (x1, kernel_func, sig, x2=None,) :
    x1 = np.asarray(x1)
    n1 =len(x1)
    if x2 is None :
        x2 = x1
        n2 = n1
    else:
        x2 = np.asarray(x2)
        n2 = len(x2)
    K = np.zeros((n1,n2))
    for i in range(n1) :
        for j in range(n2) :
            K[i,j] = kernel_func(x1[i], x2[j], sig)
    return K

print('Q5.a')
gamma = [2**(-i) for i in range(40, 25, -1)]
sigma = [2**(7+(0.5*i)) for i in range(13)]


# Solves least squares equation, y = θw, returns weights w
def ker_ridg_sol (K, gamma, y) :
    l = len(y)
    # return np.linalg.inv(theta.T@theta)@theta.T@y
    alpha = np.linalg.solve((K+(gamma*l*np.identity(l))), y)
    return alpha

def ker_reg_sol(x,y, kernel_func, sig, gam) :
    # theta = poly_basis(x,k)
    K = full_kern(x, kernel_func=kernel_func, sig=sig)
    # K = kernel_func(x, **kwargs)
    # print('Theta = ')
    # print(theta)
    alpha = ker_ridg_sol(K, gam, y)
    y_hat = K @ alpha
    mse = np.mean((y - y_hat)**2)
    return [alpha, y_hat, mse]

# def cross_validation (xtrain, ytrain, folds, kernel_func, sig, gam) :
#
#     positions = np.arange(len(xtrain))
#     np.random.shuffle(positions)
#     foldPositions = np.array_split(positions, folds)
#     x, y = [], []
#     for i in foldPositions:
#         x.append(xtrain.iloc[i])
#         y.append(ytrain.iloc[i])
#
#     paraMatrix = []
#
#     for i in sig :
#         for j in gam :
#             paraMatrix.append([i,j])
#     cross_results = []
#
#     for pair in paraMatrix:
#         results = []
#         for i in range(folds) :
#             train_fold = [j for j in range(folds) if j!=i]
#             xtrain_fold = pd.concat([x[j] for j in train_fold])
#             ytrain_fold = pd.concat([y[j] for j in train_fold])
#             krm = ker_reg_sol(xtrain_fold, ytrain_fold, kernel_func, sig=pair[0], gam=pair[1])
#             mse_train = krm[2]
#             alpha = krm[0]
#             K_test = full_kern(x[i], kernel_func=kernel_func, sig=pair[0],  x2=xtrain_fold)
#             y_hat_test = K_test @ alpha
#             mse_test = np.mean((y[i] - y_hat_test) ** 2)
#             mse = [mse_train, mse_test]
#             results.append(mse)
#         avgmse = [0, 0]
#
#         for i in range(2):
#             total = 0
#             for j in range(folds):
#                 total += results[j][i]
#             avgmse[i] = total / folds
#
#         avgmse_df = pd.DataFrame({'pair': [pair], 'avg_mse_train': [avgmse[0]], 'avg_mse_test': [avgmse[1]]})
#         cross_results.append(avgmse_df)
#     cross_results = pd.concat(cross_results)
#     min_mse = cross_results.loc[cross_results['avg_mse_test'].idxmin()]
#
#     return list(min_mse['pair']), min_mse['avg_mse_train'], min_mse['avg_mse_test']

def cross_validation(xtrain, ytrain, folds, kernel_func, siglist, gamlist):
    positions = np.arange(len(xtrain))
    np.random.shuffle(positions)
    foldPositions = np.array_split(positions, folds)
    x, y = [], []
    for i in foldPositions:
        x.append(xtrain.iloc[i])
        y.append(ytrain.iloc[i])


    K_precal = {sig: full_kern(xtrain, kernel_func=kernel_func, sig=sig) for sig in siglist}


    paraMatrix = []

    for i in siglist:
        for j in gamlist:
            paraMatrix.append([i, j])
    cross_results = []

    for sig, gam in paraMatrix:
        K_full = K_precal[sig]
        results = []

        for i in range(folds):

            xtrain_fold = np.concatenate([foldPositions[j] for j in range(folds) if j != i])
            xtest_fold = foldPositions[i]


            K_train = K_full[np.ix_(xtrain_fold, xtrain_fold)]
            K_test = K_full[np.ix_(xtest_fold, xtrain_fold)]
            ytrain_fold = ytrain.iloc[xtrain_fold]
            ytest_fold = ytrain.iloc[xtest_fold]


            alpha = ker_ridg_sol(K_train, gam, ytrain_fold)
            y_hat_train = K_train @ alpha
            y_hat_test = K_test @ alpha
            mse_train_fold = np.mean((ytrain_fold - y_hat_train) ** 2)
            mse_test_fold = np.mean((ytest_fold - y_hat_test) ** 2)
            mse = [mse_train_fold, mse_test_fold]
            results.append(mse)
        avgmse = [0, 0]

        for i in range(2):
            total = 0
            for j in range(folds):
                total += results[j][i]
            avgmse[i] = total / folds

        avgmse_df = pd.DataFrame({'sig': [sig], 'gam' : [gam], 'avg_mse_train_fold': [avgmse[0]], 'avg_mse_test_fold': [avgmse[1]]})
        cross_results.append(avgmse_df)
    cross_results = pd.concat(cross_results)
    cross_results = cross_results.sort_values(by ='avg_mse_test_fold', ascending=True)
    min_mse = cross_results.iloc[0]
    return float(min_mse['sig']), float(min_mse['gam']), min_mse['avg_mse_train_fold'], min_mse['avg_mse_test_fold'], cross_results

# Runs r reps of model fitting, each with different train/test splits, returns the average MSE across runs (train and test)
def r_sims_ker_reg_sol(df, r, kernel_func, sig, gam, plot=False, **kwargs) :
    results=[]
    fig, ax = None, None
    for i in range(r):
        xtrain, xtest, ytrain, ytest = train_test_split(df, i)
        minsig, mingam, _, _, crossValdf= cross_validation(xtrain, ytrain, 5, kernel_func, sig, gam)
        print(f'For r = {i+1}, minsig = 2**{np.log2(minsig)}, mingam = 2**{np.log2(mingam)}')
        krm = ker_reg_sol(xtrain, ytrain, kernel_func, sig=minsig, gam=mingam)
        mse_train = krm[2]
        alpha = krm[0]
        K_test = full_kern(xtest, kernel_func=kernel_func, sig=minsig, x2=xtrain)
        y_hat_test = K_test @ alpha
        mse_test = np.mean((ytest - y_hat_test) ** 2)
        mse = [mse_train, mse_test]
        results.append(mse)
    avgmse = [0,0]

    for i in range(2):
        total = 0
        for j in range(r):
            total += results[j][i]
        avgmse[i] = total / r
    avgmse_df = pd.DataFrame({'avg_mse_train': [avgmse[0]], 'avg_mse_test': [avgmse[1]]})

    if plot == True and r>0:
        fig, ax = plot_crossVal_mse(crossValdf)
        return avgmse_df, alpha, fig, ax
    else:
        return avgmse_df, alpha


# k = full_kern(xtrain, kernel_func=guass_kern, sig=)
avgmse_df, alpha = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=sigma, gam=gamma)
print(avgmse_df)
# print(np.mean(np.abs(alpha)))
# avgmse_df, alpha = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=2**13, gam=2**-26)
# print(np.mean(np.abs(alpha)))


print('Q5.b')
avgmse_df, alpha, fig, ax = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=sigma, gam=gamma, plot=True)
print(avgmse_df)
plt.show()
plt.close()

plt.show()



