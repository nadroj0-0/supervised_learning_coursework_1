import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

xdata = np.array([1, 2, 3, 4])
ydata = np.array([3, 2, 0, 5])
print(f'X = {xdata}')
print(f'y = {ydata}')


def poly_basis (x,k) :
    return np.column_stack([np.asarray(x) ** i for i in range(k)])

def lin_reg_sol (theta, y) :
    return np.linalg.inv(theta.T@theta)@theta.T@y

def lin_reg_sol_k(x,y,k) :
    theta = poly_basis(x,k)
    print('Theta = ')
    print(theta)
    w = lin_reg_sol(theta,y)
    y_hat = theta @ w
    mse = np.mean((y - y_hat)**2)
    return [w, y_hat, mse]

def fit_lin_reg(xdata, ydata, k_values, ) :
    results = []
    for k in k_values:
        print('for k=', k)
        sol = lin_reg_sol_k(xdata, ydata, k)
        print('w =', sol[0])
        print('y_hat = ', sol[1])
        print('mse = ', sol[2])
        print(' ')
        results.append((k, sol[0], sol[1], sol[2]))
    return results

def plot_models(xdata, ydata, results, xmin, xmax, ymin, ymax, colormap='tab10' ) :
    x_points = np.linspace(xmin, xmax, 100)
    fig, ax = plt.subplots()

    ax.scatter(xdata, ydata, color='black', label='data', s=10)

    cmap = getattr(cm, colormap)
    colours = cmap(np.linspace(0.38, 1, len(results)))

    for (k, w, _, _), colour in zip(results, colours):
        theta_points = poly_basis(x_points, k)
        y_points = theta_points @ w
        ax.plot(x_points, y_points, color=colour, label=f'k = {k}', linewidth=0.8)

    ax.set_ylim(ymin, ymax)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='inout', length=4)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(which='minor', length=2)

    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Polynomial basis model of degree k')
    ax.legend()
    return fig, ax

print(' ')
print('Q1.1')
#w_list=[]
k_values = [1,2,3,4]

lm = fit_lin_reg(xdata, ydata, k_values)
fig, ax = plot_models(xdata, ydata, lm, 0, 5, -5, 8, colormap='Blues')

#for k in k_values :
#    print('for k=',k)
#    sol = lin_reg_sol_k(xdata,ydata,k)
#    w_list.append(sol[0])
#    print('w =',sol[0])
#    print('y_hat = ',sol[1])
#    print('mse = ',sol[2])
#    print(' ')

print(' ')

#x_points = np.linspace(0,5,100)
#fig, ax = plt.subplots()

#ax.scatter(xdata, ydata, color='black', label='data', s=10)

# colours = cm.Blues(np.linspace(0.38, 1, len(w_list)))
#
# for k, w, colour in zip(k_values, w_list, colours) :
#     theta_points = poly_basis(x_points,k)
#     y_points = theta_points @ w
#     ax.plot(x_points,y_points, color = colour,label=f'k = {k}', linewidth=0.8)
#
# ax.set_ylim(-5, 8)
#
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.tick_params(axis='both', direction='inout', length=4)
#
# ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
# ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
# ax.tick_params(which='minor', length=2)
#
# ax.grid(True, linestyle=':', linewidth=0.5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Polynomial basis model of degree k')
# ax.legend()

#plt.show()
#plt.close('all')

print('Q1.2')
import random
from scipy.stats import norm

def g(x) :
    return np.sin(2 * np.pi *x) ** 2

ydata = []
xdata = []
for i in range(30) :
    rand = random.uniform(0,1)
    xdata.append(rand)
    rande = random.uniform(0,1)
    epsilon = norm.ppf(rande, loc=0, scale=0.07)
    ydata.append(g(rand) + epsilon)

print(f'x = {xdata}')
print(f'y = {ydata}')

x_points = np.linspace(0,1,100)
fig, ax = plt.subplots()

ax.scatter(xdata, ydata, color='black', label='data', s=10)

ax.plot(x_points,g(x_points), color = 'black' ,label='Sin^2(2pix)', linewidth=0.8)

ax.set_ylim(-0.1, 1.1)

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='both', direction='inout', length=4)

ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.tick_params(which='minor', length=2)

ax.grid(True, linestyle=':', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(' ')
ax.legend()

#plt.show()
#plt.close('all')


w_list = []
k_values = [2,5,10,14,18]
lm = fit_lin_reg(xdata, ydata, k_values)
fig, ax = plot_models(xdata, ydata, lm, 0, 1, -0.1, 1.1)
plt.show()
# for k in k_values :
#     print('for k=',k)
#     sol = lin_reg_sol_k(xdata,ydata,k)
#     w_list.append(sol[0])
#     print('w =',sol[0])
#     print('y_hat = ',sol[1])
#     print('mse = ',sol[2])
#     print(' ')

# x_points = np.linspace(0,1,100)
# fig, ax = plt.subplots()
#
# ax.scatter(xdata, ydata, color='black', label='data', s=10)
#
# colours = cm.tab10(np.linspace(0.38, 1, len(w_list)))
#
# for k ,w, colour in zip(k_values, w_list, colours) :
#     theta_points = poly_basis(x_points,k)
#     y_points = theta_points @ w
#     ax.plot(x_points,y_points, color = colour,label=f'k = {k}', linewidth=0.8)
#
# ax.set_ylim(-0.1, 1.1)
#
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.tick_params(axis='both', direction='inout', length=4)
#
# ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
# ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
# ax.tick_params(which='minor', length=2)
#
# ax.grid(True, linestyle=':', linewidth=0.5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Polynomial basis model of degree k')
# ax.legend()

plt.show()
plt.close('all')

























import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import random
from scipy.stats import norm
import pandas as pd

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
def lin_reg_sol(x,y, basis_func = ones_basis) :
    # theta = poly_basis(x,k)
    theta = basis_func(x)
    # print('Theta = ')
    # print(theta)
    w = least_sqr_sol(theta,y)
    y_hat = theta @ w
    mse = np.mean((y - y_hat)**2)
    return [w, y_hat, mse]


# Runs r reps of model fitting, each with different train/test splits, returns the average MSE across runs (train and test)
def r_sims(df, r, basis_func=ones_basis) :
    results=[]
    for i in range(r):
        xtrain, xtest, ytrain, ytest = train_test_split(df, i)
        lm = lin_reg_sol(xtrain, ytrain, basis_func)
        mse_train = lm[2]
        w = lm[0]
        theta_test = basis_func(xtest)
        y_hat_test = theta_test @ w
        mse_test = np.mean((ytest - y_hat_test) ** 2)
        mse = [mse_train, mse_test]
    return






















import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import random
from scipy.stats import norm
import pandas as pd

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

def full_kern (x1, kernel_func, x2=None, **kwargs) :
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
            K[i,j] = kernel_func(x1[i], x2[j], **kwargs)
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

def ker_reg_sol(x,y, kernel_func, **kwargs) :
    # theta = poly_basis(x,k)
    K = full_kern(x, kernel_func=kernel_func, sig=kwargs['sig'])
    # K = kernel_func(x, **kwargs)
    # print('Theta = ')
    # print(theta)
    alpha = ker_ridg_sol(K, kwargs['gam'], y)
    y_hat = K @ alpha
    mse = np.mean((y - y_hat)**2)
    return [alpha, y_hat, mse]

def cross_validation (xtrain, ytrain, folds, kernel_func, para) :
    # x = [0 for i in range(folds)]
    # y = [0 for i in range(folds)]
    # for i in range(folds):
    #     x[i] = xtrain.iloc[i * (len(xtrain)//folds): (i + 1) * (len(xtrain)//folds)]
    #     y[i] = ytrain.iloc[i * (len(xtrain)//folds): (i + 1) * (len(xtrain)//folds)]

    positions = np.arrange(len(xtrain))
    np.random.shuffle(positions)
    foldPositions = np.array_split(positions, folds)
    x, y = [], []
    for i in foldPositions:
        x.append(xtrain.iloc[i])
        y.append(ytrain.iloc[i])

    paraMatrix = []

    for i in para['sig'] :
        for j in para['gam'] :
            paraMatrix.append((i,j))
    cross_results = []
    for pair in paraMatrix:
        results = []
        for i in range(folds) :
            train_fold = [j for j in range(folds) if j!=i]
            xtrain_fold = pd.concat([x[j] for j in train_fold])
            ytrain_fold = pd.concat([y[j] for j in train_fold])
            krm = ker_reg_sol(xtrain_fold, ytrain_fold, kernel_func, sig=pair[0], gam=pair[1])
            mse_train = krm[2]
            alpha = krm[0]
            K_test = full_kern(x[i], kernel_func=kernel_func, x2=xtrain_fold, sig=pair[0])
            y_hat_test = K_test @ alpha
            mse_test = np.mean((y[i] - y_hat_test) ** 2)
            mse = [mse_train, mse_test]
            results.append(mse)
        avgmse = [0, 0]

        for i in range(2):
            total = 0
            for j in range(folds):
                total += results[j][i]
            avgmse[i] = total / folds

        avgmse_df = pd.DataFrame({'pair': pair, 'avg_mse_train': [avgmse[0]], 'avg_mse_test': [avgmse[1]]})
        cross_results.append(avgmse_df)
    cross_results = pd.concat(cross_results)
    min_mse = cross_results.loc[cross_results['avg_mse_test'].idxmin()]

    return min_mse['pair'], min_mse['avg_mse_train'], min_mse['avg_mse_test']

# Runs r reps of model fitting, each with different train/test splits, returns the average MSE across runs (train and test)
def r_sims_ker_reg_sol(df, r, kernel_func, **kwargs) :
    results=[]
    for i in range(r):
        xtrain, xtest, ytrain, ytest = train_test_split(df, i)
        pair, _, _ = cross_validation(xtrain, ytrain, 5, kernel_func, **kwargs)
        krm = ker_reg_sol(xtrain, ytrain, kernel_func, sig=pair[0], gam=pair[1])
        mse_train = krm[2]
        alpha = krm[0]
        K_test = full_kern(xtest, kernel_func=kernel_func, x2=xtrain, sig=pair[0])
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
    return avgmse_df, alpha

# k = full_kern(xtrain, kernel_func=guass_kern, sig=)
avgmse_df, alpha = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=sigma, gam=gamma)
print(np.mean(np.abs(alpha)))
# avgmse_df, alpha = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=2**13, gam=2**-26)
# print(np.mean(np.abs(alpha)))