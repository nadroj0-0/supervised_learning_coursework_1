import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import random
from scipy.stats import norm
import pandas as pd


# Basis Functions
# Return polynomial basis matrix with powers 0 to k-1
def poly_basis (x,k) :
    return np.column_stack([np.asarray(x) ** i for i in range(k)])

# Return sin basis matrix, sin(iπx), for i from 1 to k
def sin_basis (x,k) :
    return np.column_stack([np.sin((i+1)*np.pi*np.asarray(x)) for i in range(k)])


# Linear Regression Functions
# Solves least squares equation, y = θw, returns weights w
def lin_reg_sol (theta, y) :
    # return np.linalg.inv(theta.T@theta)@theta.T@y
    w, *_ = np.linalg.lstsq(theta, y, rcond=None)
    return w


# Fits a single polynomial model degree k and returns weights, predictions and MSE
def lin_reg_sol_k(x,y,k, basis_func=poly_basis) :
    # theta = poly_basis(x,k)
    theta = basis_func(x,k)
    print('Theta = ')
    print(theta)
    w = lin_reg_sol(theta,y)
    y_hat = theta @ w
    mse = np.mean((y - y_hat)**2)
    return [w, y_hat, mse]


# Fits multiple polynomial models of degree k for a defined set of k values, returns weights, predictions and MSE for each model
def fit_lin_reg(xdata, ydata, k_values, basis_func=poly_basis) :
    results = []
    for k in k_values:
        print('for k=', k)
        sol = lin_reg_sol_k(xdata, ydata, k, basis_func)
        print('w =', sol[0])
        print('y_hat = ', sol[1])
        print('mse = ', sol[2])
        print(' ')
        results.append((k, sol[0], sol[1], sol[2]))
    return results


# Runs r reps of model fitting for k degrees of model, on N samples, returns the average MSE across runs
def r_sims(N, r, k_values, basis_func=poly_basis) :
    results=[]
    for i in range(r):
        xdata, ydata = gen_data(N)
        lm = fit_lin_reg(xdata, ydata, k_values, basis_func)
        mse = []
        for j in lm:
            mse.append(j[3])

        results.append(mse)
    avgmse = [0] * len(k_values)

    for i in range(len(k_values)):
        total = 0
        for j in range(r) :
            total += results[j][i]
        avgmse[i] = total/r

    df = pd.DataFrame({'k' : k_values, 'avg_mse' : avgmse})
    return df


# Data Generation Functions
# True underlying function g(x) = sin²(2πx)
def g(x) :
    return np.sin(2 * np.pi *x) ** 2


# Generates N samples from g(x) with Guassian noise variance σ = 0.07
def gen_data(N) :
    ydata = []
    xdata = []
    for i in range(N) :
        rand = random.uniform(0,1)
        xdata.append(rand)
        rande = random.uniform(0,1)
        epsilon = norm.ppf(rande, loc=0, scale=0.07)
        ydata.append(g(rand) + epsilon)
    return xdata, ydata


# Plotting Functions
# Plots data points and fitted regression curves. Has the option to also plot the true function
def plot_models(xdata, ydata, results, xmin, xmax, ymin, ymax, colormap='tab10' , truFunc = None, basis_func=poly_basis, basis_name=None) :
    if basis_name is None:
        basis_name = basis_func.__name__.replace('_basis','').capitalize()

    x_points = np.linspace(xmin, xmax, 100)
    fig, ax = plt.subplots()

    ax.scatter(xdata, ydata, color='black', label='data', s=10)

    cmap = getattr(cm, colormap)
    colours = cmap(np.linspace(0.38, 1, len(results)))

    for (k, w, _, _), colour in zip(results, colours):
        theta_points = basis_func(x_points, k)
        y_points = theta_points @ w
        ax.plot(x_points, y_points, color=colour, label=f'k = {k}', linewidth=0.8)

    if truFunc is not None:
        ax.plot(x_points, truFunc(x_points), color='black', linewidth=1)
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
    ax.set_title(f'{basis_name} basis model of degree k')
    ax.legend()
    return fig, ax


# Plots ln(MSE) vs model degree k for sample size N and reps r
def plot_mse(k, mse, N, r, basis_func=poly_basis, basis_name=None) :
    if basis_name is None:
        basis_name = basis_func.__name__.replace('_basis','').capitalize()

    fig, ax = plt.subplots()

    ax.plot(k, mse, color='black', linewidth=0.8)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='inout', length=4)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(which='minor', length=2)

    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('ln(mse)')
    ax.set_title(f'Natural log of the training error vs the {basis_name.lower()} dimensions k, N={N}, r={r}')
    return fig, ax



# Main Questions
xdata = np.array([1, 2, 3, 4])
ydata = np.array([3, 2, 0, 5])
print(f'X = {xdata}')
print(f'y = {ydata}')
print(' ')

print('Q1')
k_values = [1,2,3,4]
lm = fit_lin_reg(xdata, ydata, k_values)
fig, ax = plot_models(xdata, ydata, lm, 0, 5, -5, 8, colormap='Blues')
print(' ')


print('Q2')
xdata, ydata = gen_data(30)
print(f'x = {xdata}')
print(f'y = {ydata}')

print('Q2.a.i')
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


print('Q2.a.ii')
k_values = [2,5,10,14,18]
lm = fit_lin_reg(xdata, ydata, k_values)
fig, ax = plot_models(xdata, ydata, lm, 0, 1, -0.5, 1.6, truFunc = g)


print('Q2.b-d')
def experiment_Q2_b_d (smallN, largeN, r, k_values, basis_func=poly_basis):
    print('(b) Single run, N =', smallN)
    xdata, ydata = gen_data(smallN)
    lm = fit_lin_reg(xdata, ydata, k_values, basis_func)
    df = pd.DataFrame(lm, columns=['k', 'w', 'y_hat', 'mse'])
    fig, ax = plot_mse(df['k'], np.log(df['mse']), smallN, 1, basis_func=basis_func)

    print('(c) Single run, N =', largeN)
    xdata, ydata = gen_data(largeN)
    lm = fit_lin_reg(xdata, ydata, k_values, basis_func)
    df = pd.DataFrame(lm, columns=['k', 'w', 'y_hat', 'mse'])
    fig, ax = plot_mse(df['k'], np.log(df['mse']), largeN, 1, basis_func=basis_func)

    print('(d) Average results over r runs')
    df = r_sims(smallN, r, k_values, basis_func)
    fig, ax = plot_mse(df['k'], np.log(df['avg_mse']), smallN, r, basis_func=basis_func)
    df = r_sims(largeN, r, k_values, basis_func)
    fig, ax = plot_mse(df['k'], np.log(df['avg_mse']), largeN, r, basis_func=basis_func)

    return

k_values = [i for i in range(1,19)]
experiment_Q2_b_d(30, 1000, 100, k_values)


print('Q3')
k_values = [i for i in range(1, 19)]
experiment_Q2_b_d(30, 1000, 100, k_values, basis_func=sin_basis)
plt.show()
plt.close('all')

print('Q4')
url = "http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv"
df = pd.read_csv(url)
print(df.head())