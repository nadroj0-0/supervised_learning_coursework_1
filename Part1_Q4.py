import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import random
from scipy.stats import norm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import json
import os


# Saving results and figures functions
#Saves a figure in Figures folder
def save_figure(fig, filename: str):
    path = os.path.join("outputs", "figures", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")

#Saves a json file in results folder
def save_json(data: dict, filename: str):
    path = os.path.join("outputs", "results", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

#Saves a latex file in the tex folder for writing up in latex
def df_to_tex(df, filename: str):
    path = os.path.join("outputs", "tex", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    df.to_latex(path, index=False, float_format="%.4f", escape=False)

def json_to_tex(data: dict, filename: str):
    rows = []
    for k, v in data.items():
        rows.append([k, v])

    df = pd.DataFrame(rows)
    path = os.path.join("outputs", "tex", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_latex(path, index=False, header=False, float_format="%.4f", escape=False)




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
def r_sims(df, r, basis_func=ones_basis, splits=None,  **kwargs) :
    results=[]
    for i in range(r):
        if splits is None :
            xtrain, xtest, ytrain, ytest = train_test_split(df, i)
        else :
            xtrain, xtest, ytrain, ytest = splits[i]
        lm = lin_reg_sol(xtrain, ytrain, basis_func, **kwargs)
        mse_train = lm[2]
        w = lm[0]
        theta_test = basis_func(xtest, **kwargs)
        y_hat_test = theta_test @ w
        mse_test = np.mean((ytest - y_hat_test) ** 2)
        mse = [mse_train, mse_test]
        results.append(mse)
    avgmse = [0,0]
    avgmse_sd = [0,0]

    for i in range(2):
        total = 0
        for j in range(r):
            total += results[j][i]
        avgmse[i] = total / r

    if r > 1 :
        for i in range(2):
            total = 0
            for j in range(r):
                total += (results[j][i]-avgmse[i])**2
            avgmse_sd[i] = total / (r-1)
            avgmse_sd[i] = avgmse_sd[i]**(1/2)
    else :
        avgmse_sd = [None,None]

    avgmse_df = pd.DataFrame({'avg_mse_train': [avgmse[0]], 'avg_mse_test': [avgmse[1]], 'avg_mse_train_sd': [avgmse_sd[0]], 'avg_mse_test_sd': [avgmse_sd[1]]})
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

# === Auto-generate LaTeX table for Q4(c) ===
def generate_single_attr_table_tex(results_dict, filename="Q4/single_attr.tex"):
    rows = []
    for idx, (attr, vals) in enumerate(results_dict.items()):
        train_mean, train_std, test_mean, test_std, w0, wi = vals
        rows.append([
            f"({idx}) {attr}",
            f"{train_mean:.5f} ± {train_std:.5f}",
            f"{test_mean:.5f} ± {test_std:.5f}",
            f"{w0:.5f}",
            f"{wi:.5f}"
        ])

    df = pd.DataFrame(rows, columns=[
        "Attribute",
        "Train MSE",
        "Test MSE",
        "$w_{0}$",
        "$w_{1}$"
    ])

    df_to_tex(df, filename)


def generate_full_attr_weight_table_tex(weights, attributes, filename="Q4/full_attr_weights.tex"):
    rows = []

    # w0 separately
    rows.append(["$w_0$ (bias)", f"{weights[0]:.5f}"])

    # remaining weights paired with attribute name
    for i in range(1, len(weights)):
        rows.append([f"$w_{{{i}}}$ ({attributes[i - 1]})", f"{weights[i]:.5f}"])

    df = pd.DataFrame(rows, columns=["Weight", "Value"])
    df_to_tex(df, filename)


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

# === Q4.a JSON ===
save_json({
    "method": "naive_regression",
    "train_mean": avgmse_df['avg_mse_train'][0],
    "train_std": avgmse_df['avg_mse_train_sd'][0],
    "test_mean": avgmse_df['avg_mse_test'][0],
    "test_std": avgmse_df['avg_mse_test_sd'][0],
    "weights": w.tolist()
}, "Q4/naive.json")

print(' ')

print('Q4.c')
runs = 20
single_attr_results = {}
for i in range(df.shape[1]-1):
    avgmse_df, w = r_sims(df, runs, basis_func=single_attr_basis, att_col=i)
    print(f'Linear Regression with single attribute {i} : {df.columns[i]}')
    print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")

    # store data for tex generation
    single_attr_results[df.columns[i]] = (
        avgmse_df['avg_mse_train'][0],
        avgmse_df['avg_mse_train_sd'][0],
        avgmse_df['avg_mse_test'][0],
        avgmse_df['avg_mse_test_sd'][0],
        w[0],
        w[1]# <-- add weights
    )

    save_json({
        "attribute": int(i),
        "train_mean": avgmse_df['avg_mse_train'][0],
        "train_std": avgmse_df['avg_mse_train_sd'][0],
        "test_mean": avgmse_df['avg_mse_test'][0],
        "test_std": avgmse_df['avg_mse_test_sd'][0],
        "weights": w.tolist()
    }, f"Q4/attribute_{i}.json")

    print(' ')

# Write table
generate_single_attr_table_tex(single_attr_results)

print('Q4.d')
runs = 20
print('Linear Regression with all attributes')
avgmse_df, w = r_sims(df, runs, basis_func=all_attr_basis)
print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")

generate_full_attr_weight_table_tex(w, df.columns[:-1])


# Save JSON for Q4(d)
save_json({
    "method": "all_attributes",
    "train_mean": avgmse_df['avg_mse_train'][0],
    "train_std": avgmse_df['avg_mse_train_sd'][0],
    "test_mean": avgmse_df['avg_mse_test'][0],
    "test_std": avgmse_df['avg_mse_test_sd'][0],
    "weights": w.tolist()
}, "Q4/all_attributes.json")

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
def r_sims_ker_reg_sol(df, r, kernel_func, sig, gam, splits=None,  plot=False, **kwargs) :
    results=[]
    fig, ax = None, None

    best_params = []

    for i in range(r):
        if splits is None:
            xtrain, xtest, ytrain, ytest = train_test_split(df, i)
        else:
            xtrain, xtest, ytrain, ytest = splits[i]
        minsig, mingam, _, _, crossValdf= cross_validation(xtrain, ytrain, 5, kernel_func, sig, gam)

        best_params.append({
            "run": int(i),
            "best_sigma": float(minsig),
            "best_gamma": float(mingam)
        })

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
    avgmse_sd = [0,0]

    for i in range(2):
        total = 0
        for j in range(r):
            total += results[j][i]
        avgmse[i] = total / r

    if r > 1 :
        for i in range(2):
            total = 0
            for j in range(r):
                total += (results[j][i]-avgmse[i])**2
            avgmse_sd[i] = total / (r-1)
            avgmse_sd[i] = avgmse_sd[i]**(1/2)
    else :
        avgmse_sd = [None,None]


    avgmse_df = pd.DataFrame({'avg_mse_train': [avgmse[0]], 'avg_mse_test': [avgmse[1]], 'avg_mse_train_sd': [avgmse_sd[0]], 'avg_mse_test_sd': [avgmse_sd[1]]})

    # Save aggregated best params (Q5a) after all runs
    if r > 1:  # only when doing repeated runs
        save_json({"runs": best_params}, "Q5/best_params.json")

    if plot == True and r>0:
        fig, ax = plot_crossVal_mse(crossValdf)
        return avgmse_df, alpha, fig, ax, minsig, mingam, crossValdf
    else:
        return avgmse_df, alpha, minsig, mingam


# k = full_kern(xtrain, kernel_func=guass_kern, sig=)
#avgmse_df, alpha, _, _ = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=sigma, gam=gamma)
#print(avgmse_df)
# print(np.mean(np.abs(alpha)))
# avgmse_df, alpha = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=2**13, gam=2**-26)
# print(np.mean(np.abs(alpha)))

avgmse_df, alpha, fig, ax, minsig1, mingam1, crossVal = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=sigma, gam=gamma, plot=True)
print(avgmse_df)

# compute exponents
exp_sig = np.log2(minsig1)
exp_gam = np.log2(mingam1)

# Save JSON for record
save_json({
    "sigma_pow2": float(exp_sig),
    "gamma_pow2": float(exp_gam),
    "train_mse": float(avgmse_df['avg_mse_train'][0]),
    "test_mse": float(avgmse_df['avg_mse_test'][0])
}, "Q5/Q5a.json")

# Export TeX-friendly snippet
json_to_tex({
    r"$\sigma$": fr"$2^{{{exp_sig:.1f}}}$",
    r"$\gamma$": fr"$2^{{{exp_gam:.1f}}}$",
    "Train MSE": f"{avgmse_df['avg_mse_train'][0]:.5f}",
    "Test MSE": f"{avgmse_df['avg_mse_test'][0]:.5f}"
}, "Q5/Q5a.tex")


print('Q5.b')
#avgmse_df, alpha, fig, ax, minsig1, mingam1 = r_sims_ker_reg_sol(df, 1, kernel_func=guass_kern, sig=sigma, gam=gamma, plot=True)

save_figure(fig, "Q5/crossval_surface.png")
save_json(crossVal.to_dict(orient="records"), "Q5/Q5b_surface.json")
df_to_tex(crossVal, "Q5/Q5b_surface.tex")


print('5.c')
# Save JSON for Q5(c)
save_json({
    "method": "kernel_ridge_regression_single_run",
    "train_mse": avgmse_df['avg_mse_train'][0],
    "test_mse": avgmse_df['avg_mse_test'][0],
    "best_sigma": float(minsig1) if 'minsig1' in locals() else None,
    "best_gamma": float(mingam1) if 'mingam1' in locals() else None
}, "Q5/krr_single_run.json")

json_to_tex({
    "Train MSE": f"{avgmse_df['avg_mse_train'][0]:.5f}",
    "Test MSE": f"{avgmse_df['avg_mse_test'][0]:.5f}"
}, "Q5/Q5c.tex")

print('Q5.d')
mse_table = pd.DataFrame(columns=['Method', 'MSE train', 'MSE test'])

# 4.a
runs = 20
rsplits = []
for i in range(runs):
    rsplits.append(train_test_split(df, i))
avgmse_df, w = r_sims(df, runs, splits=rsplits)
print('Naive Regression')
print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")
print(' ')
mse_table.loc[len(mse_table)] = {'Method' : 'Naive Regression', 'MSE train' : f"{avgmse_df['avg_mse_train'][0]}±{avgmse_df['avg_mse_train_sd'][0]}", 'MSE test' : f"{avgmse_df['avg_mse_test'][0]}±{avgmse_df['avg_mse_test_sd'][0]}"}

 # 4.c
runs = 20
for i in range(df.shape[1]-1):
    avgmse_df, w = r_sims(df, runs, basis_func=single_attr_basis, splits=rsplits, att_col=i)
    print(f'Linear Regression with single attribute {i} : {df.columns[i]}')
    print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")
    print(' ')
    mse_table.loc[len(mse_table)] = {'Method': f'Linear Regression (attribute {i})', 'MSE train': f"{avgmse_df['avg_mse_train'][0]}±{avgmse_df['avg_mse_train_sd'][0]}", 'MSE test': f"{avgmse_df['avg_mse_test'][0]}±{avgmse_df['avg_mse_test_sd'][0]}"}

# 4.d
runs = 20
print('Linear Regression with all attributes')
avgmse_df, w = r_sims(df, runs, basis_func=all_attr_basis, splits= rsplits)
print(f"avg mse train {avgmse_df['avg_mse_train'][0]}, avg mse test {avgmse_df['avg_mse_test'][0]}, weight vector {w}")
print(' ')
mse_table.loc[len(mse_table)] = {'Method': 'Linear Regression (all attributes)', 'MSE train': f"{avgmse_df['avg_mse_train'][0]}±{avgmse_df['avg_mse_train_sd'][0]}", 'MSE test': f"{avgmse_df['avg_mse_test'][0]}±{avgmse_df['avg_mse_test_sd'][0]}"}


# 5.a,c
runs = 20
gamma = [2**(-i) for i in range(40, 25, -1)]
sigma = [2**(7+(0.5*i)) for i in range(13)]

avgmse_df, alpha, _, _ = r_sims_ker_reg_sol(df, runs, kernel_func=guass_kern, sig=sigma, gam=gamma, splits=rsplits)
print(avgmse_df)
mse_table.loc[len(mse_table)] = {'Method': 'Kernel Ridge Regression', 'MSE train': f"{avgmse_df['avg_mse_train'][0]}±{avgmse_df['avg_mse_train_sd'][0]}", 'MSE test': f"{avgmse_df['avg_mse_test'][0]}±{avgmse_df['avg_mse_test_sd'][0]}"}

# Save JSON for Q5(c)
save_json({
    "method": "kernel_ridge_regression",
    "train_mean": avgmse_df['avg_mse_train'][0],
    "train_std": avgmse_df['avg_mse_train_sd'][0],
    "test_mean": avgmse_df['avg_mse_test'][0],
    "test_std": avgmse_df['avg_mse_test_sd'][0]
}, "Q5/krr_mse.json")

print (mse_table)
plt.show()
plt.close()

# Save JSON for Q5(d) combined comparison table
comparison = []
for _, row in mse_table.iterrows():
    comparison.append({
        "method": row["Method"],
        "mse_train": row["MSE train"],
        "mse_test": row["MSE test"]
    })

save_json({"methods": comparison}, "Q5/summary.json")

# Export TeX for combined table
df_tex = mse_table.rename(columns={
    "Method": "method",
    "MSE train": "mse_train",
    "MSE test": "mse_test"
})

# ROUNDING (presentation only)
for col in ["mse_train", "mse_test"]:
    df_tex[col] = df_tex[col].apply(
        lambda x: f"{float(x.split('±')[0]):.5g} ± {float(x.split('±')[1]):.5g}"
    )

# Rename AFTER formatting
df_tex.rename(columns={
    "method": "Method",
    "mse_train": "Training MSE",
    "mse_test": "Test MSE"
}, inplace=True)


df_to_tex(df_tex, "Q5/summary.tex")






