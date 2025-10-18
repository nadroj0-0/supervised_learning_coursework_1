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