import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

xdata = np.array([1, 2, 3, 4])
ydata = np.array([3, 2, 0, 5])
print(f'X = {xdata}')
print(f'y = {ydata}')


def poly_basis (x,k) :
    return np.column_stack([x ** i for i in range(k)])

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

print(' ')
print('Q1.1')
w_list=[]
for i in range (4) :
    print('for k=',i+1)
    sol = lin_reg_sol_k(xdata,ydata,i+1)
    w_list.append(sol[0])
    print('w =',sol[0])
    print('y_hat = ',sol[1])
    print('mse = ',sol[2])
    print(' ')

print(' ')

x_points = np.linspace(0,5,100)
fig, ax = plt.subplots()

ax.scatter(xdata, ydata, color='black', label='data', s=10)

colours = cm.Blues(np.linspace(0.38, 1, len(w_list)))

for i, (w, colour) in enumerate(zip(w_list, colours)) :
    theta_points = poly_basis(x_points,i+1)
    y_points = theta_points @ w
    ax.plot(x_points,y_points, color = colour,label=f'k = {i+1}', linewidth=0.8)

ax.set_ylim(-5, 8)

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

plt.show()

print('Q1.2')



