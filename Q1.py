import numpy as np

X = np.array([1, 2, 3, 4])
y = np.array([3, 2, 0, 5])
print(f'X = {X}')
print(f'y = {y}')


def polyBasis (X,k) :
    return np.column_stack([X ** i for i in range(k)])

def lin_reg_sol (Theta, y) :
    sol = np.linalg.inv(Theta.T@Theta)@Theta.T@y
    return sol
#print(polyBasis(X,3))
#Theta=polyBasis(X,3)
#w = lin_reg_sol(Theta,y)
#print(w)
#s= Theta @ w
#print(s)

def lin_reg_sol_k(X,y,k) :
    theta = polyBasis(X,k)
    print('Theta = ')
    print(theta)
    w = lin_reg_sol(theta,y)
    #print('this is w ', w)
    #print('dim', np.ndim(w))
    y_hat = theta @ w
    #print('y_hat', y_hat)
    mse = np.mean((y - y_hat)**2)
    return [w, y_hat, mse]

print(' ')
print('Q1.1')
for i in range (4) :
    print('for k=',i+1)
    sol = lin_reg_sol_k(X,y,i+1)
    print('w =',sol[0])
    print('y_hat = ',sol[1])
    print('sse = ',sol[2])
    print(' ')

print(' ')
print('Q1.2')



