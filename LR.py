import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([1.5, 3.0, 2.5, 4.0, 5.0, 5.5, 6.5, 7.0, 8.5, 9.0])
n = len(X)

plt.scatter(X, Y, color='blue', label='Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot for Line')
plt.show()


#Normal Eqation 
x_mean = np.mean(X)
y_mean = np.mean(Y)

beta1 = np.sum((X - x_mean)*(Y - y_mean)) / np.sum((X - x_mean)**2)
beta0 = y_mean - beta1 * x_mean

print("Analytical Solution -> Intercept:", beta0, "Slope:", beta1)

Y_pred = beta0 + beta1 * X

plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, Y_pred, color='red', label='Regression line')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Linear Regression Fit (Normal Equation)")
plt.legend()
plt.show()


beta_0=0
beta_1=0
LR = 0.01
EPOCH = 10000
beta_h=[]

for epoch in range(EPOCH):
    y_pred = beta_0 + beta_1 * X
    d_beta_0 = (-2/n) * np.sum(Y - y_pred)
    d_beta_1 = (-2/n) * np.sum((Y - y_pred) * X)
    beta_0 = beta_0 - LR * d_beta_0
    beta_1 = beta_1 - LR * d_beta_1
    
    beta_h.append((beta_0, beta_1))
    
print("Gradient Descent Solution -> Intercept:", beta_0, "Slope:", beta_1)

fig, ax = plt.subplots()
ax.scatter(X, Y, color='blue', label='Data points')
line, = ax.plot([], [], color='red', label='Fitting Line')
ax.set_xlim(0, 11)
ax.set_ylim(0, 10)
ax.set_xlabel("Study Hours")
ax.set_ylabel("Score")
ax.set_title("Gradient Descent Linear Regression")

def animate(i):
    beta0_i, beta1_i = beta_h[i]
    Y_line = beta0_i + beta1_i * X
    line.set_data(X, Y_line)
    return line,

ani = FuncAnimation(fig, animate, frames=len(beta_h), interval=100, blit=True)
plt.legend()
plt.show()
