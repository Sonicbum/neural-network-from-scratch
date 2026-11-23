import numpy as np
import matplotlib.pyplot as plt
#1st step
X = np.random.randint(-100, 100, size=(10,2)) #create a random array. We generate 10 lines and 2 columns
x1 = X[:, 0] #it means take all lines from the #0 column
x2 = X[:, 1]
Y = (x1+x2 > 0).astype(int)


W = np.zeros(2) #weights. We have 2 parameters (x1, x2) so we need 2 weights. z = (w1*x1)+(w2*x2) +b
b = 0.0
#2nd step
def sigmoid(z):
    return 1/(1 + np.exp(-z))

alpha=0.1#learning rate
epochs = 100 #how many times we run our examples through the network
for i in range(epochs):
    Z = np.dot(X, W) +b
    Y_hat = sigmoid(Z)

    loss = np.mean(-Y*np.log(Y_hat+1e-15) - (1-Y)*np.log(1-Y_hat+ 1e-15))#i added 1e-15 cause z is too large, then def sigmoid strictly rounds to either 0.0 or 1.0, but we can't take the log of 0

    #Gradient descent. 3rd step
    m = len(Y)
    dZ = Y_hat - Y
    dW = (1/m)*np.dot(X.T, dZ)
    db = (1/m)*np.sum(dZ)

    W = W - alpha*dW
    b = b - alpha*db
    
#4th step visualization 
plt.scatter(x1, x2, c=Y, cmap='coolwarm', edgecolors='black', s=100, label='Training Data')
x_line = np.array([np.min(x1), np.max(x1)])
y_line = (-W[0]*x_line - b) / W[1]
plt.plot(x_line, y_line, color='black', linewidth=2, linestyle='--', label='Decision Boundary')

plt.title("Training Set: How the Network Separated the Data")
plt.legend()
plt.grid(True)
plt.show()

#inference
n_X = np.array([[1, 5], 
                [56, -100], 
                [100, -200],
                [89, -45],
                [78, 2],
                [0, -6],
                [103, -200],
                [732, -657],
                [53, -488],
                [-25, -8]])
n_Z = np.dot(n_X, W) +b
n_Y_hat = sigmoid(n_Z)
asw = (n_Y_hat >= 0.5).astype(int)
print(asw)
n_x1 = n_X[:, 0]
n_x2 = n_X[:, 1]
plt.scatter(n_x1, n_x2, c=asw, cmap='coolwarm', edgecolors='black', s=400, label='New Test Points')
x_line = np.array([np.min(n_x1), np.max(n_x1)])
y_line = (-W[0]*x_line - b) / W[1]
plt.plot(x_line, y_line, color='black', linewidth=2, linestyle='--', label='Decision Boundary')

plt.title("Test Set: Inference on New Points")
plt.legend()
plt.grid(True)
plt.show()
print(asw)

    