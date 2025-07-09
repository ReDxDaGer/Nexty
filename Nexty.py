import numpy as np 
import math


def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)

def mse(y_pred,y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

class NextyNet:

    def __init__(self,input_size,hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1,hidden_size))
        self.W2 = np.random.randn(hidden_size,1)
        self.b2 = np.zeros((1,1))
    
    def forward(self,x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self,x,y,output,lr=0.01):
        error = mse_derivative(output,y)
        dW2 = self.a1.T @ error
        db2 = np.sum(error, axis=0, keepdims=True)
        dz1 = error @ self.W2.T * relu_derivative(self.z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self,x,y,epochs=500,lr=0.01):
        for epoch in range(epochs):
            output = self.forward(x)
            loss = mse(output,y)
            self.backward(x,y,output,lr)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self,x):
        return self.forward(x)

X = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6],
              [5, 6, 7],
              [6, 7, 8]])
y = np.array([[4], [5], [6], [7], [8], [9]])


model = NextyNet(input_size=3, hidden_size=5)
model.train(X, y, epochs=1000, lr=0.01)

while True:
    user_input = input("\nEnter 3 numbers separated by space (q to quit): ")
    if user_input.lower() == 'q':
        break

    try:
        nums = list(map(float, user_input.strip().split()))
        if len(nums) != 3:
            print("Please enter exactly 3 numbers.")
            continue
        inp = np.array(nums).reshape(1, -1)
        prediction = model.predict(inp)
        print(f"Predicted next number: {prediction[0][0]:.2f}")
    except ValueError:
        print("Invalid input. Please enter numbers only.")
