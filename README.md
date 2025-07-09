# 📘 NextyNet - A Minimal Neural Network to Predict the Next Number

Welcome to NextyNet — a tiny but powerful nural network built using only NumPy. This project is ideal for learning how forward propagation, backpropagation, and loss functions work in real neural networks — all without hiding behind high-level libraries like PyTorch or TensorFlow.

## 🚀 What This Project Does

Given a small sequence of 3 numbers like [1, 2, 3], the model learns to predict the next number in the pattern — for example: **4**.

## ✅ Example:
    Input:  [4, 5, 6]
    Output: 7.00

You can train the model once, then dynamically enter any 3-number sequence during runtime to get predictions.

-------------------------------------------

## File Overview
Nexty.py contains the entire model:

- Custom implementation of a 2-layer neural network

- ReLU activation

- Mean Squared Error (MSE) loss function

- Manual backpropagation

- Interactive user input

## How It Works

- Custom implementation of a 2-layer neural network

- ReLU activation

- Mean Squared Error (MSE) loss function

- Manual backpropagation

- Interactive user input

## Training
- 6 input-output pairs (e.g., [1, 2, 3] -> 4)

- Trained over 1000 epochs

- Loss printed every 100 epochs

## 🤔 Why ReLU Instead of Sigmoid or Tanh?

We use ReLU here because:

- It's simple and computationally fast

- Avoids gradient vanishing for positive inputs

- Performs better in most real-world models

Even NanoGPT and modern transformers use ReLU (or variants like GELU).