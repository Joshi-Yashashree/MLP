import numpy as np

def relu(x):
    """ReLU activation: keeps positives, zeros out negatives."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU: 1 for x>0, else 0."""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation: squashes input into (0,1)."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid, useful in backprop."""
    s = sigmoid(x)
    return s * (1 - s)


def forward(x, W1, b1, W2, b2):
    """
    Perform one forward pass through the network.
    Input -> Hidden layer (ReLU) -> Output layer (Sigmoid).
    """
    # Hidden layer
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)

    # Output layer
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    return z1, a1, z2, y_hat

def train(X, y, hidden_size=3, epochs=1000, lr=0.01):
    """
    Train a simple neural net with 1 hidden layer.
    - X: input data
    - y: labels
    - hidden_size: neurons in hidden layer
    - epochs: training iterations
    - lr: learning rate
    """
    n_samples, n_features = X.shape

    # Random weight initialization
    W1 = np.random.randn(n_features, hidden_size) * 0.1
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size) * 0.1
    b2 = 0.0

    for epoch in range(epochs):
        loss = 0

        # Loop over all samples
        for i in range(n_samples):
            x = X[i]
            target = y[i]

            # Forward pass 
            z1, a1, z2, y_hat = forward(x, W1, b1, W2, b2)

            # Compute loss (Binary Cross-Entropy)
            loss += -(target * np.log(y_hat + 1e-9) +
                      (1 - target) * np.log(1 - y_hat + 1e-9))

            # Backpropagation 
            dz2 = y_hat - target          # error at output
            dW2 = a1 * dz2                # gradient for W2
            db2 = dz2                     # gradient for b2

            dz1 = (W2 * dz2) * relu_derivative(z1)  # backprop into hidden
            dW1 = np.outer(x, dz1)        # gradient for W1
            db1 = dz1                     # gradient for b1

            # Update weights 
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {loss/n_samples:.4f}")

    return W1, b1, W2, b2


np.random.seed(42)


durations = np.random.randint(1, 10, 200)  # 1–10 hours
tips = np.random.randint(0, 20, 200)       # 0–20 currency units


labels = ((durations > 5) & (tips > 10)).astype(int)


X = np.column_stack((durations, tips))
y = labels



W1, b1, W2, b2 = train(X, y, hidden_size=3, epochs=1000, lr=0.01)


# Interactive testing

while True:
    # Take user input
    d = float(input("Enter duration stayed (hours): "))
    t = float(input("Enter tip given: "))

    # Forward pass on new input
    z1, a1, z2, pred = forward(np.array([d, t]), W1, b1, W2, b2)

    # Show details
    print("\n--- Forward Pass Details ---")
    print("Hidden layer sums (z1):", z1)
    print("Hidden activations (a1):", a1)
    print("Output sum (z2):", z2)
    print("Predicted satisfaction probability =", float(pred))
    print("Prediction:", "Satisfied" if pred > 0.5 else "Not satisfied")
    print("-----------------------------\n")

    # Ask to continue
    again = input("Test another? (y/n): ")
    if again.lower() != "y":
        break

