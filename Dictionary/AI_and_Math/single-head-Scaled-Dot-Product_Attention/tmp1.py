import numpy as np
from scipy.special import softmax

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Dot product of Q and Kᵀ
    matmul_qk = np.dot(Q, K.T)

    # Step 2: Scale the scores
    d = K.shape[1]  # Key dimension
    scaled_scores = matmul_qk / np.sqrt(d)

    # Step 3: Apply softmax to get attention weights
    attention_weights = softmax(scaled_scores, axis=1)

    # Step 4: Multiply attention weights by V
    output = np.dot(attention_weights, V)

    # Display results
    print("Attention Weights:")
    print(attention_weights)
    print("\nOutput:")
    print(output)

# Inputs
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

# Run the function
scaled_dot_product_attention(Q, K, V)