import numpy as np
from numpy.linalg import eig

def calculate_geometric_mean(matrix):
    num_rows, num_columns = matrix.shape
    
    weights = np.product(matrix, axis=1) ** (1 / num_columns)
    normalized_weights = weights / np.sum(weights)
    
    return normalized_weights

def calculate_consistency_index(matrix):
    eigenvalues, _ = eig(matrix)
    max_eigenvalue = np.max(eigenvalues.real)
    n = matrix.shape[0]
    consistency_index = (max_eigenvalue - n) / (n - 1)
    return consistency_index

pairwise_matrix = np.array([
    [1, 3, 3, 9, 9, 9, 9, 9],
    [1/3, 1, 2, 8, 3, 8, 9, 9],
   [1/3, 1/2, 1, 6, 2, 7, 9, 9],
   [1/9, 1/8, 1/6, 1, 1/3, 3, 3, 9],
   [1/9, 1/3, 1/2, 3, 1, 5, 8, 8],
   [1/9, 1/8, 1/7, 1/3, 1/3, 1, 3, 3],
   [1/9, 1/9, 1/9, 1/3, 1/5, 1/3, 1, 2],
   [1/9, 1/9, 1/9, 1/9, 1/5, 1/3, 1/2, 1]



], dtype=float)

normalized_weights = calculate_geometric_mean(pairwise_matrix)
consistency_index = calculate_consistency_index(pairwise_matrix)

print("Normalized Weights:")
for i, weight in enumerate(normalized_weights, start=1):
    print(f"Criterion {i}: {weight:.4f}")

print(f"\nConsistency Index: {consistency_index:.4f}")
