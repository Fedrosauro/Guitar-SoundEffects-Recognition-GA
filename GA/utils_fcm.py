import numpy as np

def initialize_membership_matrix(n, c):
    # Initialize a random partition matrix U with constraints
    U = np.random.rand(n, c)
    U = U / np.sum(U, axis=1, keepdims=True)
    return U

def compute_centroids(X, U, c, lambda_exp):
    # Calculate the centroids for each cluster
    n_features = X.shape[1]
    centroids = np.zeros((c, n_features))
    for j in range(c):
        numerator = np.sum((U[:, j] ** lambda_exp)[:, np.newaxis] * X, axis=0)
        denominator = np.sum(U[:, j] ** lambda_exp)
        centroids[j, :] = numerator / denominator
    return centroids

def update_membership_matrix(X, centroids, c, lambda_exp):
    # Update the membership matrix U
    n = X.shape[0]
    U_new = np.zeros((n, c))
    for i in range(n):
        for j in range(c):
            sum_terms = np.sum([(np.linalg.norm(X[i] - centroids[j]) / np.linalg.norm(X[i] - centroids[k])) ** (2 / (lambda_exp - 1)) for k in range(c)])
            U_new[i, j] = 1 / sum_terms
    return U_new

def fuzzy_c_means(X, c, lambda_exp=2, max_iter=100, tolerance=1e-4):
    # Fuzzy C-Means Clustering Algorithm
    n = X.shape[0]
    U = initialize_membership_matrix(n, c)
    
    for _ in range(max_iter):
        #print(f"Iteration {_}")
        centroids = compute_centroids(X, U, c, lambda_exp)
        U_new = update_membership_matrix(X, centroids, c, lambda_exp)
        
        #print(f"Difference between U_new and U: {np.linalg.norm(U_new - U)}")
        # Check for convergence
        if np.linalg.norm(U_new - U) < tolerance:
            break
        U = U_new
    
    return U, centroids

def calculate_dissimilarity_fcm(TA, TB, U):
    # Calculate the dissimilarity based on FCM
    n, m = len(TA), len(TB)
    dissimilarity = 0
    for i in range(n):
        for j in range(m):
            dissimilarity += U[i, j] * np.linalg.norm(TA[i] - TB[j])
    return dissimilarity / (n * m)
