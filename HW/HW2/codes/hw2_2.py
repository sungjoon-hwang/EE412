import numpy as np


def l2_norm(x):
    return np.sqrt(np.sum(x**2))


def power_iteration(M, delta: float = 1e-6):
    """
    INPUT: matrix M and threshold delta
    OUTPUT: dominant eigenvector of M

    DESCRIPTION: Returns the dominant eigenvector of M using power iteration.
    Does not use np.linalg.norm, instead normalizes manually.
    """
    # Step 1: Initialize a random vector `b_k`.
    # Step 2: Iterate until convergence by updating `b_k` and normalizing.
    # Step 3: Return the dominant eigenvector.

    n = M.shape[1]

    b_k = np.ones(n)
    b_k /= l2_norm(b_k)

    diff = np.inf
    while diff > delta:
        b_k_next = np.dot(M, b_k)
        b_k_next /= l2_norm(b_k_next)
        b_k_next = b_k_next if b_k_next[0] > 0 else -b_k_next

        diff = l2_norm(b_k_next - b_k)
        b_k = b_k_next

    return b_k


def deflate_matrix(M, eigvec):
    """
    INPUT: matrix M, and dominant eigenvector eigvec
    OUTPUT: deflated matrix M after removing the contribution of eigvec

    DESCRIPTION: Deflates the matrix M to find subsequent eigenvectors.
    """
    # Deflate the matrix to remove the contribution of the eigenvector.
    eigval = np.mean(np.dot(M, eigvec) / eigvec)

    col = eigvec[:, np.newaxis]
    row = eigvec[np.newaxis, :]
    return M - eigval * np.dot(col, row)


def compute_eigenvalues(M, num_eigenvalues=2):
    """
    INPUT: matrix M
    OUTPUT: list of eigenvalues using power iteration for each eigenvector

    DESCRIPTION: Uses power iteration to find the dominant eigenvalue of M,
    deflates the matrix and finds subsequent eigenvalues.
    """
    # Use power iteration to compute multiple eigenvalues.
    # Deflate the matrix for each iteration to find subsequent eigenvalues.
    M_walk = M
    eigvals = []

    for _ in range(num_eigenvalues):
        eigvec = power_iteration(M_walk)

        eigval = np.dot(eigvec, np.dot(M_walk, eigvec))
        M_walk = deflate_matrix(M_walk, eigvec)

        eigvals.append(eigval)

    return eigvals


def svd_manual(M):
    """
    INPUT: matrix M
    OUTPUT: matrices U, Sigma, V using SVD

    DESCRIPTION: Computes SVD by calculating eigenvalues and eigenvectors for M^T M and M M^T.
    """
    # Step 1: Compute M^T M and M M^T.

    MTM = np.dot(M.transpose(), M)
    MMT = np.dot(M, M.transpose())

    # Step 2: Compute the eigenvectors using power iteration.

    sigma = np.sqrt(compute_eigenvalues(MTM))
    R = len(sigma)

    V = []
    U = []
    for i in range(R):
        # MTM
        vi = power_iteration(MTM)
        MTM = deflate_matrix(MTM, vi)
        V.append(vi)

        # MMT
        ui = power_iteration(MMT)
        MMT = deflate_matrix(MMT, ui)
        U.append(ui)

    # Step 3: Compute singular values and construct U, Sigma, and V.

    U = np.array(U).transpose()
    Sigma = np.diag(sigma)
    V = np.array(V).transpose()

    return U, Sigma, V


def matrix_approximation(U, Sigma, V):
    """
    INPUT: matrices U, Sigma, V
    OUTPUT: matrix approximation with one singular value set to zero

    DESCRIPTION: Computes the matrix approximation by setting the smaller singular value to zero.
    """
    # Set the smaller singular value to zero and recompute the matrix.
    U1 = U[:, :-1]
    S1 = Sigma[:-1, :-1]
    V1 = V[:, :-1]

    return np.dot(U1, np.dot(S1, V1.transpose()))


def energy_retained(Sigma):
    """
    INPUT: list of singular values
    OUTPUT: percentage of energy retained

    DESCRIPTION: Calculates the energy retained by the one-dimensional approximation.
    The energy is the sum of the squares of the singular values.
    """
    # Compute total energy and retained energy, return the percentage.

    sigma = np.array([Sigma[i][i] for i in range(Sigma.shape[0])])  # diagonal
    total = np.sum(sigma**2)
    retained = np.sum(sigma[:-1] ** 2)

    return 100 * (retained / total)


# Main execution
if __name__ == "__main__":
    # Given matrix M
    M = np.array([[1, 2, 3], [3, 4, 5], [5, 4, 3], [0, 2, 4], [1, 3, 5]])

    # Part (a): Compute M^T M and M M^T
    MTM = np.dot(M.transpose(), M)
    MMT = np.dot(M, M.transpose())
    print("M^T M:", MTM)
    print("M M^T:", MMT)

    # Part (b): Eigenpairs using numpy.linalg.eig()
    eigvals_MTM, eigvecs_MTM = np.linalg.eig(MTM)
    eigvals_MMT, eigvecs_MMT = np.linalg.eig(MMT)
    print("Eigenvalues of M^T M:", eigvals_MTM)
    print("Eigenvectors of M^T M:", eigvecs_MTM)

    eigenvalues_manual_MTM = compute_eigenvalues(MTM)
    eigenvalues_manual_MMT = compute_eigenvalues(MMT)
    print("Eigenvalues of M^T M using power iteration:", eigenvalues_manual_MTM)
    print("Eigenvalues of M M^T using power iteration:", eigenvalues_manual_MMT)

    # Part (c): SVD implementation
    U, Sigma, V = svd_manual(M)
    print("U:", U)
    print("Sigma:", Sigma)
    print("V:", V)

    # Part (d): One-dimensional approximation
    M_approx = matrix_approximation(U, Sigma, V)
    print("One-dimensional approximation of M:", M_approx)

    # Part (e): Energy retained
    energy_percentage = energy_retained(Sigma)
    print("Energy retained (%)  :", energy_percentage)
