import numpy as np
import matplotlib.pyplot as plt

#take the dimensions of matrix n as input
n = int(input("Enter the dimensions of the matrix(n): "))

#produce a random n*n invertible matrix A. 
A = []

while True:
    for i in range(n):
        A.append([])
        for j in range(n):
            A[i].append(np.random.randint(n))
    A = np.array(A)
    if np.linalg.det(A) != 0:
        break

print("A matrix is = \n", A)

#perform Eigen Decomposition using numpy
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues are = \n", eigenvalues)
print("Eigenvectors are = \n", eigenvectors)

#reconstruct the matrix A from the eigenvalues and eigenvectors
A_reconstructed = eigenvectors.dot(np.diag(eigenvalues)).dot(np.linalg.inv(eigenvectors))
print("Reconstructed matrix is = \n", A_reconstructed)

#check if the reconstructed matrix is equal to the original matrix
print("Is the reconstructed matrix equal to the original matrix? ", np.allclose(A, A_reconstructed))