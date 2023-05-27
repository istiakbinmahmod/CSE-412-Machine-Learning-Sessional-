import numpy as np
import matplotlib.pyplot as plt

#take the dimensions of matrix n,m as input
n = int(input("Enter the dimensions of the matrix(n): "))
m = int(input("Enter the dimensions of the matrix(m): "))

#produce a random n*m invertible matrix A.
# A = np.random.rand(n,m)
A = []

for i in range(n):
    A.append([])
    for j in range(m):
        A[i].append(np.random.randint(m))
A = np.array(A)

print("A matrix is = \n", A)

#perform Singular Value Decomposition using numpy
U, s, V = np.linalg.svd(A)
print("U is = \n", U)
print("s is = \n", s)
print("V is = \n", V)

s = np.diag(s)
s_row, s_col = s.shape

if n > s_row:
    s = np.pad(s, ((0, n-s_row), (0, 0)), 'constant', constant_values=(0, 0))
if m > s_col:
    s = np.pad(s, ((0, 0), (0, m-s_col)), 'constant', constant_values=(0, 0))

#Calculate the Moore-Penrose pseudoinverse of A
A_pseudoinverse = V.T.dot(np.linalg.pinv(s)).dot(U.T)
print("Pseudoinverse of A is = \n", A_pseudoinverse)

#Calculate the Moore-Penrose pseudoinverse of A using numpy
A_pseudoinverse_numpy = np.linalg.pinv(A)
print("Pseudoinverse of A using numpy is = \n", A_pseudoinverse_numpy)

#check if the pseudoinverse of A is equal to the pseudoinverse of A using numpy
print("Is the pseudoinverse of A equal to the pseudoinverse of A using numpy? ", np.allclose(A_pseudoinverse, A_pseudoinverse_numpy))