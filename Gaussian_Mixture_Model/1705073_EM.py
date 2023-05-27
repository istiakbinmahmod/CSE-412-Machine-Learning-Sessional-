import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import inv
import seaborn as sns

def plot_gmm(X, mu, sigma, pi, R, iteration):
    # Plot data points
    # plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=R.argmax(axis=1), cmap='viridis')
    # Plot gaussian distributions
    for i in range(len(mu)):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        pos = np.dstack((x, y))
        rv = multivariate_normal(mean=mu[i], cov=sigma[i])
        plt.contour(x, y, rv.pdf(pos))
        plt.title('Iteration ' + str(iteration))
        # plt.clf()
    # plt.title('Iteration ' + str(iteration))
    # plt.pause(0.005)

max_iters = 100

#Read in the data file and store in numpy array
data = np.loadtxt("data2D.txt")

# Assume range for number of components (k)
k_range = range(1, 11)

# Lists to store log-likelihoods and k values
log_likelihoods = []
ks = []

for k in k_range:
    # Initialize parameters
    n, d = data.shape
    means = data[np.random.choice(n, k, False), :]
    covs = [np.eye(d) for _ in range(k)]
    sigmas = np.ones(k) / k
    responsibilities = np.zeros((n, k))
    
    # Converge 
    log_likelihood = None

    # EM algorithm
    for _ in range(max_iters):
        # E-step
        for i in range(k):
            responsibilities[:, i] = sigmas[i] * multivariate_normal.pdf(data, mean=means[i], cov=covs[i], allow_singular=True)
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

        # M-step
        for i in range(k):
            Nk = responsibilities[:, i].sum()
            means[i] = responsibilities[:, i].dot(data) / Nk
            covs[i] = np.sum([responsibilities[j, i] * np.outer(data[j] - means[i], data[j] - means[i]) for j in range(n)], axis=0) / Nk
            sigmas[i] = Nk / n
        
        # Check for convergence and log_likelihood calc
        log_likelihood_new = 0
        for j in range(k):
            log_likelihood_new += sigmas[j] * multivariate_normal.pdf(data, mean=means[j], cov=covs[j], allow_singular=True)
        log_likelihood_new = np.log(log_likelihood_new).sum()

        if log_likelihood is not None and np.abs(log_likelihood_new - log_likelihood) < 1e-6:
            break
        log_likelihood = log_likelihood_new

         # Regularize covariances
        for i in range(k):
            covs[i] = covs[i] + 1e-6 * np.eye(d)
            
    # Store log-likelihood and k value
    log_likelihoods.append(log_likelihood)
    ks.append(k)
    

# Plot log-likelihoods against k values
plt.plot(ks, log_likelihoods)
plt.xlabel("Number of Components (k)")
plt.ylabel("Converged Log-Likelihood")
plt.show()

# take k_star as user input
k = int(input("Enter k_star: "))

# Initialize parameters
n, d = data.shape
means = data[np.random.choice(n, k, False), :]
covs = [np.eye(d) for _ in range(k)]
weights = np.ones(k) / k
resp = np.zeros((n, k))


# EM algorithm
for iter in range(max_iters):
    # E-step
    for i in range(k):
        resp[:, i] = weights[i] * multivariate_normal.pdf(data, mean=means[i], cov=covs[i])
    resp = resp / resp.sum(axis=1, keepdims=True)

    # M-step
    for i in range(k):
        Nk = resp[:, i].sum()
        means[i] = resp[:, i].dot(data) / Nk
        covs[i] = np.sum([resp[j, i] * np.outer(data[j] - means[i], data[j] - means[i]) for j in range(n)], axis=0) / Nk
        weights[i] = Nk / n
    plt.ion()
    plt.clf()
    plot_gmm(data, means, weights, covs, resp, iter)
    plt.draw()
    plt.pause(0.005)
    plt.ioff()
    # plt.show()        