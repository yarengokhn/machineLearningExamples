#part a 
import matplotlib.pyplot as plt
import numpy as np

#

n=40000
Z=np.random.randn(n)
plt.step(sorted(Z), np.arange(1,n+1)/float(n),label="Gaussian")
# plt.xlim(-3,3)
# plt.xlabel("Observations")
# plt.ylabel("Proability")
# plt.show()

#part b 
k_values = [1, 8, 64, 512]
for k in k_values:
    Zk = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1./k), axis=1)

    # Calculate empirical CDF for every Zk values 
    FZk = np.arange(1, n+1) / float(n)

    # Plot the empirical CDF
    plt.step(sorted(Zk), FZk, label= k)

plt.xlim(-3,3)
plt.xlabel("Observations")
plt.ylabel("Proability")
plt.legend()
plt.show()
