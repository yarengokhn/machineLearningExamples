import numpy as np
import matplotlib.pyplot as plt

n = 40000
Z = np.random.randn(n)

plt.step(sorted(Z), np.arange(1, n + 1) / float(n), label="Gaussian")

# ============================ Question 10.b ============================
for k in [1, 8, 64, 512]:
    Zk = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1.0 / k), axis=1)
    plt.step(sorted(Zk), np.arange(1, n + 1) / float(n), label=f"k = {k}")
# =======================================================================

plt.xlim(-3,3)
plt.xlabel("Observations")
plt.ylabel("Probability")
plt.legend()
plt.show()
