import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Create a bell curve plot using numpy and stats
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
plt.plot(x, norm.pdf(x))

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
