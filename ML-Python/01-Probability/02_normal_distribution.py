import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

b = ss.distributions.binom
n = ss.distributions.norm

for flips in [5, 10, 20, 40, 80]:
    success = np.arange(flips)
    our_distribution = b.pmf(success, flips, .5)
    plt.hist(success, flips, weights=our_distribution)

    mu = flips * .5
    std_dev = np.sqrt(flips * .5 * (1 - .5))
    norm_x = np.linspace(mu - 3 * std_dev, mu + 3 * std_dev, 100)
    norm_y = n.pdf(norm_x, mu, std_dev)
    plt.plot(norm_x, norm_y, 'k')

plt.xlim(0, 55)
plt.show()
