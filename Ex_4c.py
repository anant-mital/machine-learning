import matplotlib.pyplot as plt
from PyUtils import gaussian,log_gaussian

mu = 0
sigma = 1
x = np.linspace(-5,5,num=1000)
plt.plot(x, gaussian(x,mu,sigma), 'r')
plt.show()

plt.plot(x, log_gaussian(x,mu,sigma), 'r')
plt.show()

