import matplotlib.pyplot as plt
import numpy as np

w = np.sqrt(0.25)*np.random.randn(100)
f = 0.017
x = np.zeros(100)
for i in range(100):
    x[i] = np.sin(2*np.pi*i) + w[i]

plt.plot(x)

scores = []
frequencies = []

for f in np.linspace(0,0.5,1000):
    n = np.arange(100)
    z = -2*np.pi*1j*f*n
    e = np.exp(z)
    
    score = np.abs(np.dot(x,e))
    scores.append(score)
    frequencies.append(f)
fHat = frequencies[np.argmax(scores)]
print(fHat)