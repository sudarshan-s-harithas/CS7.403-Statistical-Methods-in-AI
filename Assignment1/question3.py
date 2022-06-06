import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


exponential_dist = np.random.exponential(1, 10000)
exponential_dist1 = np.random.exponential(exponential_dist, 10000)



normal_dist = np.random.normal(1, 1, 10000)

s_normal = pd.Series(normal_dist)

s_exp = pd.Series(exponential_dist1)

print("----------------")


variance_exp = np.var( exponential_dist)
mean_exp = np.mean(exponential_dist)


print( "variance of exponential distribution" + str( variance_exp)) 
print( "mean of exponential distribution" +  str(mean_exp))


print("----------------")

variance_normal = np.var( normal_dist)
mean_normal = np.mean(normal_dist)

print( "variance of normal distribution" +  str( variance_normal)) 
print(  "mean of normal distribution" +  str(mean_normal))


print("----------------")


ax1 = s_exp.plot.kde(label ='Exponential distibution' )
ax2 = s_normal.plot.kde(label =' Normal Distribution')

plt.xlim([-5, 5.0])
plt.ylabel("Probablity Density" , fontsize=16)
plt.legend()

plt.show()

