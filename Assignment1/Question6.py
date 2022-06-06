import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special
import math


var = 3*math.sqrt(2)
U = np.arange(  0.001, 1 , 1/10000 )
temp1 = 2*U - 1
inv_err = special.erfinv(temp1)
normal_dist = var*inv_err

print(normal_dist)

s_normal = pd.Series(normal_dist)


ax1 = s_normal.plot.kde(label ='Normal distibution' )
plt.show()


sigma = 1.0
temp_rayleigh = -2*(sigma**2)*(np.log( 1- U) )
s_rayleigh = pd.Series(temp_rayleigh)

ax1 = s_rayleigh.plot.kde(label ='Rayleigh distibution' )
plt.show()


lambda_scale = 1.0
temp_exp = -lambda_scale*(np.log( 1- U) )
s_exp = pd.Series(temp_exp)

ax1 = s_exp.plot.kde(label ='Exponential distibution' )
plt.show()


