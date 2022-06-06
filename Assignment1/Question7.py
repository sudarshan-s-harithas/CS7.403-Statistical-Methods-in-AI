import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special
import math
import random 


U = np.arange(  0.001, 1 , 1/10000 )


def generate_500_random_pts( uniform_dist ):

	sampled_values = random.choices(uniform_dist, k=500)

	sum_val = np.sum(sampled_values)

	return sampled_values , sum_val





union_sample_list = np.array([0])

for i in range(50000):

	temp , sum_val= generate_500_random_pts(U)

	union_sample_list  = np.append( union_sample_list , [sum_val] , axis =0)
	# print(temp)


print(union_sample_list)

print( np.shape(union_sample_list))
plt.hist(union_sample_list, bins=[k for k in range(round (min(union_sample_list)-1), round(max(union_sample_list)+1))], color ="blue")
plt.show()

