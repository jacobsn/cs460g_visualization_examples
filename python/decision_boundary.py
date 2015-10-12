#
# Example code that shows how to plot the function value and a decision
# boundary for a simple logistic regression model using python
#
# Author: Nathan Jacobs (with some initial code by Hampton Young)
#

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# define our hypothesis (vectorized!)
def f(x): 
  return expit(np.matrix([0, 1, -.5,.5])*x);
 
# create the domain for the plot
x_min = -5; x_max = 5
y_min = -5; y_max = 5

x1 = np.linspace(x_min, x_max, 200)
y1 = np.linspace(y_min, y_max , 200)
x,y = np.meshgrid(x1, y1)

#
# evalute it in a vectorized way (and reshape into a matrix)
#

# make a 3 x N matrix of the sample points
data = np.vstack((
   np.ones(x.size), # add the bias term
   x.ravel(), # make the matrix into a vector
   y.ravel(), 
   y.ravel()**2)) # add a quadratic term for fun

z = f(data)

z = z.reshape(x.shape)

#
# Make the plots
#

# show the function value in the background
cs = plt.imshow(z,
  extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
  cmap=plt.cm.jet)
plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
levels = np.array([.5])
cs_line = plt.contour(x,y,z,levels)

# add a color bar
CB = plt.colorbar(cs)

plt.show()

