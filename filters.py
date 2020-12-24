import numpy as np

# Just assorted pre-defined filters returned by calling a function to create the numpy array

# Returns a central differencing filter (x direction)
def central_x():
    return np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

# Returns a central differencing filter (y direction)
def central_y():
    return np.array([[-1,-2,-1],[0,0,0],[1,2,1]])