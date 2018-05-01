
# coding: utf-8

# # Week 1: Mean/Covariance of a data set and effect of linear transformation
# 
# In this week, we are going to investigate how the mean and (co)variance of a dataset changes
# when we apply affine transformation to the dataset.

# ## Learning objectives
# 1. Get Farmiliar with basic programming using Python and Numpy/Scipy.
# 2. Learn to appreciate implementing
#    functions to compute statistics of dataset in vectorized way.
# 3. Understand the effects of affine transformations on a dataset.
# 4. Understand the importance of testing in programming for machine learning.

# Here are a few links for your reference. You may want to refer back to them throughout the whole course.
# 
# - If you are less comfortable with programming in Python, have a look at this Coursera course https://www.coursera.org/learn/python.
# - To learn more about using Scipy/Numpy, have a look at the [Getting Started Guide](https://scipy.org/getting-started.html). You should also refer to the numpy [documentation](https://docs.scipy.org/doc/) for references of available functions.
# 
# - If you want to learn more about creating plots in Python, checkout the tutorials found on matplotlib's website 
# https://matplotlib.org/tutorials/index.html. Once you are more familiar with plotting, check out this excellent blog post http://pbpython.com/effective-matplotlib.html.
# 
# - There are more advanced libraries for interactive data visualization. For example, [bqplot](https://github.com/bloomberg/bqplot) or [d3.js](https://d3js.org/). You may want to check out other libraries if you feel adventurous.
# 
# - Although we use Jupyter notebook for these exercises, you may also want to check out [Jupyter Lab](https://github.com/jupyterlab/jupyterlab) when you want to work on your own projects.

# First, let's import the packages that we will use for the week. Run the cell below to import the packages.

# In[1]:

# PACKAGE: DO NOT EDIT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_mldata, fetch_olivetti_faces
import time
import timeit


# In[2]:

get_ipython().magic('matplotlib inline')
from ipywidgets import interact


# Next, we are going to retrieve Olivetti faces dataset.
# 
# When working with some datasets, before digging into further analysis, it is almost always
# useful to do a few things to understand your dataset. First of all, answer the following
# set of questions:
# 
# 1. What is the size of your dataset?
# 2. What is the dimensionality of your data?
# 
# The dataset we have are usually stored as 2D matrices, then it would be really important
# to know which dimension represents the dimension of the dataset, and which represents
# the data points in the dataset. 

# In[3]:

image_shape = (64, 64)
# Load faces data
dataset = fetch_olivetti_faces()
faces = dataset.data

print('Shape of the faces dataset: {}'.format(faces.shape))
print('{} data points'.format(faces.shape[0]))


# When your dataset are images, it's a really good idea to see what they look like.
# 
# One very
# convenient tool in Jupyter is the `interact` widget, which we use to visualize the images (faces). For more information on how to use interact, have a look at the documentation [here](http://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html).

# In[4]:

@interact(n=(0, len(faces)-1))
def display_faces(n=0):
    plt.figure()
    plt.imshow(faces[n].reshape((64, 64)), cmap='gray')
    plt.show()


# ## 1. Mean and Covariance of a Dataset
# 
# You will now need to implement functions to which compute the mean and covariance of a dataset.
# 
# There are two ways to compute the mean and covariance. The naive way would be to iterate over the dataset
# to compute them. This would be implemented as a `for` loop in Python. However, computing them for large
# dataset would be slow. Alternatively, you can use the functions provided by numpy to compute them, these are much
# faster as numpy uses machine code to compute them. You will implment function which computes mean and covariane both
# in the naive way and in the fast way. Later we will compare the performance between these two approaches. If you need to find out which numpy routine to call, have a look at the documentation https://docs.scipy.org/doc/numpy/reference/.
# It is a good exercise to refer to the official documentation whenever you are not sure about something.

# __When you implement the functions for your assignment, make sure you read
# the docstring which dimension of your inputs corresponds to the number of data points and which 
# corresponds to the dimension of the dataset.__

# In[5]:

# ===YOU SHOULD EDIT THIS FUNCTION===
def mean_naive(X):
    """Compute the mean for a dataset by iterating over the dataset
    
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    N, D = X.shape
    mean = np.zeros(D)
    for n in range(N):
        mean = mean + X[n,:]
    mean = mean/float(N)
    return mean

# ===YOU SHOULD EDIT THIS FUNCTION===
def cov_naive(X):
    """Compute the covariance for a dataset
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    covariance: (D, D) ndarray which is the covariance matrix of the dataset.
    
    """
    N, D = X.shape
    covariance = np.zeros((D, D))
    for n in range(N):
        covariance = np.zeros((D, D))
    return covariance


# In[6]:

# GRADED FUNCTION: DO NOT EDIT THIS LINE

# ===YOU SHOULD EDIT THIS FUNCTION===
def mean(X):
    """Compute the mean for a dataset
    
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    N, D = X.shape
    mean = np.dot(X.T, np.ones((N,1)))/float(N)
    return mean
 
# ===YOU SHOULD EDIT THIS FUNCTION===
def cov(X):
    """Compute the covariance for a dataset
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    covariance_matrix: (D, D) ndarray which is the covariance matrix of the dataset.
    
    """
    # It is possible to vectorize our code for computing the covariance, i.e. we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    covariance_matrix = np.cov(X.T) # EDIT THIS
    return covariance_matrix


# With the `mean` function implemented, let's take a look at the _mean_ face of our dataset!

# In[7]:

def mean_face(faces):
    """Compute the mean of the `faces`
    
    Arguments
    ---------
    faces: (N, 64 * 64) ndarray representing the faces dataset.
    
    Returns
    -------
    mean_face: (64, 64) ndarray which is the mean of the faces.
    """
    mean_face = mean(faces)
    return mean_face

plt.imshow(mean_face(faces).reshape((64, 64)), cmap='gray');


# To put things into perspective, we can benchmark the two different implementation with the `%time` function
# in the following way:

# In[8]:

# We have some huge data matrix, and we want to compute its mean
X = np.random.randn(100000, 20)
# Benchmarking time for computing mean
get_ipython().magic('time mean_naive(X)')
get_ipython().magic('time mean(X)')
pass


# In[9]:

# Benchmarking time for computing covariance
get_ipython().magic('time cov_naive(X)')
get_ipython().magic('time cov(X)')
pass


# Alternatively, we can also see how running time increases as we increase the size of our dataset.
# In the following cell, we run `mean`, `mean_naive` and `cov`, `cov_naive` for many times on different sizes of
# the dataset and collect their running time. If you are less familiar with Python, you may want to spend
# some time understanding what the code does. __Understanding how your code scales with the size of your dataset (or dimensionality of the dataset) is crucial__ when you want to apply your algorithm to larger dataset. This is really important when we propose alternative methods a more efficient algorithms to solve the same problem. We will use these techniques again later in this course to analyze the running time of our code.

# In[10]:

def time(f, repeat=100):
    """A helper function to time the execution of a function.
    
    Arguments
    ---------
    f: a function which we want to time it.
    repeat: the number of times we want to execute `f`
    
    Returns
    -------
    the mean and standard deviation of the execution.
    """
    times = []
    for _ in range(repeat):
        start = timeit.default_timer()
        f()
        stop = timeit.default_timer()
        times.append(stop-start)
    return np.mean(times), np.std(times)


# In[11]:

fast_time = []
slow_time = []

for size in np.arange(100, 5000, step=100):
    X = np.random.randn(size, 20)
    f = lambda : mean(X)
    mu, sigma = time(f)
    fast_time.append((size, mu, sigma))
    
    f = lambda : mean_naive(X)
    mu, sigma = time(f)
    slow_time.append((size, mu, sigma))

fast_time = np.array(fast_time)
slow_time = np.array(slow_time)


# In[12]:

fig, ax = plt.subplots()
ax.errorbar(fast_time[:,0], fast_time[:,1], fast_time[:,2], label='fast mean', linewidth=2)
ax.errorbar(slow_time[:,0], slow_time[:,1], slow_time[:,2], label='naive mean', linewidth=2)
ax.set_xlabel('size of dataset')
ax.set_ylabel('running time')
plt.legend();


# In[12]:

## === FILL IN THIS, follow the approach we have above ===
fast_time_cov = []
slow_time_cov = []
for size in np.arange(100, 5000, step=100):
    X = np.random.randn(size, 20)
    f = lambda : cov(X)           # EDIT THIS
    mu, sigma = time(f) # EDIT THIS
    fast_time_cov.append((size, mu, sigma))
    
    f = lambda : cov_naive(X)         # EDIT THIS
    mu, sigma = time(f) # EDIT THIS
    slow_time_cov.append((size, mu, sigma))

fast_time_cov = np.array(fast_time_cov)
slow_time_cov = np.array(slow_time_cov)


# In[13]:

fig, ax = plt.subplots()
ax.errorbar(fast_time_cov[:,0], fast_time_cov[:,1], fast_time_cov[:,2], label='fast covariance', linewidth=2)
ax.errorbar(slow_time_cov[:,0], slow_time_cov[:,1], slow_time_cov[:,2], label='naive covariance', linewidth=2)
ax.set_xlabel('size of dataset')
ax.set_ylabel('running time')
plt.legend();


# ## 2. Affine Transformation of Dataset
# In this week we are also going to verify a few properties about the mean and
# covariance of affine transformation of random variables.
# 
# Consider a data matrix $\boldsymbol{X}$ of size (N, D). We would like to know
# what is the covariance when we apply an affine transformation $\boldsymbol{A}\boldsymbol{x}_i + \boldsymbol{b}$ with a matrix $\boldsymbol A$ and a vector $\boldsymbol b$ to each datapoint $\boldsymbol{x}_i$ in $\boldsymbol{X}$, i.e.
# we would like to know what happens to the mean and covariance for the new dataset if we apply affine transformation.

# In[58]:

# GRADED FUNCTION: DO NOT EDIT THIS LINE

# ===YOU SHOULD EDIT THIS FUNCTION===
def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        mean: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    affine_m = ((A @ mean).T + b).T # EDIT THIS
    return affine_m

# ===YOU SHOULD EDIT THIS FUNCTION===
def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation
    Args:
        S: ndarray, the covariance matrix
        A, b: affine transformation applied to each element in X        
    Returns:
        covariance matrix after the transformation
    """
    affine_cov = A @ S @ A.T # EDIT THIS
    return affine_cov


# Once the two functions above are implemented, we can verify the correctness our implementation. Assuming that we have some matrix $\boldsymbol A$ and vector $\boldsymbol b$.

# In[15]:

random = np.random.RandomState(42)
A = random.randn(4,4)
b = random.randn(4)


# Next we can generate some random dataset $\boldsymbol{X}$

# In[16]:

X = random.randn(100, 4)


# Assuming that for some dataset $\boldsymbol X$, the mean and covariance are $\boldsymbol m$, $\boldsymbol S$, and for the new dataset after affine transformation $ \boldsymbol X'$, the mean and covariance are $\boldsymbol m'$ and $\boldsymbol S'$, then we would have the following identity:
# 
# $$\boldsymbol m' = \text{affine_mean}(\boldsymbol m, \boldsymbol A, \boldsymbol b)$$
# 
# $$\boldsymbol S' = \text{affine_covariance}(\boldsymbol S, \boldsymbol A, \boldsymbol b)$$

# In[17]:

X1 = ((A @ (X.T)).T + b)  # applying affine transformation once
X2 = ((A @ (X1.T)).T + b) # and again


# One very useful way to compare whether arrays are equal/similar is use the helper functions
# in `numpy.testing`. the functions in `numpy.testing` will throw an `AssertionError` when the output does not satisfy the assertion.

# In[59]:

np.testing.assert_almost_equal(mean(X1), affine_mean(mean(X), A, b))
np.testing.assert_almost_equal(cov(X1),  affine_covariance(cov(X), A, b))
print('correct')


# Fill in the `???` below

# In[61]:

np.testing.assert_almost_equal(mean(X2), affine_mean(mean(X1), A, b))
np.testing.assert_almost_equal(cov(X2),  affine_covariance(cov(X1), A, b))
print('correct')


# Check out the numpy [documentation](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.testing.html)
# for details.
# 
# If you are interested in learning more about floating point arithmetic, here is a good [paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.22.6768).

# In[23]:

cov(X1)


# In[52]:

A @ cov(X) @ A.T + b.T


# In[53]:

cov(((A @ (X.T)).T))


# In[ ]:




# In[57]:

A @ cov(X) @ A.T


# In[ ]:




# In[ ]:



