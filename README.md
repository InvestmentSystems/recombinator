# Recombinator - Statistical Resampling in Python


## Overview

Recombinator is a Python package for statistical resampling in Python. It provides various algorithms for the iid bootstrap, the block bootstrap, as well as optimal block-length selection. 

## Algorithms

*   I.I.D. bootstrap: Standard i.i.d. bootstrap for one-dimensional and multi-dimensional data, balanced bootstrap, anthithetic bootstrap  
*   Block based bootstrap: Moving Block Bootstrap, Circular Block Bootstrap, Stationary Bootstrap, Tapered Block-Bootstrap 
*   Optimal block-length selection algorithm for Circular Block Bootstrap and Stationary Bootstrap

## Table of Contents

1.  [Installation](#installation)
2.  [Getting Started](#getting-started)


## Installation
### Latest Release
```shell
    pip install recombinator
```
or 
```shell
    pip3 install recombinator
```
if not using Anaconda.

To get the latest version, clone the repository from github, 
open a terminal/command prompt, navigate to the root folder and install via
```shell
    pip install .
```
or 
```shell
    pip3 install . 
```
if not using Anaconda.

### Most Recent Version on GitHub
1. Clone the github repository via

```shell
    git clone https://github.com/InvestmentSystems/recombinator.git
```
    
2. Navigate to the Recombinator base directory and run
```shell
    pip install .
``` 
    
## Getting Started
Please see the Jupyter notebooks 'notebooks/Block Bootstrap.ipynb' and 'notebooks/IID Bootstrap.ipynb' for more examples.

### Basic I.I.D. Bootstrap
Import modules  

```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np

from recombinator.iid_bootstrap import \
    iid_balanced_bootstrap, \
    iid_bootstrap, \
    iid_bootstrap_vectorized, \
    iid_bootstrap_via_choice, \
    iid_bootstrap_via_loop, \
    iid_bootstrap_with_antithetic_resampling
```

For illustrative purposes, generate an original dataset of sample size n=100 to resample from.  

```python
n=100
np.random.seed(1)
x = np.abs(np.random.randn(n))
```

Estimate the 75th percentile from the original sample
```python
percentile = 75
original_statistic = np.percentile(x, percentile)
print(original_statistic)
```

Now generate 100000 new bootstrap samples by drawing from the original sample with replacement.
```python
R = 100000
x_resampled = iid_bootstrap(x, replications=R)
```
This produces a 100000 x 100 dimensional NumPy array. Each row is a new sample.

If we instead wanted to resample without replacement, we could draw shorter samples. 
This is known as as subsampling. See these lecture notes by Charles Geyer for statistical background: 
http://www.stat.umn.edu/geyer/5601/notes/sub.pdf. 

In recombinator, subsampling is achieved by using the keyword arguments "sub_sample_length" and "replace". 
To draw samples of size 50 without replacement we would write
```python
x_resampled = iid_bootstrap(x, replications=R, sub_sample_length=50, replace=False)
```

Let us return to the original example of resampling at the full sample size with replacement.
We are interested in the sampling distribution of a statistic (in this example the 75th percentile of the distribution).
Hence, we calculate the statistic on each of the new bootstrap samples:
```python
resampled_statistic = np.percentile(x_resampled, percentile, axis=1)
```

We can use Recombinator to estimate the bootstrap standard error and the 95% 
confidence interval of the statistic as follows.
```python
from recombinator.statistics import \
    estimate_confidence_interval_from_bootstrap, \
    estimate_standard_error_from_bootstrap
```

Estimate the standard error of the estimate of the 75th percentile via bootstrap:
```python
estimate_standard_error_from_bootstrap(bootstrap_estimates=resampled_statistic,
                                       original_estimate=original_statistic)
```

Estimate the 95% confidence interval of the 75th percentile via bootstrap:
```python
estimate_confidence_interval_from_bootstrap(bootstrap_estimates=resampled_statistic, 
                                            confidence_level=95)
```
 
Recombinator supports other variations of the standard i.i.d. bootstrap 
(the balanced and the antithetic bootstrap). Please see the Jupyter notebook on the I.I.D. boostrap for examples.

### Block-Based Bootstrap for Time-Series
Recombinator offers the following block-based approaches to resample temporally dependent data:
* Moving Block Bootstrap - Kuensch (1989)
* Circular Block Bootstrap - Politis and Romano (1992)
* Stationary Bootstrap - Politis and Romano (1994)
* Tapered Block Bootstrap - Paparoditis and Politis (2001)  

Import statsmodels for the estimation of time-series models.
```python
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.ar_model import AR
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
```


Generate a sample of length n=1000 from an AR(1) process with autoregressive coefficient 0.5:
```python
np.random.seed(1)

# number of time periods
T = 1000

# draw random errors
e = np.random.randn(T)
y = np.zeros((T,))

# y is an AR(1) with phi_1 = 0.5
phi_1 = 0.5
y[0] = e[0]*np.sqrt(1.0/(1.0-phi_1**2))
for t in range(1, T):
    y[t]=phi_1*y[t-1] + e[t]
```


#### Optimal Block-Length
In order to preserve temporal dependence in time-series data, 
bootstrap algorithms sample from the original data in blocks rather than 
sampling single observations. 
A key question is what block length to use. 
We are using a data based block length selection algorithm due to Politis and White (2004) 
with corrections by Patton, Politis, and White (2007). 
This algorithm produces optimal block-lengths for the circular block bootstrap and the stationary bootstrap for the estimation of the variance of the mean.
  
Import the optimal block-length selection functionality:
```python
from recombinator.optimal_block_length import optimal_block_length
```

Compute the optimal block length of the sample from the AR(1) process created above:
```python
b_star = optimal_block_length(y)
b_star_sb = b_star[0].b_star_sb
b_star_cb = math.ceil(b_star[0].b_star_cb)
print(f'optimal block length for stationary bootstrap = {b_star_sb}')
print(f'optimal block length for circular bootstrap = {b_star_cb}')
```

#### Resampling using a Block-Based Bootstrap
We illustrate the procedure using the circular-block bootstrap.
```python
from recombinator.block_bootstrap import circular_block_bootstrap
```

The true autoregressive coefficient of the process we simulated is 0.5. 
The estimated coefficient from the simulated time-series is obtained as follows
```python
ar = AR(y)
estimate_from_original_data = ar.fit(maxlag=1)
print(estimate_from_original_data.params[1])
```

Statsmodels can produce an estimate of the standard error of the estimate of the autoregressive coefficient resorting to asymptotic theory:
```python
print(estimate_from_original_data.bse[1])
```
 
Generate 10000 new time-series by resampling using a circular-block bootstrap
```python
# number of replications for bootstraps (number of resampled time-series to generate)
B = 10000

y_star_cb \
    = circular_block_bootstrap(y, 
                               block_length=b_star_cb, 
                               replications=B, 
                               replace=True)
```

Now estimate the AR coefficient on each of the bootstrap samples
```python
estimates_from_bootstrap = []
ar_estimates_from_bootstrap = np.zeros((B, ))

for b in range(B):
    y_bootstrap = y_star_cb[b, :]
    ar_bootstrap = AR(y_bootstrap)
    estimate_from_bootstrap = ar_bootstrap.fit(maxlag=1)
    estimates_from_bootstrap.append(estimate_from_bootstrap)
    ar_estimates_from_bootstrap[b] = estimate_from_bootstrap.params[1]
```

Plot the sampling distribution and compute its mean and median
```python
plt.hist(ar_estimates_from_bootstrap, bins=20)
print(f'mean={np.mean(ar_estimates_from_bootstrap)}')
print(f'median={np.median(ar_estimates_from_bootstrap)}')
```

It turns out that the mean and median are below the AR coefficient estimated on the original sample. 
This is due to the fact that the block bootstrap approach breaks the temporal 
dependence whenever a new block starts. 
One can reduce this effect by choosing a higher block-length at the expense of 
reducing the number of possible permutations of the data.

Another way to reduce this issue is to use a tapered-block bootstrap which is 
designed to mitigate artificial structural breaks introduced by the resampling 
procedure at the block-transition points.

Import the function
```python
from recombinator.tapered_block_bootstrap import tapered_block_bootstrap
```

Run the tapered-block bootstrap
```python
y_star_tbb \
    = tapered_block_bootstrap(y, 
                              block_length=b_star_cb, 
                              replications=B)
```

Again estimate the AR coefficient on each of the new bootstrap samples
```python
estimates_from_bootstrap = []
ar_estimates_from_bootstrap = np.zeros((B, ))

for b in range(B):
    y_bootstrap = y_star_tbb[b, :]
    ar_bootstrap = AR(y_bootstrap)
    estimate_from_bootstrap = ar_bootstrap.fit(maxlag=1)
    estimates_from_bootstrap.append(estimate_from_bootstrap)
    ar_estimates_from_bootstrap[b] = estimate_from_bootstrap.params[1]
```

... and plot the sampling distribution
```python
plt.hist(ar_estimates_from_bootstrap, bins=20)
print(f'mean={np.mean(ar_estimates_from_bootstrap)}')
print(f'median={np.median(ar_estimates_from_bootstrap)}')
```

The mean and median of the distribution are now much closer to the estimate from the original time series.

### Running Recombinator on a GPU
Resampling can be a computationally intensive task, which is highly parallelizable. 
The implementations of various algorithms fall into two broad categories:
* Loop-based (via Numba)
* Vectorized (via NumPy)

Vectorized implementations can be run on the GPU depending on the availability a 
GPU package with a NumPy compatible interface. 
To this end, vectorized implementations in Recombinator lets the user specify 
alternative modules and functions that are to be used internally.

A NumPy compatible package that supports both CUDA and OpenCL is Cocos 
available at https://github.com/michaelnowotny/cocos.

Using Cocos, resampling on the GPU using Recombinator can be performed as follows.

Import the NumPy-like Cocos package. This requires an ArrayFire installation, a compatible GPU, and the installation of Cocos.
```python
import cocos.numerics as cn
from cocos.device import gpu_sync_wrapper, sync
```

#### I.I.D. Boostrap
Generate an original sample to work with
```python
n = 100
cn.random.seed(1)
x_gpu = cn.absolute(cn.random.randn(n))
plt.hist(x_gpu);
```

Resample using an i.i.d. boostrap
```python
x_resampled_vectorized_gpu \
    = iid_bootstrap_vectorized(x_gpu, 
                               replications=R, 
                               randint=cn.random.randint)
sync()
```

In this case, GPU support is achieved by simply specifying an alternative 
implementation of NumPy's randint function.

#### Block-Based Bootstrap
Transfer the array created in the time-series example above to the GPU
```python
y_gpu = cn.array(y)
```

Run a circular-block boostrap on the GPU
```python
y_star_cb \
    = circular_block_bootstrap_vectorized(y_gpu, 
                                          block_length=b_star_cb, 
                                          replications=B, 
                                          replace=True, 
                                          num_pack=cn, 
                                          choice=cn.random.choice)
```

Estimate the AR coefficient on each of the bootstrap samples
```python
estimates_from_bootstrap = []
ar_estimates_from_bootstrap = np.zeros((len(y_star_cb), ))

for b in range(len(y_star_cb)):
    y_bootstrap = np.array(y_star_mb[b, :].squeeze())
    ar_bootstrap = AR(y_bootstrap)
    estimate_from_bootstrap = ar_bootstrap.fit(maxlag=1)
    estimates_from_bootstrap.append(estimate_from_bootstrap)
    ar_estimates_from_bootstrap[b] = estimate_from_bootstrap.params[1]
```

Plot the empirical sampling distribution and the compute its mean and median
```python
plt.hist(ar_estimates_from_bootstrap, bins=20)
print(f'mean={np.mean(ar_estimates_from_bootstrap)}')
print(f'median={np.median(ar_estimates_from_bootstrap)}')
```

For the block-based bootstrap, GPU support requires the specification of a 
function compatible with NumPy's random.choice as well as a package 
(num_pack) that is compatible with NumPy.

#### Cocos-Specific Convenience Functions
The module recombinator.bootstrap_cocos provides Cocos specific wrapper-functions 
for GPU enabled vectorized bootstrap procedures.