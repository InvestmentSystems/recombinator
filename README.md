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
<pre>
    pip install recombinator
</pre>
or 
<pre>
    pip3 install recombinator
</pre>
if not using Anaconda.

To get the latest version, clone the repository from github, 
open a terminal/command prompt, navigate to the root folder and install via
<pre>
    pip install .
</pre>
or 
<pre>
    pip3 install . 
</pre>
if not using Anaconda.

### Most Recent Version on GitHub
1. Clone the github repository via

<pre>
    git clone https://github.com/InvestmentSystems/recombinator.git
</pre>
    
2. Navigate to the recombinator base directory and run
<pre>
    pip install .
</pre> 
    
## Getting Started
Please see the Jupyter notebooks 'notebooks/Block Bootstrap.ipynb' and 'notebooks/IID Bootstrap.ipynb' for examples.
