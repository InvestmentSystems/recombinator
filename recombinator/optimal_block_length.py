# This module is a Python adaptation of Andrew Patten's Matlab implementation
# available at http://public.econ.duke.edu/~ap172/opt_block_length_REV_dec07.txt


# from dataclasses import dataclass
import numpy as np
import typing as tp


def mlag(x: np.ndarray,
         n: tp.Optional[int] = 1,
         init: tp.Optional[float] = 0.0) -> np.ndarray:
    """
    Purpose: generates a matrix of n lags from a matrix (or vector)
    containing a set of vectors (For use in var routines)

    Usage:     xlag = mlag(x,nlag)
    or: xlag1 = mlag(x), which defaults to 1-lag
    where: x is a nobs by nvar NumPy array

    Args:
        nlag = # of contiguous lags for each vector in x
        init = (optional) scalar value to feed initial missing values
                (default = 0)

    Returns: xlag = a matrix of lags (nobs x nvar*nlag)
    x1(t-1), x1(t-2), ... x1(t-nlag), x2(t-1), ... x2(t-nlag) ...


    original Matlab version written by:
    James P. LeSage, Dept of Economics
    University of Toledo
    2801 W. Bancroft St,
    Toledo, OH 43606
    jpl@jpl.econ.utoledo.edu

    Adapted for Python August 12, 2018 by Michael C. Nowotny
    """

    nobs, nvar = x.shape

    xlag = np.ones((nobs, nvar * n), dtype=x.dtype) * init
    icnt = 0
    for i in range(nvar):
        for j in range(n):
            xlag[j + 1:, icnt + j] = x[0:-j - 1, i]
        icnt += n

    return xlag


def lam(kk: np.ndarray) -> np.ndarray:
    """
    Helper function, calculates the flattop kernel weights.

    Adapted for Python August 12, 2018 by Michael C. Nowotny
    """
    return (np.abs(kk) >= 0) * (np.abs(kk) < 0.5) \
           + 2 * (1.0 - np.abs(kk)) * (np.abs(kk) >= 0.5) * (np.abs(kk) <= 1)


# @dataclass(frozen=True)
class OptimalBlockLength(tp.NamedTuple):
    b_star_sb: float  # optimal block length for stationary bootstrap
    b_star_cb: float  # optimal block length for circular block bootstrap


# ToDo: Add calculation of optimal block length for moving block bootstrap
# ToDo: Add calculation of optimal block length for tapered block bootstrap
def optimal_block_length(data: np.ndarray) -> tp.Sequence[OptimalBlockLength]:
    """
    This is a function to select the optimal (in the sense of minimising the MSE
    of the estimator of the long-run variance) block length for the stationary
    bootstrap or circular bootstrap.
    The code follows Politis and White, 2001,
    "Automatic Block-Length Selection for the Dependent Bootstrap".

        DECEMBER 2007: CORRECTED TO DEAL WITH ERROR IN LAHIRI'S PAPER, PUBLISHED
        BY NORDMAN IN THE ANNALS OF STATISTICS

        NOTE: The optimal average block length for the stationary bootstrap,
              and it does not need to be an integer.
              The optimal block length for the circular bootstrap should be an
              integer. Politis and White suggest rounding the output UP to the
              nearest integer.

     Args:
         data, an nxk matrix

     Returns: a 2xk NumPy array of optimal bootstrap block lengths,
              [[b_star_sb], [b_star_cb]], where
              b_star_sb: optimal block length for stationary bootstrap
              b_star_cb: optimal block length for circular bootstrap

    original Matlab version written by:
    Andrew Patton

    4 December, 2002
    Revised (to include CB): 13 January, 2003.

    Helpful suggestions for this code were received from
    Dimitris Politis and Kevin Sheppard.

    Modified 23.8.2003 by Kevin Sheppard for speed issues

    Adapted for Python August 12, 2018 by Michael C. Nowotny
    """

    if data.ndim == 1:
        data = data.reshape((-1, 1))
    elif data.ndim > 2:
        raise ValueError(
            'data must be a two dimensional NumPy array'
            '(number of observations x number of variables)')
    n, k = data.shape

    # these are optional in the original Matlab implementation
    # opt_block_length_full.m, but fixed at default values here
    kn = int(max(5, np.sqrt(np.log10(n))))

    # adding kn extra lags to employ Politis' (2002) suggestion
    # for finding largest significant m
    m_max = int(np.ceil(np.sqrt(n)) + kn)

    # maximum value of b_star_sb to consider.
    # dec07: new idea for rule-of-thumb to put an upper bound on estimated
    # optimal block length
    b_max = np.ceil(min(3 * np.sqrt(n), n / 3))

    c = 2
    original_data = data
    # b_star_final = np.zeros((2, k), dtype=np.float64)
    b_star_final = []

    for i in range(k):
        data = original_data[:, i].reshape((-1, 1))

        # FIRST STEP: finding m_hat-> the largest lag for which the
        # auto-correlation is still significant.
        temp = mlag(data, m_max)

        # dropping the first m_max rows, as they are filled with zeros
        temp = temp[m_max:, :]
        temp = np.corrcoef(np.hstack((data[m_max:], temp)), rowvar=False)
        temp = temp[1:, 0].reshape((-1, 1))

        # We follow the empirical rule suggested in
        # Politis, 2002, "Adaptive Bandwidth Choice".
        # as suggested in Remark 2.3, setting c=2, kn=5

        # looking at vectors of auto-correlations,
        # from lag m_hat to lag m_hat+kn
        temp2 = np.hstack((np.transpose(mlag(temp, kn)), temp[-kn:]))

        # dropping the first kn-1, as the vectors have empty cells
        temp2 = temp2[:, kn:]

        # checking which are less than the critical value
        temp2 = np.abs(temp2) < (c * np.sqrt(np.log10(n) / n)
                                 * np.ones((kn, m_max - kn + 1)))

        # this counts the number of insignificant autocorrelations
        temp2 = np.sum(temp2, axis=0).reshape((1, -1))
        temp3 = np.hstack((np.arange(1, temp2.shape[1] + 1).reshape((-1, 1)),
                           temp2.transpose()))

        # selecting all rows where ALL kn auto-correlations are not significant
        temp3 = temp3[np.squeeze(temp2 == kn), :]

        if temp3.size == 0:
            # this means that NO collection of kn auto-correlations were all
            # insignificant, so pick largest significant lag
            m_hat = max(
                np.flatnonzero(np.abs(temp) > (c * np.sqrt(np.log10(n) / n))))
        else:
            # if more than one collection is possible, choose the smallest m
            m_hat = temp3[0, 0]

        if 2 * m_hat > m_max:
            m = m_max
        else:
            m = 2 * m_hat

        del temp, temp2, temp3

        # SECOND STEP: computing the inputs to the function for b_star_sb
        kk = np.arange(-m, m + 1).reshape((-1, 1))
        if m > 0:
            temp = mlag(data, m)

            # dropping the first m_max rows, as they're filled with zeros
            temp = temp[m:, :]
            temp = np.cov(np.hstack((data[m:], temp)).transpose())

            # auto-covariances
            acv = temp[:, 0].reshape((-1, 1))
            acv2 = np.hstack(
                (-np.arange(1, m + 1).reshape((-1, 1)), acv[1:, :]))
            if acv2.shape[0] > 1:
                acv2 = acv2[acv2[:, 0].argsort(),]

            # auto-covariances from -m to m
            acv = np.vstack((acv2[:, 1].reshape((-1, 1)), acv))
            del acv2

            g_hat = np.sum(lam(kk / m) * np.abs(kk) * acv)
            dcb_hat = (4.0 / 3.0) * np.sum(lam(kk / m) * acv) ** 2

            # first part of dsb_hat (note cos(0)=1)
            dsb_hat = 2 * (np.sum(lam(kk / m) * acv) ** 2)

            # FINAL STEP: constructing the optimal block length estimator

            # optimal block lenght for stationary bootstrap
            b_star_sb = ((2 * (g_hat ** 2) / dsb_hat) ** (1.0 / 3.0)) \
                     * (n ** (1.0 / 3.0))
            if b_star_sb > b_max:
                b_star_sb = b_max

            # optimal block length for circular bootstrap
            b_star_cb = ((2 * (g_hat ** 2) / dcb_hat) ** (1.0 / 3.0)) \
                        * (n ** (1.0 / 3.0))
            if b_star_cb > b_max:
                b_star_cb = b_max

            # b_star = (b_star_sb, b_star_cb)
            b_star = OptimalBlockLength(b_star_sb=b_star_sb,
                                        b_star_cb=b_star_cb)
        else:
            b_star = OptimalBlockLength(b_star_sb=1.0, b_star_cb=1.0)

        b_star_final.append(b_star)

    return tuple(b_star_final)
