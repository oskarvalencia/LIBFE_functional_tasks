# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:17:13 2023

@author: 
https://nbviewer.org/github/demotu/BMC/blob/master/notebooks/Stabilography.ipynb
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading COP data from a CSV file hosted on GitHub
CoPOA = pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/CoP_signals/G4_BMPOST_OA.csv', sep=',')

# Extracting relevant columns from the DataFrame
Co1 = CoPOA.iloc[:, 1:3].values

# Displaying the imported COP data
CoPOA

#%%

# Reordering columns to move 'CoPy' and 'CoPx' to the end
cols_to_move = ['COPy', 'COPx']
new_cols = np.hstack((CoPOA.columns.difference(cols_to_move), cols_to_move))
df = CoPOA.loc[:, new_cols]

# Scaling and converting data to centimeters
COP1 = (df.iloc[:, 1:3].values) * 10

# Displaying the reordered and scaled COP data
COP1

#%%

def cop_plot(freq, COP, units='cm'):
    '''
    Plot COP data from postural sway measurement.
    '''
    import matplotlib.gridspec as gridspec
    
    # Generating time vector
    t = np.linspace(0, COP.shape[0] / freq, COP.shape[0])
    
    # Setting up plot parameters
    plt.rc('axes', labelsize=16, titlesize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    # Creating a figure with two subplots
    plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
    ax1 = plt.subplot(gs[0])
    
    # Plotting COP data on the stabilogram
    ax1.plot(t, COP[:, 0], lw=2, color=[0, 0, 1, 1], label='ap')
    ax1.plot(t, COP[:, 1], lw=2, color=[1, 0, 0, 1], label='ml')
    ax1.set_xlim([t[0], t[-1]])
    ax1.grid()
    ax1.locator_params(axis='both', nbins=5)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(f'COP [{units}]')
    ax1.set_title('Stabilogram')
    ax1.legend(fontsize=12, loc='best', framealpha=.5)
    
    ax2 = plt.subplot(gs[1])
    
    # Plotting COP data on the statokinesigram
    ax2.plot(COP[:, 1], COP[:, 0], lw=2, color='g')
    ax2.set_xlabel(f'COP ml [{units}]')
    ax2.set_ylabel(f'COP ap [{units}]')
    ax2.set_title('Statokinesigram')
    
    # Optional: Plotting axes with the same colors as COP data
    if 0:
        ax2.xaxis.label.set_color('red')
        ax2.spines['bottom'].set_color('red')
        ax2.tick_params(axis='x', colors='red')
        ax2.yaxis.label.set_color('blue')
        ax2.spines['left'].set_color('blue')
        ax2.tick_params(axis='y', colors='blue')
    
    ax2.grid()
    ax2.locator_params(axis='both', nbins=5)
    
    # Adjusting layout for a cleaner presentation
    plt.tight_layout()
    plt.show()

#%%

# Setting the frequency and plotting COP data
fr = 100
cop_plot(fr, Co1)  # Plotting COP data

#%%

# Calculating and displaying measurements of spatial variability
m = np.mean(COP1, axis=0)  # mean
sd = np.std(COP1, axis=0)  # standard deviation
rms = np.sqrt(np.mean(COP1 ** 2, axis=0))  # root-mean square
rang = np.max(COP1, axis=0) - np.min(COP1, axis=0)  # range (maximum - minimum)
tpath = np.sum(np.abs(np.diff(COP1, axis=0)), axis=0)  # total path (length of the COP displacement)

unit = 'cm'
print('Measurements of spatial variability')
print('{0:12} {1:^16}'.format('Variable', 'Direction'))
print('{0:12} {1:^8} {2:^5}'.format('', 'ap', 'ml'))
print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('Mean:', m[0], m[1], unit))
print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('SD:', sd[0], sd[1], unit))
print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('RMS:', rms[0], rms[1], unit))
print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('Range:', rang[0], rang[1], unit))
print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('Total path:', tpath[0], tpath[1], unit))

#%%

# Calculating mean velocity and mean resultant velocity
mvel = np.sum(np.abs(np.diff(COP1, axis=0)), axis=0) / 20  # seconds
mvelr = np.sum(np.abs(np.sqrt(np.sum(np.diff(COP1, axis=0) ** 2, axis=1))), axis=0) / 20  # seconds

# Displaying mean velocity and mean resultant velocity
print('{0:15} {1:^16}'.format('Variable', 'Direction'))
print('{0:15} {1:^8} {2:^5}'.format('', 'ap', 'ml'))
print('{0:15} {1:>6.2f} {2:>6.2f} {3:>5}'.format('Mean velocity:', mvel[0], mvel[1], unit + '/s'))
print('')
print('{0:22} {1:>6.2f} {2:>5}'.format('Mean resultant velocity:', mvelr, unit + '/s'))

#%%

from __future__ import division, print_function
import numpy as np
# %load './../functions/hyperellipsoid.py'
"""Prediction hyperellipsoid for multivariate data."""



__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = "1.0.3"
__license__ = "MIT"


def hyperellipsoid(P, y=None, z=None, pvalue=.95, units=None, show=True, ax=None):
    """
    Prediction hyperellipsoid for multivariate data.

    The hyperellipsoid is a prediction interval for a sample of a multivariate
    random variable and is such that there is pvalue*100% of probability that a
    new observation will be contained inside the hyperellipsoid [1]_.  
    The hyperellipsoid is also a tolerance region such that the average or
    expected value of the proportion of the population contained in this region
    is exactly pvalue*100% (called Type 2 tolerance region by Chew (1966) [1]_).

    The directions and lengths of the semi-axes are found, respectively, as the
    eigenvectors and eigenvalues of the covariance matrix of the data using
    the concept of principal components analysis (PCA) [2]_ or singular value
    decomposition (SVD) [3]_ and the length of the semi-axes are adjusted to
    account for the necessary prediction probability.

    The volume of the hyperellipsoid is calculated with the same equation for
    the volume of a n-dimensional ball [4]_ with the radius replaced by the
    semi-axes of the hyperellipsoid.

    This function calculates the prediction hyperellipsoid for the data,
    which is considered a (finite) sample of a multivariate random variable
    with normal distribution (i.e., the F distribution is used and not
    the approximation by the chi-square distribution).

    Parameters
    ----------
    P : 1-D or 2-D array_like
        For a 1-D array, P is the abscissa values of the [x,y] or [x,y,z] data.
        For a 2-D array, P is the joined values of the multivariate data.
        The shape of the 2-D array should be (n, p) where n is the number of
        observations (rows) and p the number of dimensions (columns).
    y : 1-D array_like, optional (default = None)
        Ordinate values of the [x, y] or [x, y, z] data.
    z : 1-D array_like, optional (default = None)
        Ordinate values of the [x, y] or [x, y, z] data.
    pvalue : float, optional (default = .95)
        Desired prediction probability of the hyperellipsoid.
    units : str, optional (default = None)
        Units of the input data.
    show : bool, optional (default = True)
        True (1) plots data in a matplotlib figure, False (0) to not plot.
        Only the results for p=2 (ellipse) or p=3 (ellipsoid) will be plotted.
    ax : a matplotlib.axes.Axes instance (default = None)

    Returns
    -------
    hypervolume : float
        Hypervolume (e.g., area of the ellipse or volume of the ellipsoid).
    axes : 1-D array
        Lengths of the semi-axes hyperellipsoid (largest first).
    angles : 1-D array
        Angles of the semi-axes hyperellipsoid (only for 2D or 3D data).
        For the ellipsoid (3D data), the angles are the Euler angles
        calculated in the XYZ sequence.
    center : 1-D array
        Centroid of the hyperellipsoid.
    rotation : 2-D array
        Rotation matrix for hyperellipsoid semi-axes (only for 2D or 3D data).

    References
    ----------
    .. [1] http://www.jstor.org/stable/2282774
    .. [2] http://en.wikipedia.org/wiki/Principal_component_analysis
    .. [3] http://en.wikipedia.org/wiki/Singular_value_decomposition
    .. [4] http://en.wikipedia.org/wiki/Volume_of_an_n-ball

    Examples
    --------
    >>> from hyperellipsoid import hyperellipsoid
    >>> y = np.cumsum(np.random.randn(3000)) / 50
    >>> x = np.cumsum(np.random.randn(3000)) / 100
    >>> area, axes, angles, center, R = hyperellipsoid(x, y, units='cm')
    >>> print('Area =', area)
    >>> print('Semi-axes =', axes)
    >>> print('Angles =', angles)
    >>> print('Center =', center)
    >>> print('Rotation matrix =\n', R)

    >>> P = np.random.randn(1000, 3)
    >>> P[:, 2] = P[:, 2] + P[:, 1]*.5
    >>> P[:, 1] = P[:, 1] + P[:, 0]*.5
    >>> volume, axes, angles, center, R = hyperellipsoid(P, units='cm')
    """

    from scipy.stats import f as F
    from scipy.special import gamma

    P = np.array(P, ndmin=2, dtype=float)
    if P.shape[0] == 1:
        P = P.T
    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=float)
        if y.shape[0] == 1:
            y = y.T
        P = np.concatenate((P, y), axis=1)
    if z is not None:
        z = np.array(z, copy=False, ndmin=2, dtype=float)
        if z.shape[0] == 1:
            z = z.T
        P = np.concatenate((P, z), axis=1)
    # covariance matrix
    cov = np.cov(P, rowvar=0)
    # singular value decomposition
    U, s, Vt = np.linalg.svd(cov)
    p, n = s.size, P.shape[0]
    # F percent point function
    fppf = F.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)
    # semi-axes (largest first)
    saxes = np.sqrt(s*fppf)
    hypervolume = np.pi**(p/2)/gamma(p/2+1)*np.prod(saxes)
    # rotation matrix
    if p == 2 or p == 3:
        R = Vt
        if s.size == 2:
            angles = np.array([np.rad2deg(np.arctan2(R[1, 0], R[0, 0])),
                               90-np.rad2deg(np.arctan2(R[1, 0], -R[0, 0]))])
        else:
            angles = rotXYZ(R, unit='deg')
        # centroid of the hyperellipsoid
        center = np.mean(P, axis=0)
    else:
        R, angles = None, None

    if show and (p == 2 or p == 3):
        _plot(P, hypervolume, saxes, center, R, pvalue, units, ax)

    return hypervolume, saxes, angles, center, R


def _plot(P, hypervolume, saxes, center, R, pvalue, units, ax):
    """Plot results of the hyperellipsoid function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        # code based on https://github.com/minillinim/ellipsoid:
        # parametric equations
        u = np.linspace(0, 2*np.pi, 100)
        if saxes.size == 2:
            x = saxes[0]*np.cos(u)
            y = saxes[1]*np.sin(u)
            # rotate data
            for i in range(len(x)):
                [x[i], y[i]] = np.dot([x[i], y[i]], R) + center
        else:
            v = np.linspace(0, np.pi, 100)
            x = saxes[0]*np.outer(np.cos(u), np.sin(v))
            y = saxes[1]*np.outer(np.sin(u), np.sin(v))
            z = saxes[2]*np.outer(np.ones_like(u), np.cos(v))
            # rotate data
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], R) + center

        if saxes.size == 2:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # plot raw data
            ax.plot(P[:, 0], P[:, 1], '.-', color=[0, 0, 1, .5])
            # plot ellipse
            ax.plot(x, y, color=[0, 1, 0, 1], linewidth=2)
            # plot axes
            for i in range(saxes.size):
                # rotate axes
                a = np.dot(np.diag(saxes)[i], R).reshape(2, 1)
                # points for the axes extremities
                a = np.dot(a, np.array([-1, 1], ndmin=2))+center.reshape(2, 1)
                ax.plot(a[0], a[1], color=[1, 0, 0, .6], linewidth=2)
                ax.text(a[0, 1], a[1, 1], '%d' % (i + 1),
                        fontsize=20, color='r')
            plt.axis('equal')
            plt.grid()
            title = r'Prediction ellipse (p=%4.2f): Area=' % pvalue
            if units is not None:
                units2 = ' [%s]' % units
                units = units + r'$^2$'
                title = title + r'%.2f %s' % (hypervolume, units)
            else:
                units2 = ''
                title = title + r'%.2f' % hypervolume
        else:
            from mpl_toolkits.mplot3d import Axes3D
            if ax is None:
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_axes([0, 0, 1, 1], projection='3d')
            ax.view_init(20, 30)
            # plot raw data
            ax.plot(P[:, 0], P[:, 1], P[:, 2], '.-', color=[0, 0, 1, .4])
            # plot ellipsoid
            ax.plot_surface(x, y, z, rstride=5, cstride=5, color=[0, 1, 0, .1],
                            linewidth=1, edgecolor=[.1, .9, .1, .4])
            # ax.plot_wireframe(x, y, z, color=[0, 1, 0, .5], linewidth=1)
            #                  rstride=3, cstride=3, edgecolor=[0, 1, 0, .5])
            # plot axes
            for i in range(saxes.size):
                # rotate axes
                a = np.dot(np.diag(saxes)[i], R).reshape(3, 1)
                # points for the axes extremities
                a = np.dot(a, np.array([-1, 1], ndmin=2))+center.reshape(3, 1)
                ax.plot(a[0], a[1], a[2], color=[1, 0, 0, .6], linewidth=2)
                ax.text(a[0, 1], a[1, 1], a[2, 1], '%d' % (i+1),
                        fontsize=20, color='r')
            lims = [np.min([P.min(), x.min(), y.min(), z.min()]),
                    np.max([P.max(), x.max(), y.max(), z.max()])]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_zlim(lims)
            title = r'Prediction ellipsoid (p=%4.2f): Volume=' % pvalue
            if units is not None:
                units2 = ' [%s]' % units
                units = units + r'$^3$'
                title = title + r'%.2f %s' % (hypervolume, units)
            else:
                units2 = ''
                title = title + r'%.2f' % hypervolume
            ax.set_zlabel('Z' + units2, fontsize=18)

        ax.set_xlabel('X' + units2, fontsize=18)
        ax.set_ylabel('Y' + units2, fontsize=18)
        plt.title(title)
        plt.show()

        return ax


def rotXYZ(R, unit='deg'):
    """ Compute Euler angles from matrix R using XYZ sequence."""

    angles = np.zeros(3)
    angles[0] = np.arctan2(R[2, 1], R[2, 2])
    angles[1] = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    angles[2] = np.arctan2(R[1, 0], R[0, 0])

    if unit[:3].lower() == 'deg':  # convert from rad to degree
        angles = np.rad2deg(angles)

    return angles
#%%

Ar, axes, angles, center, R = hyperellipsoid(COP1[:, 1], COP1[:, 0], units='cm')
