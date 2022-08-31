import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Gaussian KDE
def bounded_gaussian_kde(samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[0], x, [1]])
    y = np.concatenate([[0], y, [0]])
    return x, y

#====

#Multimodal Beta Random Sample
# All inputs except size np.array of size N, where N is the number of mode
# weight = importance of each mode, higher = more probability. Must sum up to 1
# alpha = alpha parameter of unimodal beta distribution, >0
# beta = beta parameter of unimodal beta distribution, >0
# size = the output size of iid sample. tuple
#Return sample of size = size
def rand_multimodal_beta(weight, alpha, beta, size = 1):
    N_MODE = weight.size
    indicator = np.random.default_rng().choice(N_MODE, p = weight, size = size)
    return np.random.beta(a=alpha[indicator], b=beta[indicator])

#====

#Find critical load position for moment of a simply supported beam by moving the load
#Less clever than analytically solving it. May want to look into analytic solution instead (e.g. Megson 2005)
#Also try to figure out how to do this without for loops.
#Input:
#axle_load: np.array(N_VEHICLE, N_AXLE) of Axle Load
#axle_spacing: np.array(N_VEHICLE, N_AXLE) of Spacing between axles. Note that the first column should be zero.
#span: np.array(n, ) of different bridge spans
#Output: 
#
def crit_mom_load_pos(axle_load, axle_spacing, span, n_move = 100, npts = 100):
    #Preallocate result
    n_row = axle_load.shape[0]
    n_col = span.shape[0]
    crit_load_pos = np.zeros((n_row, n_col))
    crit_mom = np.zeros((n_row, n_col))
    
    #We will move the load by n_move amount from left support to end by n_move amount
    
    load_pos = np.linspace(-span, span, n_move)
    mom_array = np.zeros((n_row, n_move, n_col))
    
    init_axle_pos = np.cumsum(axle_spacing, axis=1)
    
    for i, current_span in enumerate(span):
        beam_pts = np.linspace(1/(npts+1), current_span - (1/(npts+1)), npts)
        
        for j, move_dist in enumerate(load_pos[:,i]):
            #Moments along the beam
            #M = f1 + f2 #(left + right of point of interest)
            #dim 1 : number of samples
            #dim 2 : number of beam point division
            #To the left of the point being considered
            f1 = np.zeros((n_row, npts))
            #To the right of the point being considered
            f2 = np.zeros((n_row, npts))
            for k in range(0, npts):
                #indicator_f1[:,:,k] = 
                indicator = (init_axle_pos+move_dist >=0) & (init_axle_pos+move_dist <= beam_pts[k])
                f1[:, k] = np.sum( ((init_axle_pos+move_dist) - (init_axle_pos+move_dist) * beam_pts[k] / current_span) * axle_load * indicator, axis = 1)
                indicator = (init_axle_pos+move_dist <=current_span) & (init_axle_pos+move_dist > beam_pts[k])
                f2[:, k] = np.sum( (beam_pts[k] - ((init_axle_pos+move_dist) * beam_pts[k] / current_span)) * axle_load * indicator, axis = 1)
            
            #Find critical moment and its load position
            mom_array[:, j, i] = np.max(f1+f2, axis = 1)
            
    crit_mom = np.max(mom_array, axis = 1)
    crit_load_pos = load_pos[np.argmax(mom_array, axis = 1)]
    return crit_load_pos, crit_mom

def ecdf(data):
    #Produce ranked results
    #Treatment separated columnwise
    x_ecdf = np.sort(data, axis = 0)
    n = x_ecdf.shape[0]
    F = np.arange(1, n+1) / float(n+1) #F = i/(n+1)
    return x_ecdf, F

def retake_block_max(data, period):
    #Cutoff data at the end
    data = data[0: data.size - (data.size%period)]
    
    tmp = np.reshape(data, (int(data.size/period), period))
    return np.max(tmp, axis = 1)

def retake_block_max_multi(data, period):
    result = np.zeros((int(data.shape[0]/period), data.shape[1]))
    for i in range (0, data.shape[1]):
        result[:, i] = retake_block_max(data[:,i], period)
    return result

def GEV_CDF(x, mu, sigma, xi):
    if np.all(xi != 0):
        tx = (1 + xi*((x-mu)/sigma)) ** (-1/xi)
    else:
        return "NaN"
        #tx = np.exp(-(x-mu)/sigma)
    return np.exp(-tx)
def inv_GEV_CDF(F, mu, sigma, xi):
    if np.all(xi < 0):
        return sigma/xi * ( (-np.log(F))**(-xi) - 1) + mu
    else:
        return "NaN"
    
def ols_regression(x, y):
    model = sm.OLS(y,x)
    results = model.fit()
    params = results.params
    se = results.bse
    return params, se

def plot_ols(est_params, x, y, x_plot):
    fig, ax = plt.subplots(2)
    ax[0].plot(x_plot, y);
    ax[0].plot(x_plot, np.sum(est_params * x, axis = 1));
    ax[1].plot(x_plot, np.sum(est_params * x, axis = 1) - y);
    ax[1].axhline(y = 0, color = 'k');
    return fig, ax

#===
# AESARA \xi SOLVER
#

from scipy import optimize
from scipy import special
import aesara.tensor as at
from aesara.graph.op import Op
from aesara import function

def func_xi(xi, z_bc):
    if np.any(xi>=0):
        if isinstance(xi, float):
            return 999999
        else:
            return np.ones(xi.shape) * 999999
    g1 = special.gamma(1-xi)
    g2 = special.gamma(1-2*xi)
    value = np.log(g1)-0.5*np.log(g2-g1**2)-np.log(z_bc)
    return value
def fprime_xi(xi, z_bc):
    g1 = special.gamma(1-xi)
    g2 = special.gamma(1-2*xi)
    g1_prime = g1*special.digamma(1-xi)
    g2_prime = g2*special.digamma(1-2*xi)
    try:
        fprime = -special.digamma(1-xi) - (-g2_prime + g1*g1_prime) /(g2-g1*g1)
    except:
        print(xi)
        print(g1)
        print(g2)
    return fprime
def xi_from_zbc(z_bc, init_guess):
    return optimize.newton(func_xi, init_guess, fprime = fprime_xi, args=(z_bc,))

at_x = at.dvector('at_x')
at_z = at.switch(at.le(at_x,1e1), -1e-1, 
                 at.switch(at.le(at_x,1e2), -1e-2,
                          at.switch(at.le(at_x,1e3), -1e-3, 
                                    at.switch(at.le(at_x,1e4), -1e-4, 
                                             at.switch(at.le(at_x,1e5), -1e-5,
                                                       at.switch(at.le(at_x,1e6), -1e-6,
                                                                 at.switch(at.le(at_x,1e7), -1e-7, -1e-8)
                                                                )
                                                      )
                                             )
                                   )
                          )
                )
init_guess = function([at_x], at_z)

class XiFromZbc(Op):
    itypes = [at.dvector]
    otypes = [at.dvector]
    
    def perform(self, node, inputs, outputs):
        z_bc, = inputs
        xi = xi_from_zbc(z_bc, init_guess(z_bc))
        outputs[0][0] = np.array(xi)
        
    def grad(self, inputs, g):
        z_bc, = inputs
        xi = self(z_bc)
        g1 = at.gamma(1-xi)
        g2 = at.gamma(1-2*xi)
        dg1 = at.digamma(1-xi)
        dg2 = at.digamma(1-2*xi)
        
        nom = at.pow(g2-at.pow(g1, 2), 1.5)
        denom = g1*g2*(dg2 - dg1)
        
        return [g[0] * nom/denom]
    
at_xi_from_zbc = XiFromZbc()

#=======================================
#         genmaxima() class
#=======================================
#Generic Maxima Distribution
#Distribution of a maxima of some generic parent distribution, given a block size
#Theoretically this converge to a GEV given large enough block size (N)
#However this should give precise values for any block size, with some exceptions:
# 1. All statistical moments are numerical computed, so there will be some computational error
# 2. GEV equivalent parameterisation and CDF comaparison assumes convergence to GEV, and support ONLY Weibull domain of attraction
#
#TO DO:
# 1. Some scipy equivalent methods are not coded yet (e.g. entropy, median)
# 2. Some sort of hypothesis testing to check if the genmaxima distribution has converged to a GEV distribution
#
# NEED A LOT MORE TESTING

from scipy import integrate
from scipy.stats import uniform
from scipy.stats import genextreme

class genmaxima():
    r"""Class for generic maxima random variable
    
    Arguments:
    parent: 
        The parent distribution. You must specify all arguments for the parent at initialisation, as it cannot be changed later on.
    N:
        The block size. inf > N > 0
        Default to 1. Can be changed and set via set_N() method.
    
    """
    
    #Helper methods since not using rv_continuous class from scipy
    def __init__(self, parent, N = 1.0):
        self.N = N
        self.parent = parent
        
    def set_N(self, N):
        self.N = N
        
    def get_N(self):
        return self.N
    
    def set_parent(self, parent):
        self.parent = parent
    
    def get_parent(self):
        return self.parent
    
    # Methods similar to scipy, where possible
    def argcheck(self):
        return (self.N > 0) and (self.N < np.inf) and (np.isreal(self.N))
    
    def support(self):
        #Alias for get_support() method
        return self.get_support()
    
    def get_support(self):
        #Should be the same as the parent, although need more study to verify
        return self.parent.support()
    
    def rvs(self, size=1, random_state=None):
        """
        Generate random number. Follows scipy arguments.
        """
        u = uniform.rvs(size = size, random_state = random_state)
        return self.ppf(u)
    
    def pdf(self, x):
        return self.N * (self.parent.cdf(x) ** (self.N - 1)) * self.parent.pdf(x)
    
    def logpdf(self, x):
        return np.log(self.N) + (self.N - 1) * self.parent.logcdf(x) + self.parent.logpdf(x)
    
    def cdf(self, x):
        return self.parent.cdf(x) ** self.N
    
    def logcdf(self, x):
        return self.N * self.parent.logcdf(x)
    
    def sf(self, x):
        return 1-self.cdf(x)
    
    def logsf(self, x):
        return np.log(self.sf(x))
    
    def ppf(self, q):
        return self.parent.ppf(q ** (1/self.N))
    
    def isf(self, q):
        #return self.parent.ppf(1 - (q ** (1/self.N)))  <===== NEED TO CHECK IF CORRECT
        print("NOT CODED YET")
        return "nan"
    
    def entropy(self):
        print("NOT CODED YET")
        return "nan"
    
    def fit(self, data):
        print("NOT CODED YET")
        return "nan"
    
    def interval(self, confidence):
        print("NOT CODED YET")
        return "nan"
    
    def expect(self, args=(), lb=None, ub=None, conditional=False, **kwds):
        print("NOT CODED YET")
        return "nan"
    
    #Methods to calculate statistical moments
    #All moments are calculated numerically, so there will be some numerical error
    
    #Method to calculate non-central moment
    def moment(self, order):
        #function x^k * f(x), for uncentered moment calculations
        def xk_fx(x):
            return (x**order) * self.pdf(x)
        #Integrate via numerical method
        support = self.get_support()
        return integrate.quad(func = xk_fx, a = support[0], b = support[1])[0]
    
    #Method to calculate standardised moment
    #Unique to this, not availabel in scipy. Maybe better to code the expect() method for this
    def std_moment(self, order):
        first_moment = self.moment(order=1)
        if order==1:
            return first_moment
        else:
            #shifted and translated, for standardised moment calculations
            def std_xk_fx(x, translation, scale):
                return (((x-translation)/scale) ** order) * self.pdf(x)
            support = self.get_support()
            second_centered_moment = integrate.quad(func = std_xk_fx, a = support[0], b = support[1], args = (first_moment, 1))[0]
            if order==2:
                return second_centered_moment
            else:
                return integrate.quad(func = std_xk_fx, a = support[0], b = support[1], args = (first_moment, second_centered_moment))[0]
            
    #Method to get statistics, similar to stats() method in scipy distribution
    #Does not behave exactly the same as scipy. Need to code properly.
    def stats(self, moments):
        moments = list(moments)
        for a in moments:
            if a == 'm':
                return self.moment(order = 1)
            elif a == 'v':
                return self.std_moment(order = 2)
            elif a == 's':
                return self.std_moment(order = 3)
            elif a == 'k':
                return self.std_moment(order = 4)
            else:
                print("Unrecognised 'moments' argument. Should be 'm', 'v', 's', 'k', or combinations of those. See scipy doc for detail")
    
    def mean(self):
        return self.moment(order = 1)
    
    def var(self):
        return self.std_moment(order = 2)
    
    def std(self):
        return np.sqrt(self.var())
    
    def median(self):
        print("NOT CODED YET")
        return "nan"
    
    def get_genextreme_parameter(self, parameterisation = "coles"):
        """
        Calculate the "equivalent" GEV parameters
        Based on the mean, variance and upper bound support
        
        **Default return using Coles (2001) parameterisation**
        **Change parameterisation argument to "scipy" to use scipy parameterisation**
        
        For Coles (2001) parameterisation, either see the original textbook, or see Wikipedia on Generalised Extreme Value distribution
        scipy parameterisation has the xi parameter flipped in its sign.
        
        **This method is currently only supported for upper bounded parent distributions**
        **And assumes parent has Weibull domain of attraction**
        
        Will return "nan" for non upper bounded distribution.
        It is your responsibility to check the domain of attraction.
        The method will happily return some numbers even if the domain of attraction is incorrect.
        """
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            dist_mean = self.mean()
            dist_std = self.std()
            dist_zbc = (upper_bound - dist_mean)/dist_std #centered upper bound
            
            xi = xi_from_zbc(dist_zbc, init_guess([z_bc]))[0];
            g1 = special.gamma(1-xi)
            kx = (dist_mean - upper_bound)/g1
            sigma = kx*xi
            mu = dist_mean - kx*(g1-1)
            
            if parameterisation == "coles":
                return (mu, sigma, xi);
            elif parameterisation == "scipy":
                return (mu, sigma, -xi);
            else:
                print ("Unsupported GEV parameterisation. Either 'coles' or 'scipy'")
                return ("nan", "nan", "nan");
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            return ("nan", "nan", "nan");
    
    def compare_genextreme_cdf(self, x, scale='linear', custom_scale_fn=None):
        """
        Method to compare CDF to GenExtreme (gev) CDF
        Compare pointwise, given the point(s) in argument 'x'
        
        Return the difference between GEV CDF and genmaxima CDF, i.e.
        return F_{GEV}(x) - F_{genmaxima}(x)
        
        **This method is currently only supported for upper bounded parent distributions**
        **And assumes parent has Weibull domain of attraction**
        
        Will return "nan" for non upper bounded distribution.
        It is your responsibility to check the domain of attraction.
        The method will happily return some numbers even if the domain of attraction is incorrect.
        
        
        Arguments:
        x:
            Points along which the CDF will be compared
        scale:
            The scale in which the distribution will be compared. Supported are:
                + 'linear': comparison of CDF in linear scale (no transformation)
                + 'gumbel': comparison in gumbel probability plot scale, i.e. -ln(-ln(CDF))
                + 'custom': provide your own via the custom_scale_fn argument
        custom_scale_fn:
            The custom scaling function. You MUST SET the 'scale' argument for this function to be used.
        """
        
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            #Get the GEV parameters
            gev_param = self.get_genextreme_parameter(parameterisation='scipy')
            if scale == 'linear':
                return genextreme(loc = gev_param[0], scale = gev_param[1], c = gev_param[2]).cdf(x) - self.cdf(x)
            elif scale == 'gumbel':
                return -np.log(-np.log(genextreme(loc = gev_param[0], scale = gev_param[1], c = gev_param[2]).cdf(x))) - -np.log(-np.log(self.cdf(x)))
            elif scale == 'custom':
                return custom_scale_fn(genextreme(loc = gev_param[0], scale = gev_param[1], c = gev_param[2]).cdf(x)) - custom_scale_fn(self.cdf(x))
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            return "nan";