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
# See for reference: https://docs.pymc.io/en/v3/Advanced_usage_of_Theano_in_PyMC3.html
# Reference still applies for pymc4, just use aesara instead of theano
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
def xi_from_zbc(z_bc, init_guess, **kwargs):
    return optimize.newton(func_xi, init_guess, fprime = fprime_xi, args=(z_bc,), **kwargs)

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
    
#JAX implementation
#Ref: https://aesara.readthedocs.io/en/latest/extending/creating_a_numba_jax_op.html
#NOT WORKING!

"""
from aesara.link.jax.dispatch import jax_funcify
#import jax.scipy.special as jss
import jax.lax as jlax

@jax_funcify.register(XiFromZbc)
def jax_funcify_XiFromZbc(op, node, storage_map, **kwargs):
    itypes = op.itypes
    otypes = op.otypes
    
    def perform(self, node, inputs, outputs):
        zbc, = inputs
        xi = xi_from_zbc(zbc, init_guess(z_bc))
        outputs[0][0] = np.array(xi)
        
    def grad(self, inputs, g):
        z_bc, = inputs
        xi = self(z_bc)
        g1 = jlax.exp(jlax.lgamma(1-xi))
        g2 = jlax.exp(jlax.lgamma(1-2*xi))
        dg1 = jlax.digamma(1-xi)
        dg2 = jlax.digamma(1-2*xi)
        
        nom = jlax.pow(g2-jlax.pow(g1, 2), 1.5)
        denom = g1*g2*(dg2 - dg1)
        
        return [g[0] * nom/denom]
"""
    
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
# IMPORTANT: COME UP WITH A BETTER NUMERICAL INTEGRATION. Its currently producing inaccurate results at large N
# ROMBERG INTEGRATION ISNT WORKING. NEED TO REDO
# 1. Some scipy equivalent methods are not coded yet (e.g. entropy, median)
# 2. Some sort of hypothesis testing to check if the genmaxima distribution has converged to a GEV distribution
#
# NEED A LOT MORE TESTING

from scipy import integrate
from scipy.stats import uniform
from scipy.stats import genextreme

class genmaxima():
    r"""Class for generic maxima random variable
    
    **Critical Note**: Numerical instability at large and 0 < N << 1 has been observed. The root cause is due to errors from numerical integration that are used
    to calculate the statistical moments. Please use with caution!. Any suggestions to improve this is more than welcomed!
    
    To circumvent this, it is recommended to convert this distribution to a genxtreme_WR object via get_genxtreme_WR() method. From there, you can
    proceed all statistical analysis again assuming that your distribution has converged. **Note this is only available for parent distributions**
    **with bounded upper bound and Weibull domain of attraction**. It is your responsibility to check the domain of attraction and convergence to GEV distribution
    
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
    
    def rvs(self, size=1, random_state=None, method="direct", progress_bar=None, max_mem = int(1e8)):
        
        """
        Generate random number. Follows scipy arguments.
        Two different sampling methods, set via 'method' argument:
        1. 'direct': Directly sample from the genmaxima distribution
        2. 'parent': Sample from parent, then take block maxima. Only available for N is integer and N >= 1 
        """
        
        if method == 'direct':
            u = uniform.rvs(size = size, random_state = random_state)
            return self.ppf(u)
        
        elif method == 'parent':
            #Check that N>=1
            if (self.N < 1):
                print("Invalid N for sampling method 'parent'. N must be equal or larger than 1 to use 'parent' sampling method")
                return "nan"
            if not isinstance(self.N, int):
                print("Invalid N for sampling method 'parent'. N must be an integer ('int' instance)")
                return "nan"
            
            max_mem = int(max_mem)
            
            #Determine if we can sample all at once, or need to be looped
            if (self.N * size) > max_mem:
                #Loop sample
                out_arr = np.zeros(size)
                #Determine max number of parallel sampling
                max_parallel_dim = max(int(max_mem/self.N),1)
                #Calculate the amount of loops needed
                num_loops = int(np.ceil(size/max_parallel_dim))
                
                if (progress_bar == False) or (num_loops < 2):
                    for i in range(0,num_loops):
                        #Calculate index to store
                        start_idx = i*max_parallel_dim
                        end_idx = np.min(((i+1)*max_parallel_dim, size))
                        num_parallel_dim = end_idx-start_idx
                        #Sample
                        out_arr[start_idx:end_idx] = np.max(self.parent.rvs(size=(self.N, num_parallel_dim), 
                                                                            random_state=random_state), 
                                                            axis = 0)
                else:
                    from tqdm import tqdm
                    for i in tqdm(range(0,num_loops)):
                        #Calculate index to store
                        start_idx = i*max_parallel_dim
                        end_idx = np.min(((i+1)*max_parallel_dim, size))
                        num_parallel_dim = end_idx-start_idx
                        #Sample
                        out_arr[start_idx:end_idx] = np.max(self.parent.rvs(size=(self.N, num_parallel_dim), 
                                                                            random_state=random_state), 
                                                            axis = 0)
            else:
                out_arr = retake_block_max(self.parent.rvs(size=(self.N*size), random_state=random_state), self.N)
                
            return out_arr
        else:
            print("Invalid 'method' argument specified. Must be either 'direct' or 'parent'")
            return "nan"
            
    
    def pdf(self, x):
        #Assume that if return nan, the pdf is zero
        parent_F = self.parent.cdf(x)
        t2 = np.power(parent_F, (self.N - 1), out = np.zeros_like(x), where = (parent_F != 0)) 
        return self.N * t2 * self.parent.pdf(x)
    
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
    def moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        if integral_method == "romberg":
            print("'romberg' integration method contains error right now. Please use with caution!")
            
        #Auto set bound if not specified to distribution bounds
        support = self.get_support()
        if integral_lb == None:
            integral_lb = support[0]
            if (integral_lb == -np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
                print("ERR: 'romberg' integration method selected, but distribution supports -infinite lower bound. Please supply the integral lower bound via 'integral_lb' parameter")
                return "nan"
        if integral_ub == None:
            integral_ub = support[1]
            if (support[1] == np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
                print("ERR: 'romberg' integration method selected, but distribution supports infinite upport bound. Please supply the integral upper bound via 'integral_ub' parameter")
                return "nan"
        
        #function x^k * f(x), for uncentered moment calculations
        def xk_fx(x):
            return (x**order) * self.pdf(x)
        
        #Integrate via numerical method
        if integral_method == "quad":
            return integrate.quad(func = xk_fx, a = integral_lb, b = integral_ub, **integral_kwargs)[0]
        elif integral_method == "romberg":
            return integrate.romberg(function = xk_fx, a = integral_lb, b = integral_ub, **integral_kwargs)
        else:
            print("ERR: Invalid integration method. Please set 'integral_method' to either 'quad' or 'romberg'")
            return "nan"
    
    #Method to calculate standardised moment
    #Unique to this, not availabel in scipy. Maybe better to code the expect() method for this
    def std_moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        #Auto set bound if not specified to distribution bounds
        support = self.get_support()
        if integral_lb == None:
            integral_lb = support[0]
            if (integral_lb == -np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
                print("ERR: 'romberg' integration method selected, but distribution supports -infinite lower bound. Please supply the integral lower bound via 'integral_lb' parameter")
                return "nan"
        if integral_ub == None:
            integral_ub = support[1]
            if (support[1] == np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
                print("ERR: 'romberg' integration method selected, but distribution supports infinite upport bound. Please supply the integral upper bound via 'integral_ub' parameter")
                return "nan"
        
        first_moment = self.moment(order=1, integral_method = integral_method, integral_lb = integral_lb, integral_ub = integral_ub, **integral_kwargs)
        if order==1:
            return first_moment
        else:
            #shifted and translated, for standardised moment calculations
            def std_xk_fx(x, translation, scale):
                return (((x-translation)/scale) ** order) * self.pdf(x)
            
            if integral_method == "quad":
                second_centered_moment = integrate.quad(func = std_xk_fx, a = integral_lb, b = integral_ub, args = (first_moment, 1), **integral_kwargs)[0]
            elif integral_method == "romberg":
                second_centered_moment = integrate.romberg(function = std_xk_fx, a = integral_lb, b = integral_ub, args = (first_moment, 1), **integral_kwargs)
            else:
                print("ERR: Invalid integration method. Please set 'integral_method' to either 'quad' or 'romberg'")
                return "nan"
                
            if order==2:
                return second_centered_moment
            else:
                if integral_method == "quad":
                    return integrate.quad(func = std_xk_fx, a = integral_lb, b = integral_ub, args = (first_moment, second_centered_moment), **integral_kwargs)[0]
                elif integral_method == "romberg":
                    return integrate.romberg(function = std_xk_fx, a = integral_lb, b = integral_ub, args = (first_moment, second_centered_moment), **integral_kwargs)
                else:
                    print("ERR: Invalid integration method. Please set 'integral_method' to either 'quad' or 'romberg'")
                    return "nan"
            
    #Method to get statistics, similar to stats() method in scipy distribution
    #Does not behave exactly the same as scipy. Need to code properly.
    def stats(self, moments, **kwargs):
        moments = list(moments)
        for a in moments:
            if a == 'm':
                return self.moment(order = 1, **kwargs)
            elif a == 'v':
                return self.std_moment(order = 2, **kwargs)
            elif a == 's':
                return self.std_moment(order = 3, **kwargs)
            elif a == 'k':
                return self.std_moment(order = 4, **kwargs)
            else:
                print("Unrecognised 'moments' argument. Should be 'm', 'v', 's', 'k', or combinations of those. See scipy doc for detail")
    
    def mean(self, **kwargs):
        return self.moment(order = 1, **kwargs)
    
    def var(self, **kwargs):
        return self.std_moment(order = 2, **kwargs)
    
    def std(self, **kwargs):
        return np.sqrt(self.var(**kwargs))
    
    def median(self):
        print("NOT CODED YET")
        return "nan"
    
    # Methods to calculate equivalent GEV
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
            
            xi = xi_from_zbc(dist_zbc, init_guess([dist_zbc]))[0];
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
    
    #Method to return reparameterised GEV distribution with Weivbull domain of attraction (genextreme_WR object)
    def get_genextreme_WR_dist(self, warning = True):
        #
        # TO DO:
        # Should check for convergence here
        #
        if warning:
            print("Warning: returned GEV distribution assumes 1) Convergence, 2) Weibull domain of attraction")
            print("It is your responsibility to check both of these assumptions")
        
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            return genextreme_WR(E = self.mean(), V = self.var(), z_b = self.support()[1])
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            print("Please also check the domain of attraction!")
            return "nan";
    
    #Method to return scipy genextreme distribution (scipy.stats.genextreme object)
    def get_genextreme_dist(self, warning = True):
        #
        # TO DO:
        # Should check for convergence here
        #
        
        if warning:
            print("Warning: returned GEV distribution assumes 1) Convergence, 2) Weibull domain of attraction")
            print("It is your responsibility to check both of these assumptions")
        
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            param = self.get_genextreme_parameter(parameterisation='scipy');
            return genextreme(loc = param[0], scale = param[1], c = param[2])
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            print("Please also check the domain of attraction!")
            return "nan";
    
    
        
#=======================================
#         genmaxima() class
#=======================================

        
class genextreme_WR():
    r"""Reparameterised GEV Distribution, Strictly for Weibull Domain of Attraction
    Reparameterised genextreme distribution (GEV distribution), based on mean (E), variance(V), and upper bound (z_b).
    Allows the calculation of all statistical properties (mean, var, CDF, etc.) when raised to the power of N.
    That is equivalent to taking the maxima from a GEV distribution with block size = N.
    Still allows the regular parameterisation too.
    
    **Only support Weibull Domain of Attraction**
    i.e. z_b < infinity and xi < 0
    
    Note on the definition of N:
    The larger the N, the larger the size of the block is. For instance if you have 10 million samples, and then
    you get the maxima every 1,000 samples, i.e. now you have 10,000 samples of the maxima, then N = 1,000
    Note that N need not to be an integer. The mathematics works for non integer N. e.g. if you have maxima data only, '
    each with block size 1,000, you could theoretically find equivalent GEV distributions of sample with block maxima of 
    10 by setting N = 10/1,000 = 0.01. Note that there are caveats to this, and the interpretation is up to you as the user.
    
    """
    def __init__(self, E = None, V = None, z_b = None, mu  = None, sigma = None, xi = None, parameterisation = 'coles'):
        #For the purpose of all calculation shall be done in coles parameterisation. Output can be specified to be in scipy if desired by the user.        
        self.parameterisation = parameterisation
        if (not (E==None)) and (not (V==None)) and (not (z_b==None)):
            self.E = E
            self.V = V
            self.z_b = z_b
            og_param = self.calculate_og_base_parameter()
            self.mu = og_param[0]
            self.sigma = og_param[1]
            self.xi = og_param[2]
        elif (not (mu==None)) and (not (sigma==None)) and (not (xi==None)):
            self.mu = mu
            self.sigma = sigma
            if parameterisation == 'coles':
                self.xi = xi
            elif parameterisation == 'scipy':
                self.xi = -xi
            else:
                print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
            self.E = self.mean()
            self.V = self.var()
            self.z_b = self.get_support()[1]
        else:
            print("Error: Either 'E', 'V' and 'z_b' must be supplied, or 'mu', 'sigma' and 'xi'")
        
        if not self.argcheck():
            print("Error parameter out of bounds")
            
    
    def argcheck(self):
        return ((self.sigma > 0) \
                and (self.xi < 0) \
                and (self.V > 0) \
                and (self.z_b < np.inf) \
                and (self.z_b > -np.inf) \
                and np.isreal(self.z_b) \
                and (self.z_b > self.E))
    
    def support(self):
        #Alias for get_support
        return self.get_support()
    
    def get_support(self):
        return genextreme(loc = self.mu, scale = self.sigma, c = -self.xi).support()
    
    def calculate_og_base_parameter(self, parameterisation = 'coles'):
        z_bc = (self.z_b - self.E) / np.sqrt(self.V)
        xi = xi_from_zbc(z_bc, init_guess([z_bc]))[0]
        g1 = special.gamma(1-xi)
        kx = (self.E - self.z_b)/g1
        sigma = kx * xi
        mu = self.E - kx * (g1-1)
        if parameterisation == 'coles':
            return (mu, sigma, xi)
        elif parameterisation == 'scipy':
            return (mu, sigma, -xi)
        else:
            print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
            return ("nan", "nan", "nan")
    
    #All the methods from this point allows raising of the distribution to the power of N, for some N>0
    #This is equivalent to taking the maxima of sample with block size = N
    #Based on C. Caprani (2004)
    
    def get_parameter(self, N=1, method='og'):
        """
        Method to get the parameter mean, variance and upper bound (E, V, and z_b)
        Raised to the power of N. Default N to 1.
        
        'method' argument can be either 'og' or 'direct'.
        'og' calculates the parameter via mu, sigma and xi. Ref: Caprani, 2004. Coles, 2001.
        'direct' calculates the parameters directly.
        Either method in theory should produce the same results, barring some floating point error.
        Use method check_get_parameter() to check if both methods agree, and the difference in them.
        """
        N = 1/N #Invert input N to mathematical N. Same with all the rest of the methods.
        if N==1:
            return (self.E, self.V, self.z_b)
        elif N>0:
            if method == 'og':
                raised_og_parameter = self.get_og_parameter(N=1/N) #Note on inversion of N due the difference in definition
                mu_N = raised_og_parameter[0]
                sigma_N = raised_og_parameter[1]

                E_N = mu_N + (sigma_N/self.xi) * (special.gamma(1-self.xi) - 1)
                V_N = np.abs((sigma_N**2/self.xi**2) * (special.gamma(1-2*self.xi) - special.gamma(1-self.xi)**2))
            elif method == 'direct':
                E_N = self.E / (N ** self.xi) - self.z_b * ( 1/(N ** self.xi) - 1 )
                V_N = self.V / (N ** (2*self.xi))
            else:
                print("Invalid argument 'method'. Should be either 'og' or 'direct'")
                return ("nan", "nan", "nan")
            
            return (E_N, V_N, self.z_b)
        else:
            print("Invalid argument 'N'. 'N' must be larger than 0")
            return ("nan", "nan", "nan")
        
    def check_get_parameter(self, N = 1):
        """
        Method to check the difference between 'og' and 'direct' in get_parameter() method
        Return two tuples. First tuple is the difference, second tuple is if they are within machine tolerance using numpy isclose() function.
        """
        og = self.get_parameter(N=N, method='og')
        direct = self.get_parameter(N=N, method='direct')
        diff = ( og[0] - direct[0], og[1] - direct[1], og[2] - direct[2] )
        close = ( np.isclose(og[0], direct[0]), np.isclose(og[1], direct[1]), np.isclose(og[2], direct[2]) )
        return (diff, close)
        
                         
    
    def get_og_parameter(self, N=1, parameterisation = 'coles'):
        #Note that the mathematical definition and the definition used in this function is inverted. So
        # the input N = 1 / mathematical N
        N = 1/N #Invert input N to mathematical N.
        if N==1:
            if parameterisation == 'coles':
                return (self.mu, self.sigma, self.xi)
            elif parameterisation == 'scipy':
                return (self.mu, self.sigma, -self.xi)
            else:
                print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
                return ("nan", "nan", "nan")
        elif N>0:
            mu_N = self.mu - (self.sigma/self.xi)*(1 - N**(-self.xi))
            sigma_N = self.sigma * N**(-self.xi)
            if parameterisation == 'coles':
                return (mu_N, sigma_N, self.xi)
            elif parameterisation == 'scipy':
                return (mu_N, sigma_N, -self.xi)
            else:
                print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
                return ("nan", "nan", "nan")
        else:
            print("Invalid argument 'N'. 'N' must be larger than 0")
            return ("nan", "nan", "nan")
    
    def rvs(self, size=1, random_state=None, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).rvs(size=size, random_state=random_state)
        
    def pdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).pdf(x)
    
    def logpdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).logpdf(x)
    
    def cdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).cdf(x)
    
    def logcdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).logcdf(x)
    
    def sf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).sf(x)
    
    def logsf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).logsf(x)
    
    def ppf(self, q, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).ppf(q)
    
    def isf(self, q, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).isf(q)
    
    def moment(self, order, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).moment(order=order)
    
    def stats(self, moments='mv', N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).stats(moments=moments)
    
    def entropy(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).entropy()
    
    def fit(self, data, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).fit(data)
    
    def expect(self, func, args=(), lb=None, ub=None, conditional=False, N=1, **kwds):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return enextreme(loc = param[0], scale = param[1], c = param[2]).expect(
            func = func, 
            args = args, 
            lb = lb, 
            ub = ub, 
            conditional = conditional, 
            kwds = kwds,
        )
    
    def median(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).median()
    
    def mean(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).mean()
    
    def var(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).var()
    
    def std(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).std()
    
    def interval(self, confidence, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).interval(confidence=confidence)