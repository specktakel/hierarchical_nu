import math 
import numpy as np
import pylab as pl
from scipy import optimize
from scipy import stats
from scipy.stats import norm
from scipy.integrate import simps
from scipy.interpolate import interp1d
from timeit import default_timer as timer



# Generate a pseudo-data set given a set of model parameters
def GeneratePseudoData(p_s, lambda_tot, model_fixed_par):
    lambda_b = (1. - p_s) * lambda_tot
    lambda_s = p_s * lambda_tot
    
    # retrieve parameters
    gaussian_mean = model_fixed_par["gaussian_mean"]
    gaussian_sigma = model_fixed_par["gaussian_sigma"]
    x_min = model_fixed_par["x_min"]
    x_max = model_fixed_par["x_max"]

    size_b = stats.poisson.rvs(mu = lambda_b)
    samples_b = np.random.uniform(x_min, x_max, int(size_b))

    size_s = stats.poisson.rvs(mu = lambda_s)
    samples_s = np.random.normal(gaussian_mean, gaussian_sigma, int(size_s))

    dataset = {
        "N_s" : size_s,
        "N_b" : size_b,
        "samples" : np.concatenate((samples_b,samples_s), axis=0),
        "samples_b" : samples_b,
        "samples_s" : samples_s
    }

    return dataset

# Negative log likelihood function 
def NLL (samples, p_s, lambda_tot, model_fixed_par):
    lambda_s = p_s * lambda_tot
    lambda_b = (1. - p_s) * lambda_tot

    # retrieve from the model the parameters needed for the calculation
    gaussian_mean = model_fixed_par["gaussian_mean"]
    gaussian_sigma = model_fixed_par["gaussian_sigma"]
    width = model_fixed_par["x_max"] - model_fixed_par["x_min"]

    # extended term
    nll = - stats.poisson.logpmf(k = samples.size, mu = lambda_s + lambda_b)

    # can we compute the log of the pdf directly?
    def pdf (x): 
       return (1./(lambda_s + lambda_b)) * (lambda_b/width + lambda_s * norm.pdf(x, gaussian_mean, gaussian_sigma))

    # event loop
    nll -= np.sum( np.log(pdf(samples)))
    
    return nll


# Generalized likelihood-ratio test statistic, i.e. ratio between the NLL 
# minimum in the restrected parameter space of the null hypothesis vs the 
# NLL absolute mimimum
def TestStatistic (dataset, p_s, model_fixed_par):

    samples = dataset["samples"] 
    N_s = dataset["N_s"]
    N_b = dataset["N_b"]

    def fmin_full_space(par):  
        return NLL(samples = samples, p_s = par[1], lambda_tot = par[0], \
                   model_fixed_par = model_fixed_par)

    NLL_abs_min = optimize.minimize(fmin_full_space, (N_b+N_s, 0.2), \
                                    bounds=((1e-20, None), (0, 1)), \
                                    options={'disp': False})
        
    # now find minimum in the restrectited parameter space
    def fmin_restricted_space(par): 
        return NLL(samples = samples, p_s = p_s, lambda_tot = par[0], \
                   model_fixed_par = model_fixed_par)

    NLL_res_min = optimize.minimize(fmin_restricted_space, (N_b), \
                                    bounds=((1e-20, None),), \
                                    options={'disp': False})

    return 2*(NLL_res_min.fun - NLL_abs_min.fun)


# Calculate test statistic probability distributions for a given value of lambda_s and lambda_b
def CalculateTestDist(n_datasets, lambda_s, lambda_b, model_fixed_par):
             
    test_statistic_values =[]

    for i in np.arange(n_datasets):

        if (i%1000)==0: print ("lambda_s = {}, lambda_b = {}: {} / {} trials done.".format(lambda_s, lambda_b,i, n_datasets))
        pseudo_data = GeneratePseudoData(lambda_s = lambda_s, lambda_b = lambda_b, model_fixed_par = model_fixed_par)
        test_statistic = TestStatistic(dataset = pseudo_data, lambda_s = lambda_s, model_fixed_par = model_fixed_par)   
        test_statistic_values.append(test_statistic)
          
    test_statistic_values = np.array(test_statistic_values)
    
    return test_statistic_values


