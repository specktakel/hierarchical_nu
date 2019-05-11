/**
 * Model for neutrino energies and arrival directions.
 * Focusing on cascade events for now, and ignoring different flavours and interaction types.  
 * Adding in the 2D Aeff and spline implementation. 
 * Adding in energy resolution and threshold.
 *
 * @author Francesca Capel
 * @date May 2019
 */

functions {

#include vMF.stan
#include interpolation.stan
#include energy_spectrum.stan
#include bspline_ev.stan
#include utils.stan
  
  /**
   * Get exposure factor from spline information and source positions.
   * Units of [m^2 yr]
   */
  vector get_exposure_factor(real T, real Emin, real alpha, vector alpha_grid, vector[] integral_grid, int Ns) {

    int K = Ns+1;
    vector[K] eps;
    
    for (k in 1:K) {

      eps[k] = interpolate(alpha_grid, integral_grid[k], alpha) * ((alpha-1) / Emin) * T;
      
    }

    return eps;
  }

  
  /**
   * Calculate the expected number of detected events from each source.
   */
  real get_Nex(vector F, vector eps) {

    int K = num_elements(F);
    real Nex = 0;
 
    for (k in 1:K) {
      Nex += F[k] * eps[k];
    }

    return Nex;
  }

  
  /**
   * Convert from unit vector omega to theta of spherical coordinate system.
   * @param omega a 3D unit vector.
   */
  real omega_to_zenith(vector omega) {

    real zenith;
    
    int N = num_elements(omega);

    if (N != 3) {
      print("Error: input vector omega must be of 3 dimensions");
    }

    zenith = pi() - acos(omega[3]);
    
    return zenith;
  }

  real Edet_lpdf(real Edet, real E, vector xknots, vector yknots, int p, matrix c) {

    real prob = bspline_func_2d(xknots, yknots, p, c, log10(E), log10(Edet));

    return log(prob);
  }

  
}

data {
  
  /* Neutrinos */
  int N;
  unit_vector[3] omega_det[N]; 
  real Emin;
  real f_E;
  vector<lower=Emin>[N] Edet;
  
  /* Sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;
  vector[Ns+1] z;
   
  /* Effective area */
  int Ngrid;
  vector[Ngrid] alpha_grid;
  vector[Ngrid] integral_grid[Ns+1];
  real T;

  /* Detection */
  real kappa;

  /* Debugging */
  real Q_scale;
  real F0_scale;

}

transformed data {

  real Mpc_to_m = 3.086e22;
  
}

parameters {

  real<lower=0, upper=1e60> Q;
  real<lower=0, upper=500> F0;

  real<lower=1.5, upper=3.5> alpha;

  vector<lower=Emin, upper=1.0e7>[N] Esrc;

}

transformed parameters {

  /* Total source flux */
  real<lower=0> Fs;   
  
  /* Source flux */
  vector[Ns] F;
  vector[Ns+1] allF;
  vector[Ns+1] eps;

  /* Associated fraction */
  real<lower=0, upper=1> f; 
  real<lower=0> FT;

  /* Association probability */
  vector[Ns+1] lp[N];
  vector[Ns+1] log_F;
  real Nex;  
  vector[N] E;
  //real Esrc;
  
  /* Define transformed parameters */
  Fs = 0;
  for (k in 1:Ns) {
    F[k] = Q / (4 * pi() * pow(D[k] * Mpc_to_m, 2));
    allF[k] = F[k];
    Fs += F[k];
  }
  allF[Ns+1] = F0;
  
  FT = F0 + Fs;
  f = Fs / FT;

  /* Likelihood calculation  */
  log_F = log(allF);
  /* Rate factor */
  for (i in 1:N) {

    lp[i] = log_F;

    for (k in 1:Ns+1) {

      //Esrc = Edet[i] * (1+z[k]);
      //lp[i, k] += pareto_lpdf(Esrc | Emin, alpha - 1);	

      lp[i, k] += pareto_lpdf(Esrc[i] | Emin, alpha - 1);	
      E[i] = Esrc[i] / (1 + z[k]);
	
      /* Sources */
      if (k < Ns+1) {

	lp[i, k] += vMF_lpdf(omega_det[i] | varpi[k], kappa);
        
      }
      
      /* Background */
      else if (k == Ns+1) {
 
	lp[i, k] += log(1 / ( 4 * pi() ));
	
      }

      /* Lognormal approx. */
      //lp[i, k] += lognormal_lpdf(Edet[i] | log(E[i] * 0.95), 0.13); // Nue_CC
      //lp[i, k] += lognormal_lpdf(Edet[i] | log(E[i] * 0.3), 0.8); // Nue_NC
      lp[i, k] += lognormal_lpdf(Edet[i] | log(E[i]), f_E); // trying out large uncertainties
	
    } 
  }

  /* Nex */
  eps = get_exposure_factor(T, Emin, alpha, alpha_grid, integral_grid, Ns);
  Nex = get_Nex(allF, eps);
  
}

model {
  
  /* Rate factor */
  for (i in 1:N) {
    target += log_sum_exp(lp[i]);
  }
  
  /* Normalise */
  target += -Nex;
  
  Q ~ normal(0, Q_scale);
  F0 ~ normal(0, F0_scale);
  
  alpha ~ normal(2.0, 2);

}


