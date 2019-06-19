/**
 * Model for neutrino energies and arrival directions.
 * Simple detector model for muon tracks and cascade events at high energies.
 * Use one representative Aeff and Eres for each event type (track/cascade) in 
 * order to show the structure without making a big mess of code.
 *
 * @author Francesca Capel
 * @date June 2019
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
  real Emin_tracks; // GeV
  real Emin_cascades; // GeV
  vector[N] Edet;
  vector[N] event_type; // 1 <=> track, 2 <=> cascade
  
  /* Sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;
  vector[Ns+1] z;
   
  /* Effective area */
  int Ngrid;
  vector[Ngrid] alpha_grid_tracks;
  vector[Ngrid] integral_grid_tracks[Ns+1];
  vector[Ngrid] alpha_grid_cascades;
  vector[Ngrid] integral_grid_cascades[Ns+1];
  real T;

  /* Energy resolution */
  int E_Ngrid;
  vector[E_Ngrid] log10_E_grid[N];
  vector[E_Ngrid] prob_grid[N];
  
  /* Detection */
  real<lower=0> kappa_tracks;
  real<lower=0> kappa_cascades;

  /* Debugging */
  real Q_scale;
  real F0_scale;

}

transformed data {

  vector[N] zenith;
  real Mpc_to_m = 3.086e22;
  
  for (i in 1:N) {

    zenith[i] = omega_to_zenith(omega_det[i]);

  }

  
}

parameters {

  real<lower=0, upper=1e60> Q;
  real<lower=0, upper=500> F0;

  real<lower=1.5, upper=3.5> alpha;

  vector<lower=Emin_cascades, upper=1.0e7>[N] Esrc;

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
  real Nex_tracks;
  real Nex_cascades;
  vector[N] E;
  
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

    if (event_type == 1) {
      lp[i] = log_F + log(pow(Emin_tracks/Emin_cascades, 1-alpha));
    }
    else if (event_type == 2) {
      lp[i] = log_F;
    }
    
    for (k in 1:Ns+1) {

      if (event_type[i] == 1) {
	lp[i, k] += pareto_lpdf(Esrc[i] | Emin_tracks, alpha - 1);
      }
      else if (event_type[i] == 2) {
	lp[i, k] += pareto_lpdf(Esrc[i] | Emin_tracks, alpha - 1);
      }
      E[i] = Esrc[i] / (1 + z[k]);
	
      /* Sources */
      if (k < Ns+1) {

	if (event_type[i] == 1) {
	  lp[i, k] += vMF_lpdf(omega_det[i] | varpi[k], kappa_tracks);
	}
	else if (event_type[i] == 2) {
	  lp[i, k] += vMF_lpdf(omega_det[i] | varpi[k], kappa_cascades);		  
	}
	
      }
      
      /* Background */
      else if (k == Ns+1) {
 
	lp[i, k] += log(1 / ( 4 * pi() ));

      }

      /* Lognormal approx. */
      lp[i, k] += lognormal_lpdf(Edet[i] | log(E[i] * 0.95), 0.13); // Nue_CC
	
      /* Actual P(Edet|E) from linear interpolation */
      //lp[i, k] += log(interpolate(log10_E_grid[i], prob_grid[i], log10(E[i])));
      
    } 
  }

  /* Nex */
  eps_tracks = get_exposure_factor(T, Emin_tracks, alpha, alpha_grid_tracks, integral_grid_tracks, Ns);
  eps_cascades = get_exposure_factor(T, Emin_cascades, alpha, alpha_grid_cascades, integral_grid_cascades, Ns);

  Nex_tracks = get_Nex(allF, eps_tracks) * pow(Emin_tracks/Emin_cascades, 1-alpha);
  Nex_cascades = get_Nex(allF, eps_cascades);
  Nex = Nex_tracks + Nex_cascades;
  
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


