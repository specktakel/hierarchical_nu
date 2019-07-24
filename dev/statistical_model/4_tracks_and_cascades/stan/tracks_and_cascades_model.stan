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
   * Evaluate a polynomial with given coefficients.
   * Highest power of x first.
   */
  real eval_poly1d(real x, vector coeffs){
    int N = num_elements(coeffs);
    real res=0;
    for(i in 1:N){
      res += coeffs[i]*pow(x, N-i);
    }
    return res;
  }
  
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
   * Calculate the expected number of detected events from each source.
   */
  real get_Nex_test(vector F, vector eps) {

    int K = num_elements(F);
    real Nex = 0;
 
    for (k in 1:K-1) {
      Nex += F[k] * eps[k] * 0.75;
    }
    Nex += F[K] * eps[K];

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
  int N_tracks;
  int N_cascades;
  unit_vector[3] omega_det_tracks[N_tracks]; 
  unit_vector[3] omega_det_cascades[N_cascades]; 
  real Emin_tracks; // GeV
  real Emin_cascades; // GeV
  vector[N_tracks] Edet_tracks;
  vector[N_cascades] Edet_cascades;
  
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
  
  /* Energy Resolution */
  int<lower=0> track_poly_deg; // Degree of polynomial
  real track_poly_emin; 
  real track_poly_emax; 
  vector[track_poly_deg] track_poly_mu_coeffs; // Polynomial coefficiencies for mean
  vector[track_poly_deg] track_poly_sigma_coeffs; // Polynomial coefficiencies for sd

  
  /* Detection */
  real<lower=0> kappa_tracks;
  real<lower=0> kappa_cascades;

  /* Debugging */
  real Q_scale;
  real F0_scale;

}

transformed data {

  vector[N_tracks] zenith_tracks;
  vector[N_cascades] zenith_cascades;

  real Mpc_to_m = 3.086e22;
  
  for (i in 1:N_tracks) {
    zenith_tracks[i] = omega_to_zenith(omega_det_tracks[i]);
  }
  for (j in 1:N_cascades) {
    zenith_cascades[j] = omega_to_zenith(omega_det_cascades[j]);
  }

  
}

parameters {

  real<lower=0, upper=1e60> Q;
  real<lower=0, upper=500> F0;

  real<lower=1.5, upper=3.5> alpha;

  vector<lower=Emin_tracks, upper=1.0e8>[N_tracks] Esrc_tracks;
  vector<lower=Emin_cascades, upper=1.0e8>[N_cascades] Esrc_cascades;

}

transformed parameters {

  /* Total source flux */
  real<lower=0> Fs;   
  
  /* Source flux */
  vector[Ns] F;
  vector[Ns+1] allF;
  vector[Ns+1] eps_tracks;
  vector[Ns+1] eps_cascades;

  /* Associated fraction */
  real<lower=0, upper=1> f; 
  real<lower=0> FT;

  /* Association probability */
  vector[Ns+1] lp_tracks[N_tracks];
  vector[Ns+1] lp_cascades[N_cascades];
  vector[Ns+1] log_F;
  real Nex;
  real Nex_tracks;
  real Nex_cascades;
  vector[N_tracks] E_tracks;
  vector[N_cascades] E_cascades;
  
  /* Energy resolution parameters */
  real e_reso_mu;
  real e_reso_sigma;
  real e_reso_eval_energy;
  
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

  /* Tracks */
  for (i in 1:N_tracks) {

    lp_tracks[i] = log_F + log(pow(Emin_tracks/Emin_cascades, 1-alpha));

    for (k in 1:Ns+1) {

      lp_tracks[i, k] += pareto_lpdf(Esrc_tracks[i] | Emin_tracks, alpha - 1);
      E_tracks[i] = Esrc_tracks[i] / (1 + z[k]);
   	
      /* Sources */
      if (k < Ns+1) {
	  lp_tracks[i, k] += vMF_lpdf(omega_det_tracks[i] | varpi[k], kappa_tracks);	
      }
      /* Background */
      else if (k == Ns+1) {
        lp_tracks[i, k] += log(1 / ( 4 * pi() ));
      }
  
      if (E_tracks[i] < track_poly_emin){
          e_reso_eval_energy = track_poly_emin;
      }
      else if (E_tracks[i] > track_poly_emax ){
          e_reso_eval_energy = track_poly_emax;
      
      }else{
          e_reso_eval_energy = E_tracks[i];
      }
      
      e_reso_mu = eval_poly1d(log10(e_reso_eval_energy), track_poly_mu_coeffs);
      e_reso_sigma = eval_poly1d(log10(e_reso_eval_energy), track_poly_sigma_coeffs);
      
      lp_tracks[i, k] += lognormal_lpdf(log10(Edet_tracks[i]) | log(e_reso_mu), e_reso_sigma);
      
      /* Lognormal approx. */
      //lp[i, k] += lognormal_lpdf(Edet[i] | log(E[i] * 0.95), 0.13); // Nue_CC
      
      //lp_tracks[i, k] += lognormal_lpdg(Edet_tracks[i] | )
      
      //lp_tracks[i, k] += lognormal_lpdf(Edet_tracks[i] | log(E_tracks[i]), 0.3);
      
      /* Actual P(Edet|E) from linear interpolation */
      //lp[i, k] += log(interpolate(log10_E_grid[i], prob_grid[i], log10(E[i])));
      
    } 
  }

  /* Cascades */
  for (j in 1: N_cascades) {

    lp_cascades[j] = log_F;

    for (k in 1:Ns+1) {

      lp_cascades[j, k] += pareto_lpdf(Esrc_cascades[j] | Emin_cascades, alpha - 1);
      E_cascades[j] = Esrc_cascades[j] / (1 + z[k]);

      /* Sources */
      if (k < Ns+1 ) { 
	lp_cascades[j, k] += vMF_lpdf(omega_det_cascades[j] | varpi[k], kappa_cascades);		  
      }
      /* Background */
      else if (k == Ns+1) {
	lp_cascades[j, k] += log(1 / ( 4 * pi() ));
      }

      lp_cascades[j, k] += lognormal_lpdf(Edet_cascades[j] | log(E_cascades[j]), 0.15);
      
    }
 
  }
  
  /* Nex */
  eps_tracks = get_exposure_factor(T, Emin_tracks, alpha, alpha_grid_tracks, integral_grid_tracks, Ns);
  eps_cascades = get_exposure_factor(T, Emin_cascades, alpha, alpha_grid_cascades, integral_grid_cascades, Ns);

  Nex_tracks = get_Nex(allF * pow(Emin_tracks/Emin_cascades, 1-alpha), eps_tracks);
  Nex_cascades = get_Nex(allF, eps_cascades);
  Nex = Nex_tracks + Nex_cascades;
  
}

model {
  
  
  for (i in 1:N_tracks) {
    target += log_sum_exp(lp_tracks[i]);
  }
  target += -Nex_tracks;
  
  
  for (j in 1:N_cascades) {
    target += log_sum_exp(lp_cascades[j]);
  }
  target += -Nex_cascades;
  
  
  Q ~ normal(0, Q_scale);
  F0 ~ normal(0, F0_scale);
  
  alpha ~ normal(2.0, 2);

}


