/**
 * Model for neutrino energies and arrival directions.
 * Focusing on cascade events for now, and ignoring different flavours and interaction types.
 * 
 * @author Francesca Capel
 * @date February 2019
 */

functions {

#include vMF.stan
#include interpolation.stan
#include energy_spectrum.stan


  /**
   * Calculate the N_ex for a given kappa by
   * interpolating over a vector of eps values
   * for each source.
   */
  real get_Nex(vector F, vector eps, vector z, real alpha) {

    real eps_from_kappa;
    int K = num_elements(F) - 1;
    real Nex = 0;

    for (k in 1:K) {
      Nex += F[k] * eps[k] * pow(1 + z[k], 1 - alpha);
    }
    Nex += F[K+1] * eps[K+1];

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
  
}

data {
  
  /* Neutrinos */
  int N;
  unit_vector[3] omega_det[N]; 
  real Emin;
  real sigmaE;
  vector<lower=Emin>[N] Edet;
  
  /* Sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;
  vector[Ns] z;
   
  /* Observatory */
  vector[Ns + 1] eps;
  real A_IC;
  
}

transformed data {

  vector[N] zenith;

  for (i in 1:N) {

    zenith[i] = omega_to_zenith(omega_det[i]);

  }
  
}

parameters {

  real<lower=0, upper=1.0e53> Q;
  real<lower=0, upper=10> F0;

  real<lower=1, upper=10> alpha;
  real<lower=1, upper=1000> kappa;

  vector<lower=Emin, upper=1e3*Emin>[N] Esrc;

}

transformed parameters {

  /* Total source flux */
  real<lower=0> Fs;   
  
  /* Source flux */
  vector[Ns + 1] F;

  /* Associated fraction */
  real<lower=0, upper=1> f; 

  real<lower=0> FT;
  real Mpc_to_m = 3.086e22;
  
  Fs = 0;
  for (k in 1:Ns) {
    F[k] = Q / (4 * pi() * pow(D[k] * Mpc_to_m, 2));
    Fs += F[k];
  }
  
  F[Ns + 1] = F0;
  FT = F0 + Fs;
  f = Fs / FT;

}

model {

  vector[Ns + 1] log_F;
  real Nex;
  vector[N] pdet;
  
  vector[N] E;

  log_F = log(F);

  /* Nex */
  Nex = get_Nex(F, eps, z, alpha);
  
  /* Rate factor */
  for (i in 1:N) {

    vector[Ns + 1] lps = log_F;

    /* Sources */
    for (k in 1:Ns + 1) {

      lps[k] += pareto_lpdf(Esrc[i] | Emin, alpha - 1);

      if (k < Ns + 1) {

	lps[k] += vMF_lpdf(omega_det[i] | varpi[k], kappa);
	E[i] = Esrc[i] / (1 + z[k]);
	
      }
      
      /* Background */
      else {

	lps[k] += log(1 / ( 4 * pi() ));
	E[i] = Esrc[i];

      }


      /* Truncated gaussian */
      lps[k] += normal_lpdf(Edet[i] | E[i], sigmaE);
      if (Edet[i] < Emin) {
	lps[k] += negative_infinity();
      }
      else {
	lps[k] += -normal_lccdf(Emin | E[i], sigmaE);
      }

      /* Exposure factor */
      lps[k] += log(A_IC * zenith[i]);

    }
    
    target += log_sum_exp(lps);

  }
  
  /* Normalise */
  target += -Nex; 

  /* Priors */
  Q ~ normal(0.0, 1.0e51);
  F0 ~ normal(0.0, 10.0);

}
