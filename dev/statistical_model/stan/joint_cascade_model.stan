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
   * Calculate the expected number of detected events.
   */
  real get_Nex(vector F, vector eps, vector z, real alpha) {

    int K = num_elements(F);
    real Nex = 0;

    for (k in 1:K) {  
      Nex += F[k] * eps[k] * pow(1 + z[k], 1 - alpha);
      /* debug */
      //Nex += F[k] * eps[k];
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
   
  /* Observatory */
  vector[Ns+1] eps;
  real A_IC;
  real kappa;
  
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
  real<lower=0, upper=10> F0;

  real<lower=1, upper=10> alpha;

  vector<lower=Emin, upper=1e5*Emin>[N] Esrc;

}

transformed parameters {

  /* Total source flux */
  real<lower=0> Fs;   
  
  /* Source flux */
  ordered[Ns] F;
  vector[Ns+1] allF;
  
  /* Associated fraction */
  real<lower=0, upper=1> f; 
  real<lower=0> FT;

  /* Association probability */
  vector[Ns+1] lp[N];
  vector[Ns+1] log_F;
  real Nex;  
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

    lp[i] = log_F;

    for (k in 1:Ns+1) {
      
      lp[i, k] += pareto_lpdf(Esrc[i] | Emin, alpha - 1);	
      E[i] = Esrc[i] / (1 + z[k]);
      /* debug */
      //E[i] = Esrc[i];
	
      /* Sources */
      if (k < Ns+1) {

	lp[i, k] += vMF_lpdf(omega_det[i] | varpi[k], kappa);
	
      }
      
      /* Background */
      else if (k == Ns+1) {

	lp[i, k] += log(1 / ( 4 * pi() ));

      }


      /* Truncated gaussian */
      lp[i, k] += normal_lpdf(Edet[i] | E[i], f_E * E[i]);
      if (Edet[i] < Emin) {
      	lp[i, k] += negative_infinity();
      }
      else {
	lp[i, k] += -normal_lccdf(Emin | E[i], f_E * E[i]);
      }

      /* Exposure factor */
      //lp[i, k] += log(A_IC * zenith[i]);

    } 
  }

  /* Nex */
  Nex = get_Nex(allF, eps, z, alpha);
  
}

model {

  /* Rate factor */
  for (i in 1:N) {
    target += log_sum_exp(lp[i]);
  }
  
  /* Normalise */
  target += -Nex;
  
  /* Priors */
  Q ~ normal(0, 1e55);
  F0 ~ normal(0, 10);
  alpha ~ normal(2, 2);

}
