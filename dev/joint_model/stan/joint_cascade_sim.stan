/**
 * Forward model for neutrino energies and arrival directions.
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
   * Calculate weights from source distances.
   */
  vector get_source_weights(real Q, vector D) {

    int K = num_elements(D);
    vector[K] weights;
    real normalisation = 0;

    for (k in 1:K) {
      normalisation += (Q / pow(D[k], 2));
    }
    for (k in 1:K) {
      weights[k] = (Q / pow(D[k], 2)) / normalisation;
    }
    
    return weights;
  }
  
  /**
   * Calculate weights from exposure integral.
   */
  vector get_exposure_weights(vector F, vector eps, vector z, real alpha) {

    int K = num_elements(F);
    vector[K+1] weights;
    
    real normalisation = 0;
    
    /* Background */
    normalisation += F[1] * eps[1];
    /* Sources */
    for (k in 2:K) {
      normalisation += F[k] * eps[k] * pow(1 + z[k], 1 - alpha);
    }

    /* Background */
    weights[1] = (F[1] * eps[1]) / normalisation; 
    /* Sources */
    for (k in 2:K) {
      weights[k] = (F[k] * eps[k] * pow(1 + z[k], 1 - alpha)) / normalisation;
    }

    return weights;
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

  /**
   * Calculate the expected number of detected events.
   */
  real get_Nex_sim(vector F, vector eps, vector z, real alpha) {

    int K = num_elements(F);
    real Nex = 0;

    Nex += F[1] * eps[1]; 
    for (k in 2:K) {
      Nex += F[k] * eps[k] * pow(1 + z[k], 1 - alpha);
    }

    return Nex;
  }

}

data {

  /* sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;
  vector[Ns] z;

  /* energies */
  real<lower=1> alpha;
  real Emin;
  real sigmaE;

  /* deflection */
  real<lower=0> kappa;
  
  /* flux */
  real<lower=0> Q;
  real<lower=0> F0;

  /* exposure */
  vector[Ns+1] eps;
  int Ngrid;
  vector[Ngrid] m_grid;
  vector[Ngrid] zenith_grid;
  
}

transformed data {

  real<lower=0> Fs = 0;
  real<lower=0> FT;
  real<lower=0, upper=1> f;
  simplex[Ns] w;
  vector[Ns+1] F;
  vector[Ns+1] w_exposure;
  real Nex;
  int N;
  real Mpc_to_m = 3.086e22;

  for (k in 1:Ns) {
    Fs += Q / (4 * pi() * pow(D[k] * Mpc_to_m, 2));
  }

  FT = F0 + Fs;
  f = Fs / FT;

  w = get_source_weights(Q, D);

  for (k in 1:Ns) {
    F[k] = w[k] * Fs;
  }
  F[Ns+1] = F0;
  
  /* N */
  w_exposure = get_exposure_weights(F, eps, z, alpha);

  Nex = get_Nex_sim(F, eps, z, alpha);
  
  N = poisson_rng(Nex);

}

generated quantities {

  int lambda[N];
  unit_vector[3] omega;
  vector[N] Esrc;
  vector[N] E;
  vector[N] Edet;
  
  real zenith[N];
  real pdet[N];
  real accept;
  simplex[2] p;
  unit_vector[3] event[N];
  real Nex_sim = Nex;
  
  for (i in 1:N) {
    
    lambda[i] = categorical_rng(w_exposure);

    /* Source */
    if (lambda[i] < Ns + 1) {

      Esrc[i] = spectrum_rng( alpha, Emin * (1 + z[lambda[i]]) );
      E[i] = Esrc[i] / (1 + z[lambda[i]]);
      
      accept = 0;
      while (accept != 1) {
	omega = varpi[lambda[i]];
	zenith[i] = omega_to_zenith(omega);
	pdet[i] = interpolate(zenith_grid, m_grid, zenith[i]);
	p[1] = pdet[i];
	p[2] = 1 - pdet[i];
	accept = categorical_rng(p);
      }
    }
    
    /* Background */
    else {

      /* Background case simply fits the neutirno spectrum at Earth */
      /* Room for improvement here... */
      Esrc[i] = spectrum_rng(alpha, Emin);
      E[i] = Esrc[i];
      
      accept = 0;
      while (accept != 1) {
	omega = sphere_rng(1);
	zenith[i] = omega_to_zenith(omega);
	pdet[i] = interpolate(zenith_grid, m_grid, zenith[i]);
	p[1] = pdet[i];
	p[2] = 1 - pdet[i];
	accept = categorical_rng(p);
      }
    }

    /* Simple normal for now  */
    /* To be replaced with something more realistic... */
    Edet[i] = normal_rng(E[i], sigmaE);
    while (Edet[i] < Emin) {
      Edet[i] = normal_rng(E[i], sigmaE);
    }
    
    event[i] = vMF_rng(omega, kappa);  	  
 
  }  

}
