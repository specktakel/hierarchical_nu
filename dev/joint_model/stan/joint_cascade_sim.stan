/**
 * Forward model for neutrino energies and arrival directions.
 * Focusing on cascade events for now.
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

    int N = num_elements(D);
    vector[N] weights;
    real Mpc_to_km = 3.086e19;
    real normalisation = 0;

    for (k in 1:N) {
      normalisation += (Q / pow(D[k] * Mpc_to_km, 2));
    }
    for (k in 1:N) {
      weights[k] = (Q / pow(D[k] * Mpc_to_km, 2)) / normalisation;
    }
    
    return weights;
  }
  
  /**
   * Calculate weights from exposure integral.
   */
  vector get_exposure_weights(vector F, vector eps) {

    int N = num_elements(F);
    vector[N] weights;
    
    real normalisation = 0;

    for (i in 1:N) {
      normalisation += F[i] * eps[i];
    }
    
    for (i in 1:N) {
      weights[i] = (F[i] * eps[i]) / normalisation;
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

  real get_Nex_sim(vector F, vector eps) {

    int N = num_elements(F);
    real Nex = 0;

    for (i in 1:N) {
      Nex += F[i] * eps[i];
    }

    return Nex;
  }

}

data {

  /* sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;

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
  w_exposure = get_exposure_weights(F, eps);

  Nex = get_Nex_sim(F, eps);
  
  N = poisson_rng(Nex);

}

generated quantities {

  int lambda[N];
  unit_vector[3] omega;
  real zenith[N];
  real pdet[N];
  real accept;
  simplex[2] p;
  unit_vector[3] event[N];
  real Nex_sim = Nex;
  
  for (i in 1:N) {
    
    lambda[i] = categorical_rng(w_exposure);

    /* source */
    if (lambda[i] < Ns + 1) {
      
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
    /* background */
    else {

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

    event[i] = vMF_rng(omega, kappa);  	  
 
  }  

}
