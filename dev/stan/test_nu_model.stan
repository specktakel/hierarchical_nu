/**
 * Testing some ideas for neutrino model.
 *
 * @author Francesca Capel
 * @date Novmeber 2018
 */

functions {


  /**
   * Interpolate x from a given set of x and y values.
   */
  real interpolate(vector x_values, vector y_values, real x) {
    real x_left;
    real y_left;
    real x_right;
    real y_right;
    real dydx;

    int Nx = num_elements(x_values);
    real xmin = x_values[1];
    real xmax = x_values[Nx];
    int i = 1;

    if (x > xmax || x < xmin) {
      print("Warning, x is outside of interpolation range!");
      print("Returning edge values.");
      print("x:", x);
      print("xmax", xmax);
      
      if(x > xmax) {
	return y_values[Nx];
      }
      else if (x < xmin) {
	return y_values[1];
      }
    }
    
    if( x >= x_values[Nx - 1] ) {
      i = Nx - 1;
    }
    else {
      while( x > x_values[i + 1] ) { i = i+1; }
    }

    x_left = x_values[i];
    y_left = y_values[i];
    x_right = x_values[i + 1];
    y_right = y_values[i + 1];

    dydx = (y_right - y_left) / (x_right - x_left);
    
    return y_left + dydx * (x - x_left);
  }

  /**
   * Calculate the N_ex for a given kappa by
   * interpolating over a vector of eps values
   * for each source.
   */
  real get_Nex(vector F, vector eps) {

    real eps_from_kappa;
    int N = num_elements(F);
    real Nex = 0;

    for (i in 1:N) {
      Nex += F[i] * eps[i];
    }

    return Nex;
  }
  
  /**
   * Define the vMF PDF.
   * NB: Cannot be vectorised.
   * Uses sinh(kappa) ~ exp(kappa)/2 
   * approximation for kappa > 100.
   */
  real vMF_lpdf(vector v, vector mu, real kappa) {

    real lprob;
    if (kappa > 100) {
      lprob = kappa * dot_product(v, mu) + log(kappa) - log(4 * pi()) - kappa + log(2);
    }
    else {
      lprob = kappa * dot_product(v, mu) + log(kappa) - log(4 * pi() * sinh(kappa));
    }
    return lprob;
    
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
  
  /* nu */
  int N;
  unit_vector[3] omega_det[N]; 

  /* sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;
    
  /* observatory */
  vector[Ns + 1] eps;
  real A_IC;
  int Ngrid;
  vector[Ngrid] m_grid;
  vector[Ngrid] zenith_grid;

  
}

transformed data {

  vector[N] zenith;

  for (i in 1:N) {

    zenith[i] = omega_to_zenith(omega_det[i]);

  }
  
}

parameters {

  real<lower=0, upper=1.0e53> L;
  real<lower=0, upper=10> F0;
  
  real<lower=1, upper=1000> kappa;
  
}

transformed parameters {

  /* total source flux */
  real<lower=0> Fs;   
  
  /* source flux */
  vector[Ns + 1] F;

  /* associated fraction */
  real<lower=0, upper=1> f; 

  real<lower=0> FT;
  real Mpc_to_m = 3.086e22;
  
  Fs = 0;
  for (k in 1:Ns) {
    F[k] = L / (4 * pi() * pow(D[k] * Mpc_to_m, 2));
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
  
  log_F = log(F);

  /* Nex */
  Nex = get_Nex(F, eps);
  
  /* rate factor */
  for (i in 1:N) {

    vector[Ns + 1] lps = log_F;

    for (k in 1:Ns + 1) {
      
      if (k < Ns + 1) {
	lps[k] += vMF_lpdf(omega_det[i] | varpi[k], kappa);
      }
      
      else {
	lps[k] += log(1 / ( 4 * pi() ));
      }

      lps[k] += log(A_IC * zenith[i]);
    }
    
    target += log_sum_exp(lps);

  }
  
  /* normalise */
  target += -Nex; 

  /* priors */
  L ~ normal(0.0, 1.0e51);
  F0 ~ normal(0.0, 10.0);

}
