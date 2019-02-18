/**
 * Simulate a simple neutrino dataset from sources.
 *
 * @author Francesca Capel
 * @date November 2018
 */

functions {

  /**
   * compute the absolute value of a vector 
   */
  real abs_val(vector input_vector) {

    real av;
    int n = num_elements(input_vector);

    real sum_squares = 0;
    for (i in 1:n) {
      sum_squares += (input_vector[i] * input_vector[i]);
    }
    av = sqrt(sum_squares);
    return av;
    
  }

  /**
   * Sample point on sphere orthogonal to mu.
   */
  vector sample_orthonormal_to_rng(vector mu) {

    int dim = num_elements(mu);
    vector[dim] v;
    vector[dim] proj_mu_v;
    vector[dim] orthto;
    
    for (i in 1:dim) {
     v[i] = normal_rng(0, 1);
    }
    
    proj_mu_v = mu * dot_product(mu, v) / abs_val(mu);
    orthto = v - proj_mu_v;
    
    return (orthto / abs_val(orthto));

  }
  
  /**
   * Rejection sampling scheme for sampling distance from center on
   * surface of the sphere.
   */
  real sample_weight_rng(real kappa, int dim) {

    real sdim = dim - 1; /* as S^{n-1} */
    real b = sdim / (sqrt(4. * pow(kappa, 2) + pow(sdim, 2)) + 2 * kappa);
    real x = (1 - b) / (1 + b);
    real c = kappa * x + sdim * log(1 - pow(x, 2));

    int i = 0;
    real z;
    real w;
    real u;
    while (i == 0) {
      z = beta_rng(sdim / 2, sdim / 2);
      w = (1 - (1 + b) * z) / (1 - (1 - b) * z);
      u = uniform_rng(0, 1);
      if (kappa * w + sdim * log(1 - x * w) - c >= log(u)) {
	i = 1;
      }
    }

    return w;
  }
  
  /**
   * Generate an N-dimensional sample from the von Mises - Fisher
   * distribution around center mu in R^N with concentration kappa.
   */
  vector vMF_rng(vector mu, real kappa) {

    int dim = num_elements(mu);
    vector[dim] result;

    real w = sample_weight_rng(kappa, dim);
    vector[dim] v = sample_orthonormal_to_rng(mu);

    result = ( v * sqrt(1 - pow(w, 2)) ) + (w * mu);
    return result;
   
  }

  /**
   * Sample a point uniformly from the surface of a sphere of 
   * a certain radius.
   */
  vector sphere_rng(real radius) {

    vector[3] result;
    real u = uniform_rng(0, 1);
    real v = uniform_rng(0, 1);
    real theta = 2 * pi() * u;
    real phi = acos( (2 * v) - 1 );

    result[1] = radius * cos(theta) * sin(phi); 
    result[2] = radius * sin(theta) * sin(phi); 
    result[3] = radius * cos(phi);

    return result;
    
  }
  
  /**
   * Calculate weights from source distances.
   */
  vector get_source_weights(real L, vector D) {

    int N = num_elements(D);
    vector[N] weights;
    real Mpc_to_km = 3.086e19;
    real normalisation = 0;

    for (k in 1:N) {
      normalisation += (L / pow(D[k] * Mpc_to_km, 2));
    }
    for (k in 1:N) {
      weights[k] = (L / pow(D[k] * Mpc_to_km, 2)) / normalisation;
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

}

data {

  /* sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;

  /* energies */
  //real<lower=1> alpha;
  //real Eth;
  //real Eerr;

  /* deflection */
  real<lower=0> kappa;
  
  /* flux */
  real<lower=0> L;
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
  real Mpc_to_km = 3.086e19;

  for (k in 1:Ns) {
    Fs += L / (4 * pi() * pow(D[k] * Mpc_to_km, 2));
  }

  FT = F0 + Fs;
  f = Fs / FT;

  w = get_source_weights(L, D);

  for (k in 1:Ns) {
    F[k] = w[k] * Fs;
  }
  F[Ns+1] = F0;
  
  /* N */
  w_exposure = get_exposure_weights(F, eps);

  Nex = get_Nex_sim(F, eps);
  print("f: ", f);
  print("Nex: ", Nex);
  print("Fs: ", Fs);
  print("F: ", F);
  print("eps: ", eps);
  print("w_exposure: ", w_exposure);
  
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


