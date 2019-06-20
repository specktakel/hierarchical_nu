/**
 * Forward model for neutrino energies and arrival directions.
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
   * Calculate weights from exposure integral.
   */
  vector get_exposure_weights(vector F, vector eps) {

    int K = num_elements(F);
    vector[K] weights;
    
    real normalisation = 0;
    
    for (k in 1:K) {
      normalisation += F[k] * eps[k];
    }

    for (k in 1:K) {
      weights[k] = F[k] * eps[k] / normalisation;
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
   * Calculate the expected number of detected events from each source.
   */
  real get_Nex_sim(vector F, vector eps) {

    int K = num_elements(F);
    real Nex = 0;

    for (k in 1:K) {
      Nex += F[k] * eps[k];
    }

    return Nex;
  }

real Edet_rng(real E, vector xknots, vector yknots, int p, matrix c) {

    int N = 100;
    vector[N] log10_Edet_grid = linspace(1.0, 7.0, N);
    vector[N] prob_grid;

    real norm;
    real prob_max;
    real prob;
    real log10_Edet;
    real Edet;
    int accept;
    
    /* Find normalisation and maximum */
    for (i in 1:N) {
      prob_grid[i] = pow(10, bspline_func_2d(xknots, yknots, p, c, log10(E), log10_Edet_grid[i]));
    }
    norm = trapz(log10_Edet_grid, prob_grid);
    prob_max = max(prob_grid) / norm;
    
    /* Sampling */
    accept = 0;
    while (accept != 1) {
      log10_Edet = uniform_rng(1.0, 7.0);
      prob = uniform_rng(0, prob_max);

      if (prob <= pow(10, bspline_func_2d(xknots, yknots, p, c, log10(E), log10_Edet)) / norm) {
	accept = 1;
      }
    }

    return pow(10, log10_Edet);
  }
    
}

data {

  /* sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;
  vector[Ns+1] z;

  /* energies */
  real<lower=1> alpha;
  real Emin_tracks; // GeV
  real Emin_cascades; // GeV

  /* deflection */
  real<lower=0> kappa_tracks;
  real<lower=0> kappa_cascades;
  
  /* flux */
  real<lower=0> Q;
  real<lower=0> F0;

  /* Effective area */
  int Ngrid;
  real T;
  int p; // spline degree
 
  vector[Ngrid] alpha_grid_cascades;
  vector[Ngrid] integral_grid_cascades[Ns+1];
  int Lknots_x_cascades; // length of knot vector
  int Lknots_y_cascades; // length of knot vector
  vector[Lknots_x_cascades] xknots_cascades; // knot sequence - needs to be a monotonic sequence
  vector[Lknots_y_cascades] yknots_cascades; // knot sequence - needs to be a monotonic sequence
  matrix[Lknots_x_cascades+p-1, Lknots_y_cascades+p-1] c_cascades; // spline coefficients
  real aeff_max_cascades;

  vector[Ngrid] alpha_grid_tracks;
  vector[Ngrid] integral_grid_tracks[Ns+1];
  int Lknots_x_tracks; // length of knot vector
  int Lknots_y_tracks; // length of knot vector
  vector[Lknots_x_tracks] xknots_tracks; // knot sequence - needs to be a monotonic sequence
  vector[Lknots_y_tracks] yknots_tracks; // knot sequence - needs to be a monotonic sequence
  matrix[Lknots_x_tracks+p-1, Lknots_y_tracks+p-1] c_tracks; // spline coefficients
  real aeff_max_tracks;

  /* Energy resolution */
  int E_p; // spline degree
  int E_Lknots_x; // length of knot vector
  int E_Lknots_y; // length of knot vector
  vector[E_Lknots_x] E_xknots; // knot sequence - needs to be a monotonic sequence
  vector[E_Lknots_y] E_yknots; // knot sequence - needs to be a monotonic sequence
  matrix[E_Lknots_x+E_p-1, E_Lknots_y+E_p-1] E_c; // spline coefficients 
  
}

transformed data {

  real<lower=0> Fs = 0;
  real<lower=0> FT;
  real<lower=0, upper=1> f;
  vector[Ns+1] F;
  simplex[Ns+1] w_exposure_tracks;
  simplex[Ns+1] w_exposure_cascades;
  simplex[2] w_event_type;
  real Nex;
  real Nex_tracks;
  real Nex_cascades;
  int N;
  real Mpc_to_m = 3.086e22;
  vector[Ns+1] eps_tracks;
  vector[Ns+1] eps_cascades;
  
  for (k in 1:Ns) {
    F[k] = Q / (4 * pi() * pow(D[k] * Mpc_to_m, 2));
    Fs += F[k];
  }
  F[Ns+1] = F0;

  FT = F0 + Fs;
  f = Fs / FT;

  /* N */
  eps_tracks = get_exposure_factor(T, Emin_tracks, alpha, alpha_grid_tracks, integral_grid_tracks, Ns);
  eps_cascades = get_exposure_factor(T, Emin_cascades, alpha, alpha_grid_cascades, integral_grid_cascades, Ns);
  Nex_tracks = get_Nex_sim(F, eps_tracks) * pow(Emin_tracks/Emin_cascades, 1-alpha);
  Nex_cascades = get_Nex_sim(F, eps_cascades);
  w_exposure_tracks = get_exposure_weights(F, eps_tracks);
  w_exposure_cascades = get_exposure_weights(F, eps_cascades);
  Nex = Nex_tracks + Nex_cascades;

  w_event_type[1] = Nex_tracks / Nex; // Tracks
  w_event_type[2] = 1 - (Nex_tracks / Nex); // Cascades
  
  N = poisson_rng(Nex);

  /* Debug */
  print("w_exposure_tracks: ", w_exposure_tracks);
  print("w_exposure_cascades: ", w_exposure_cascades);
  print("w_event_type: ", w_event_type);
  print("N: ", N);
}

generated quantities {

  int lambda[N];
  int event_type[N];
  unit_vector[3] omega;
  vector[N] Esrc;
  vector[N] E;
  vector[N] Edet;
  
  real zenith[N];
  real pdet[N];
  real accept;
  simplex[2] prob;
  unit_vector[3] event[N];
  real Nex_sim = Nex;

  real log10E;
  real cosz;
  int n_trials;
  
  for (i in 1:N) {

    /* sample interaction type */
    event_type[i] = categorical_rng(w_event_type);

    /* sample position */
    if (event_type[i] == 1) {
      lambda[i] = categorical_rng(w_exposure_tracks);
    }
    else if (event_type[i] == 2) {
      lambda[i] = categorical_rng(w_exposure_cascades);   
    }
    if (lambda[i] < Ns+1) {
      omega = varpi[lambda[i]];
    }

    accept = 0;
    n_trials = 0;
    while (accept != 1) {

      if (lambda[i] == Ns+1) {
	omega = sphere_rng(1);
      }
      zenith[i] = omega_to_zenith(omega);
      
      /* tracks */
      if (event_type[i] == 1) {

	/* Sample energy */
	Esrc[i] = spectrum_rng( alpha, Emin_tracks * (1 + z[lambda[i]]) );
	E[i] = Esrc[i] / (1 + z[lambda[i]]);

	/* check bounds of spline */
	log10E = log10(E[i]);
	cosz = cos(zenith[i]);
	if (cosz <= -0.9499) {
	  cosz = -0.9499;
	}
	if (cosz >= 0.0499) {
	  cosz = 0.0499;
	}
      	
	/* Test against Aeff */
	if (cosz > 0.1) {
	  pdet[i] = 0.0;
	}
	else if (log10E > 7.0) {
	  pdet[i] = 0.0;
	}
	else {
	  pdet[i] = pow(10, bspline_func_2d(xknots_tracks, yknots_tracks, p, c_tracks, log10E, cosz)) / aeff_max_tracks;
	  if (pdet[i] < 0) {
	    pdet[i] = 0;
	  }
	  if (pdet[i] > 1) {
	    pdet[i] = 1;
	  }
	}
	prob[1] = pdet[i];

      }
      /* cascades */
      else if (event_type[i] == 2) {
    
	/* Sample energy */
	Esrc[i] = spectrum_rng( alpha, Emin_cascades * (1 + z[lambda[i]]) );
	E[i] = Esrc[i] / (1 + z[lambda[i]]);

	/* check bounds of spline */
	log10E = log10(E[i]);
	cosz = cos(zenith[i]);
	if (log10E >= 6.96) {
	  log10E = 6.96;
	}
	if (cosz <= -0.8999) {
	  cosz = -0.8999;
	}
	if (cosz >= 0.8999) {
	  cosz = 0.8999;
	}
      
	/* Test against Aeff */
	if (log10E > 7.0) {
	  pdet[i] = 0.0;
	}
	else {
	  pdet[i] = pow(10, bspline_func_2d(xknots_cascades, yknots_cascades, p, c_cascades, log10E, cosz)) / aeff_max_cascades;
	}
	prob[1] = pdet[i];
      }
      
      prob[2] = 1 - pdet[i];
      //print("prob: ", prob);
      accept = categorical_rng(prob);
      n_trials += 1;
      if (n_trials > 1e5) {
	accept = 1;
	print("problem component: ", lambda[i])
      }
    }

    /* Detection effects */
    if (event_type[i] == 1) { 
      event[i] = vMF_rng(omega, kappa_tracks);  	  
    }
    else if (event_type[i] == 2) {
      event[i] = vMF_rng(omega, kappa_cascades);
    }
    
    /* Energy losses */
    /* Nue_CC */
    /*
    Edet[i] = Edet_rng(E[i], E_xknots, E_yknots, E_p, E_c);  
    
    if (event_type[i] == 1) {
      while (Edet[i] < Emin_tracks) {
	Edet[i] = Edet_rng(E[i], E_xknots, E_yknots, E_p, E_c);
      }
    }
    else if (event_type[i] == 2) {
      while (Edet[i] < Emin_cascades) {
	Edet[i] = Edet_rng(E[i], E_xknots, E_yknots, E_p, E_c);
      }
    }
    */

    /* Simple test case */
    Edet[i] = lognormal_rng(log(E[i]), 0.15);
    if (event_type[i] == 1) {
      while (Edet[i] < Emin_tracks) {
	Edet[i] = lognormal_rng(log(E[i]), 0.15);
      }
    }
    else if (event_type[i] == 2) {
      while (Edet[i] < Emin_cascades) {
	Edet[i] = lognormal_rng(log(E[i]), 0.15);
      }
    }
       
 
  }  

}


