/**
 * Testing Eres implementation.
 */


functions {

#include utils.stan
#include bspline_ev.stan
  
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

  real Edet_lpdf(real Edet, real E, vector xknots, vector yknots, int p, matrix c) {

    real prob;

    /* normalisation */
    int N = 100;
    vector[N] log10_E_grid = linspace(3.0, 7.0, N);
    vector[N] prob_grid;

    real norm;
    
    /* Find normalisation and maximum */
    for (i in 1:N) {
      prob_grid[i] = pow(10, bspline_func_2d(xknots, yknots, p, c, log10_E_grid[i], log10(Edet)));
    }
    norm = trapz(log10_E_grid, prob_grid);

    /* return normalised log probability */
    prob = pow(10, bspline_func_2d(xknots, yknots, p, c, log10(E), log10(Edet)) / norm);

    return log(prob);
  }

  
  
}

data {

  real E;
  int N;
  
  int E_p; // spline degree
  int E_Lknots_x; // length of knot vector
  int E_Lknots_y; // length of knot vector
  vector[E_Lknots_x] E_xknots; // knot sequence - needs to be a monotonic sequence
  vector[E_Lknots_y] E_yknots; // knot sequence - needs to be a monotonic sequence
  matrix[E_Lknots_x+E_p-1, E_Lknots_y+E_p-1] E_c; // spline coefficients  
  
}

generated quantities {

  vector[N] Edet;
  vector[N] lprob;
  
  for (i in 1:N) {

    Edet[i] = Edet_rng(E, E_xknots, E_yknots, E_p, E_c);
    lprob[i] = Edet_lpdf(Edet[i] | E, E_xknots, E_yknots, E_p, E_c);
    
  }
 

}
