/**
 * Simple Stan program to test bspline_ev.stan library.
 * Performs 1D spine basis evaluation.
 *
 * @author Francesca Capel
 * @date 21st Febraury 2019
 */

functions {

#include bspline_ev.stan

}

data {

  int p; // spline degree
  int Lknots; // length of knot vector
  vector[Lknots] knots; // knot sequence - needs to be a monotonic sequence

  int Nevals; // length of vector to evaluate
  vector<lower=knots[1], upper=knots[Lknots]>[Nevals] xvals; // vector to evaluate
  
}

transformed data {

  int Nknots = Lknots + p - 1;
  
}

generated quantities {

  vector[Nevals] yvals[Nknots];

  /* evaluate and store output */
  for (idx_spline in 1:Nknots) {

    yvals[idx_spline] = bspline_basis_eval_vec(knots, p, idx_spline, xvals);
    
  } 
  
}
