/**
 * Simple Stan program to test bspline_ev.stan library.
 * Represent an arbitrary function as a superposition of B-splines.
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
  vector[Lknots+p-1] c; // spline coefficients 
  
  int Nevals; // length of vector to evaluate
  vector<lower=knots[1], upper=knots[Lknots]>[Nevals] xvals; // vector to evaluate
  
}

generated quantities {

  vector[Nevals] yvals;
  
  for (idx in 1:Nevals) {

    yvals[idx] = bspline_func_1d(knots, p, c, xvals[idx]);

  } 
  
}
