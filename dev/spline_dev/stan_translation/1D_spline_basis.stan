/**
 * Simple Stan program to test bspline_ev.stan library.
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

  int Nknots = Lknots + p - 1; // total number of basis elements for given choice of degree and knots

  /* initialisation of bspline_basis */
  int k = p + 1; // order of spline

  vector[p] t0;
  vector[p] t1;
  vector[p+Lknots] tmp;
  vector[p+Lknots+p] t;
  real output;

  for (idx in 1:p) {
    
    t0[idx] = knots[1];
    t1[idx] = knots[Lknots];
    
  }

  tmp = append_row(t0, knots);
  t = append_row(tmp, t1);  
  
}

generated quantities {

  vector[Nevals] yvals[Nknots];

  //
  ///
  /* evaluate and store output */
  for (idx in 1:Nknots) {

    for (j in 1:Nevals) {

      yvals[idx, j] = eval_element(idx, xvals[j], k, t);

    } 

  } 
  
}
