/**
 * Stan translation of bspline_ev.py.
 * 1D and 2D spline evaluation using the Cos-de-Boor algorithm.
 *
 * @author Francesca Capel
 * @date February 2019
 */

real w(int i, int k, real x, vector t) {

  if (t[i]!=t[i+k-1]) {
    return (x - t[i]) / (t[i+k-1] - t[i]);
  }

  else {
    return 0.0;
  }
  
}

/**
 * Forward declaration to allow recursive programming in Stan.
 */
real bspline(int i, int k, real x, vector t);

/**
 * B_{i, k} (x) on extended knot sequence t (t1, ..., tq)
 * where k is the order of the spline
 * i is the index of the basis (1...q + p - 1)
 * where q is the length of the original knot sequence
 * p = k-1 (degree of spline)
 */
real bspline(int i, int k, real x, vector t) {

  real c1;
  real c2;
  
  if (k == 1) {
    
    /* piecewise constant */
    if ( (t[i] <= x) && (x < t[i+1]) ) {
      return 1.0;
    }

    else {
      return 0.0;
    }
  }

  else {

    c1 = w(i, k, x, t);
    c2 = 1.0 - w(i+1, k, x, t);

    return c1 * bspline(i, k-1, x, t) + c2 * bspline(i+1, k-1, x, t);
    
  }

}

int find_closest_knot(real x, vector t) {

  int Q = num_elements(t) - 1;
  int found = 0;

  int j = 1;
  while (found == 0) {

    if (t[j+1] > x) {
      found = 1;
    }
    else if (j == Q) {
      found = 1;
    }

    j += 1;
  }

  return j-1;

}


real eval_element(int i, real x, int k, vector t) {

  int closest_j = find_closest_knot(x, t);

  if (i <= closest_j - k) {

    return 0;
    
  }

  else {

    return bspline(i, k, x, t);
    
  }
  
}

/**
 * Constructs a 1D B-spline function from knots and coefficients,
 * 
 * @param t knot sequence without any padding (can be non-uniform)
 * @param p degree of B-spline (not order!)
 * @param c spline coefficients
 * @param x point to evaluate
 */
real eval_func_1d(vector t, int p, vector c, real x) {


}
