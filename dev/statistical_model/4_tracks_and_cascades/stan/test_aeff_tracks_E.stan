/**
 * Testing the aeff implementation for tracks.
 */

functions {

#include vMF.stan
#include bspline_ev.stan
  
}

data {

  /* Simulated nu */
  int N;
  real cosz;

  /* Aeff spline */
  int p;
  int Lknots_x; 
  int Lknots_y; 
  vector[Lknots_x] xknots; 
  vector[Lknots_y] yknots; 
  matrix[Lknots_x+p-1, Lknots_y+p-1] c;  
  real aeff_max;
  
}

generated quantities {

  vector[N] log10E;
  vector[N] pdet;

  real l10E;
  real cz;
  
  int accept;
  simplex[2] prob;
  
  for (i in 1:N) {

    accept = 0;
    while(accept != 1) {

      /* Sample energy */
      log10E[i] = uniform_rng(3.0, 7.0);

      l10E = log10E[i];
      cz = cosz;
      
      /* check bounds of spline */
      if (cz <= -0.9499) {
	cz = -0.9499;
      }
      if (cz >= 0.0499) {
	cz = 0.0499;
      }
      
      /* Test against Aeff */
      pdet[i] = pow(10, bspline_func_2d(xknots, yknots, p, c, l10E, cz) ) / aeff_max;
      prob[1] = pdet[i];
      prob[2] = 1 - pdet[i];
      accept = categorical_rng(prob);
      
    }
    
  }

}
