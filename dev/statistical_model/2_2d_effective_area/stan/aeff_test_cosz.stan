/*
 * Check Aeff implementation is working
 * using simple simulation of monoenergetic 
 * neutrinos with an isotropic distribution.
 *
 * @author Francesca Capel
 * @date May 2019
 */

functions {

#include vMF.stan
#include bspline_ev.stan
  
}

data {

  /* Simulated nu */
  int N;
  real log10E;

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

  unit_vector[3] direction[N];
  vector[N] zenith;
  vector[N] pdet;

  real cosz;
  
  int accept;
  simplex[2] prob;
  
  for (i in 1:N) {

    accept = 0;
    while(accept != 1) {

      /* Sample position */
      direction[i] = sphere_rng(1);
      zenith[i] = pi() - acos(direction[i][3]);
      cosz = cos(zenith[i]);
      
      if (cosz <= -0.8999) {
	cosz = -0.8999;
      }
      if (cosz >= 0.8999) {
	cosz = 0.8999;
      }
       
      /* Test against Aeff */
      pdet[i] = pow(10, bspline_func_2d(xknots, yknots, p, c, log10E, cosz) ) / aeff_max;
      prob[1] = pdet[i];
      prob[2] = 1 - pdet[i];
      accept = categorical_rng(prob);
      
    }
    
  }

}
