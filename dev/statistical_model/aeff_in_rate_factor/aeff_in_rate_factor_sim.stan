/**
 * Simple demonstration that aeff should be in 
 * the rate factor.
 *
 * @author Francesca Capel
 * @date July 2019
 */


data {

  /* Sources */
  int Ncomp;
  vector[Ncomp] mu;
  simplex[Ncomp] area_weight;
  vector[Ncomp] Eth;
  real Emax;
  real sigma;
  real dNdE;
  
}

transformed data {

  int N;
  real Nbar = 0;
  simplex[Ncomp] weight;
  real norm;
  
  for (k in 1:Ncomp) {

    Nbar += area_weight[k] * dNdE * (Emax - Eth[k]);

  }

  for (k in 1:Ncomp) {

    weight[k] = (area_weight[k] * dNdE * (Emax - Eth[k])) / Nbar;
    
  }
  

  N = poisson_rng(Nbar);
  
}

generated quantities {

  vector[N] label;
  vector[N] energy;
  vector[N] location;
  vector[Ncomp] weight_out = weight;

  for (i in 1:N) {

    /* Sample source */
    label[i] = categorical_rng(weight);

    /* Sample energy */
    energy[i] = uniform_rng(1, Emax);

    /* Selection */
    if (label[i] == 1) {

      while (energy[i] < Eth[1]) {
	energy[i] = uniform_rng(1, Emax);
      }
      location[i] = normal_rng(mu[1], sigma);
      
    }
    if (label[i] == 2) {

      while (energy[i] < Eth[2]) {
	energy[i] = uniform_rng(1, Emax);
      }
      location[i] = normal_rng(mu[2], sigma);
    }

  }

  
}
