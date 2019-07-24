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
  real sigma;
  real Emax;
  vector[Ncomp] Eth;
  vector[Ncomp] mu;
  vector[Ncomp] area_weight;
  real dNdE;
  
  /* Neutrinos */
  int N;
  vector[N] location;
  vector[N] energy;

  /* Toggle extra factor */
  int extra_factor;
  int include_energy;
  
}

parameters {

  simplex[Ncomp] weight;

}

transformed parameters {

  vector[Ncomp] lp[N];
  vector[Ncomp] logw = log(weight);
  real Nbar;

  /* Rate */
  for (i in 1:N) {

    lp[i] = logw;
    
    for (k in 1:Ncomp) {

      lp[i, k] += normal_lpdf(location[i] | mu[k], sigma);

      if(include_energy == 1) {

	if (extra_factor == 1) {
	  lp[i, k] += uniform_lpdf(energy[i] | Eth[k], Emax);
	}
	else {
	  lp[i, k] += uniform_lpdf(energy[i] | 1, Emax);
	}
	
      }
      
    }

  }

  /* Normalisation */
  Nbar = 0;
  for (k in 1:Ncomp) {
    Nbar += area_weight[k] * dNdE * (Emax - Eth[k]);
  }

}

model {

  /* Rate */
  for (i in 1:N) {
    target += log_sum_exp(lp[i]);
  }

  target += -Nbar;
  
}
