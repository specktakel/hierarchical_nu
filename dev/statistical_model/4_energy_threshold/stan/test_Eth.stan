/**
 * Simple model to fit the toy simulation
 * in Eth_effects.ipynb.
 * 
 * Energy in GeV throughout.
 *
 * @author Francesca Capel
 * @date May 2019
 */

functions {

  /**
   * Lower bounded power law log pdf.
   */
  real power_law_lpdf(real E, real alpha, real Emin, real F_N) {

    if (E < Emin) {
      return negative_infinity();
    }
    else {
      return log(F_N * (alpha-1)/Emin * pow(E/Emin, -alpha) );
    }
  }

  /**
   * The expected number of events. 
   * expected_Nevents = T * \int dE Aeff(E) dN/dEdtdA(E)
   *
   * For this simple toy problem, we can do this analytically.
   */
  real get_expected_Nevents(real alpha, real Emin, real F_N, real T) {

    real A = F_N * T * (alpha-1) / Emin;
    real B = 10 * pow(1/1e3, 0.5) * pow(1e5, -alpha) / ((2*alpha) - 3);
    real C = (pow(Emin, 1.5) * pow(1e5, alpha)) - (pow(1e5, 1.5) * pow(Emin, alpha));

    return A * B * C;
    
  }

  /**
   * Effective area as a function of energy [m^2]
   */
  real get_effective_area(real E, real Emin) {

    if (E > 1e5) {
      return 0;
    }
    else {
      return 5 * pow(E/Emin, 0.5);
    }
    
  }

}

data {

  int Nevents;

  real Emin;
  real T;

  vector[Nevents] Edet;


}

parameters {

  real<lower=0, upper=50> F_N;
  real<lower=1.5, upper=3.5> alpha;

  vector<lower=Emin, upper=1e5>[Nevents] E;
  
}

model {

  vector[Nevents] lp;
  real Nex;

  for (i in 1:Nevents) {

    lp[i] = 0;

    lp[i] += power_law_lpdf(E[i] | alpha, Emin, F_N);

    lp[i] += normal_lpdf(Edet[i] | E[i], 0.2*E[i]);

    /* makes no difference! */
    //lp[i] += log(get_effective_area(E[i], Emin)); // m^2

    target += lp[i];
    
  }

  Nex = get_expected_Nevents(alpha, Emin, F_N, T);
  target += -Nex;
  
}
