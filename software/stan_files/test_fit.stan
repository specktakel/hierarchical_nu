/**
 * Simple model to test
 * basic principles of 
 * actual fit
 **/

functions {

  real flux_conv(real alpha, real e_low, real e_up) {

    real f1;
    real f2;
    
    if(alpha == 1.0) {
      f1 = (log(e_up)-log(e_low));
    }
    else {
      f1 = ((1/(1-alpha))*((e_up^(1-alpha))-(e_low^(1-alpha))));
    }
    
    if(alpha == 2.0) {
      f2 = (log(e_up)-log(e_low));
    }
    else {
      f2 = ((1/(2-alpha))*((e_up^(2-alpha))-(e_low^(2-alpha))));
    }
    
    return (f1/f2);
  }
  
  real spectrum_logpdf(real E, real alpha, real e_low, real e_up) {

    real N;
    real p;

    if(alpha == 1.0) {
      N = (1.0/(log(e_up)-log(e_low)));
    }
    else {
      N = ((1.0-alpha)/((e_up^(1.0-alpha))-(e_low^(1.0-alpha))));
    }
  
    p = (N * pow(E, (alpha*-1)));

    return log(p);

  }

}

data {

  int N;
  vector[N] Edet;
  real D; // Mpc
  real Esrc_min;
  real Esrc_max;
  
}

transformed data {

  real D_m;

  D_m = D * 3.086e+22; // m

}

parameters {

  real<lower=0> L;
  real<lower=1, upper=4> alpha;
  vector<lower=Esrc_min, upper=Esrc_max> E;
  
}

transformed parameters {

  real F;

  F = (L / ( 4 * pi() * pow(D_m, 2) )) * flux_fac(alpha);
  
}

model {

  

}
