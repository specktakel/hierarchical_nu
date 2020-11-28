// Simple script to test the CC parametrization


functions {

#include ./cascade_parametrizations.stan

}

data {

    int N;
    vector[N] x;
    int N_pars;
    int N_degree_max;
    matrix[N_pars, N_degree_max] par_coeffs;
    real enu;
}

generated quantities {

    vector[N] y;
    vector[N] norm;
    vector[N_pars] pars;
    vector[N] alpha;
    vector[N] lognorm;

    for (i in 1:N) {

        real HESE = log10(6e4);
        y[i] = erec_CC_lpdf(x[i]| enu, par_coeffs, HESE);
    }
}
