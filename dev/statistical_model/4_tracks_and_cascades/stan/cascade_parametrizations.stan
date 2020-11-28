// Utility

real eval_poly1d(real x, row_vector coeffs) {

    int N = num_elements(coeffs);
    real res=0;
    for(i in 1:N){
      res += coeffs[i]*pow(x, N-i);
    }
    return res;
}

vector get_pars(real x, matrix par_coeffs) {

    int N_pars = dims(par_coeffs)[1];
    vector[N_pars] pars;
    for (i in 1:N_pars) {
        pars[i] = eval_poly1d(x, par_coeffs[i]);
    }
    return pars;
}


// Basic PDFs and CDFs

real alpha_pdf(real x, real loc, real a) {

   real phi = normal_cdf(a, 0., 1.);
   return 1. / (pow(x-loc, 2.) * phi * sqrt(2.*pi())) * exp(-0.5 * pow((a-1 / (x-loc)), 2.));

}

real truncated_lognorm_pdf(real x, real scale, real s, real upper) {

    if (x > upper) {
        return 0.;
    }

    else {
        real y;
        real prob;
        real norm;

        y = x / scale;
        prob = exp(lognormal_lpdf(y | 0., s)) / scale;
        norm = lognormal_cdf(upper/scale, 0., s);
        print(norm)
        return prob / norm;
    }
}

real alpha_cdf(real x, real loc, real a) {

    if (x <= loc) {
        return 0;
    }
    else {
        real y = x-loc;
        real integral = erf((a - 1./y) / sqrt(2.)) / (2 * normal_cdf(a, 0., 1.));
        real C = -1 / (2 * normal_cdf(a, 0., 1.));
        return integral - C;
    }
}


// HESE renormalization

real erec_NC_normalization(real upper, vector pars, real enu) {

    // calculate cdf for alpha and truncated lognorm up to HESE cut
    real alpha = 1 - alpha_cdf(-upper, -pars[1], pars[2]);
    real lognorm;
    real normalization;
    if (upper > enu) {
        lognorm = 1.;
    }
    else {
        lognorm = lognormal_cdf(upper/pars[3], 0., pars[4]) / lognormal_cdf(enu/pars[3], 0., pars[4]);
    }

    normalization = pars[5] * (1-alpha) + (1-pars[5]) * (1-lognorm);
    return normalization;
}

real erec_CC_normalization(real upper, vector pars, real enu) {
    return 1. - lognormal_cdf(upper, 0., pars[2]) / pars[1];
}


// Erec PDFs with HESE cut

real CC_model(real erec, real enu, vector pars) {

    real logprob = lognormal_lpdf(erec/pars[1]| 0., pars[2]) - log(pars[1]);
    return logprob;
}

real NC_model(real erec, real enu, vector pars) {

    real alpha = alpha_pdf(-erec, -pars[1], pars[2]);
    real lognorm = truncated_lognorm_pdf(erec, pars[3], pars[4], enu);
    real prob = pars[5] * alpha + (1-pars[5]) * lognorm;
    return log(prob);
}

real erec_CC_lpdf(real erec, real enu, matrix par_coeffs, real HESE_cut) {

    if (erec < HESE_cut) {
        return log(0);
    }
    else {
        int N_pars = dims(par_coeffs)[1];
        vector[N_pars] pars = get_pars(enu, par_coeffs);

        real normalization = erec_CC_normalization(HESE_cut, pars, enu);
        return CC_model(erec, enu, pars) - log(normalization);
    }
}

real erec_NC_lpdf(real erec, real enu, matrix par_coeffs, real HESE_cut) {

    if (erec < HESE_cut) {
        return log(0);
    }
    // Erec can't be much bigger than Enu; ensure that
    else if (erec > 1.05 * enu) {
        return log(0);
    }
    else {
        int N_pars = dims(par_coeffs)[1];
        vector[N_pars] pars = get_pars(enu, par_coeffs);

        real normalization = erec_NC_normalization(HESE_cut, pars, enu);
        return NC_model(erec, enu, pars) - log(normalization);
    }
}
