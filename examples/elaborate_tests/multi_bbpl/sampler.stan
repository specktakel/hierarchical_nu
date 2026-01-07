functions {
    #include interpolation.stan
    #include utils.stan
    #include rejection_sampling.stan
}

data {
    int N;
    int N_E;
    array[N] real slopes;
    array[N] real norms;
    vector[N]  weights;
    array[N+1] real breaks;
    array[N_E] real E;
}

transformed data {
   int N_samples = 100000;
}


generated quantities {
    array[N_samples] real samples;
    array[N_E] real pdf;
    for (i in 1:N_samples) {
        samples[i] = multiple_bbpl_rng(breaks, slopes, weights);
    }
    for (i in 1:N_E) {
        pdf[i] = multiple_bbpl_pdf(E[i], breaks, slopes, norms);
    }
}