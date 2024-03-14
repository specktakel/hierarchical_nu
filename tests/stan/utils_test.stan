functions {
    #include interpolation.stan
    #include utils.stan
}

data {
    int N;
    vector[N] F;
    vector[N] eps;
    int N_vec;
    array[N_vec] vector[3] omega;
}


generated quantities {
    vector[N] weights;
    vector[N] Nex;
    weights = get_exposure_weights(F, eps);
    Nex = get_Nex_vec(F, eps);
    array[N_vec] real zenith;
    array[N_vec] real angle;
    array[N_vec] real dec;
    for (i in 1:N_vec) {
        zenith[i] = omega_to_zenith(omega[i]);
        dec[i] = omega_to_dec(omega[i]);
        angle[i] = ang_sep(omega[i], omega[1]);
    }
}