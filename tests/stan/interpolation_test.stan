functions {
    #include interpolation.stan
    #include utils.stan
}

data {
   int N;
   int L;
   vector[N] x;
   vector[N] y;
   array[L] real test;
}

generated quantities {
   array[L] real interpolated;
   array[L] real log_interpolated;
   array[L] real interpolated_2bins;
   for (i in 1:L) {
      interpolated[i] = interpolate(x, y, test[i]);
      log_interpolated[i] = interpolate_log_y(x, y, test[i]);
      interpolated_2bins[i] = interpolate_2bins([x[1], x[10]]', [y[1], y[10]]', test[i]);
   }
}