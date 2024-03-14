functions {
    #include interpolation.stan
    #include utils.stan
}

data {
   int N;
   int L;
   array[N] real x;
   array[L] real test;
}

generated quantities {
   array[L] int search;
   for (i in 1:L) {
      search[i] = binary_search(test[i], x);
   }
}