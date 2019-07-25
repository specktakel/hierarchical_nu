/**
 * Useful tools to tidy up main code.
 *
 * @author Francesca Capel
 * @date April 2019
 */

/**
 * simple implementation of the trapezoidal rule.
 */
real trapz(vector x_values, vector y_values) {

  int N = num_elements(x_values);
  real I = 0;
  
  for (i in 1:N-1) {
    
    I += 0.5 * (x_values[i+1] - x_values[i]) * (y_values[i] + y_values[i+1]);

  }

  return I;

}

/**
 * linearly spaced vector between A and B of length N. 
 */
vector linspace(real A, real B, int N) {

  vector[N] output;
  real dx = (B - A) / (N - 1);

  for (i in 1:N) {
    output[i] = A + (i-1) * dx;
  }

  return output; 
}

/**
 * Evaluate a polynomial with given coefficients.
 * Highest power of x first.
 */
real eval_poly1d(real x, vector coeffs){
  int N = num_elements(coeffs);
  real res=0;
  for(i in 1:N){
    res += coeffs[i]*pow(x, N-i);
  }
  return res;
}
  
real truncate_value(real x, real min_val, real max_val){
     if(x < min_val){
         return min_val;
     }
     else if(x > max_val){
         return max_val;
     }
     return x;

}
