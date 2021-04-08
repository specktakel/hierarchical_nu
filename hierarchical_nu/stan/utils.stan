/**
 * Useful tools to tidy up main code.
 *
 * @author Francesca Capel
 * @date April 2019
 */


/**
 * Get exposure factor from integral grids and source info.
 * Includes atmospheric component with known index. Units of [m^2 s].
 */
vector get_exposure_factor_atmo(real src_index, real diff_index, vector src_index_grid, vector diff_index_grid,
				vector[] integral_grid, real atmo_integ_val, real T, int Ns) {

  vector[Ns+2] eps;

  /* Point sources */
  for (k in 1:Ns) {

    eps[k] = interpolate(src_index_grid, integral_grid[k], src_index);
      
  }

  /* Diffuse component */
  eps[Ns+1] = interpolate(diff_index_grid, integral_grid[Ns+1], diff_index);

  /* Atmospheric component */
  eps[Ns+2] = atmo_integ_val;

  return eps * T;
}

/**
 * Get exposure factor from integral grids and source info.
 * Units of [m^2 s].
 */
vector get_exposure_factor(real src_index, real diff_index, vector src_index_grid, vector diff_index_grid,
			   vector[] integral_grid, real T, int Ns) {

  vector[Ns+1] eps;

  /* Point sources */
  for (k in 1:Ns) {

    eps[k] = interpolate(src_index_grid, integral_grid[k], src_index);
      
  }

  eps[Ns+1] = interpolate(diff_index_grid, integral_grid[Ns+1], diff_index);

  return eps * T;
}


/**
 * For use in simple one-component sims
 */
real get_eps_simple(real index, vector index_grid, vector integral_grid, real T) {

  real eps;

  eps = interpolate(index_grid, integral_grid, index);

  return eps * T;

}

/**
 * Calculate weights from exposure integral.
 */
vector get_exposure_weights(vector F, vector eps) {

  int K = num_elements(eps);
  vector[K] weights;
    
  real normalisation = 0;
  
  for (k in 1:K) {
    normalisation += F[k] * eps[k];
  }

  for (k in 1:K) {
    weights[k] = F[k] * eps[k] / normalisation;
  }

  return weights;
}
  

/**
 * Convert from unit vector omega to theta of spherical coordinate system.
 * @param omega a 3D unit vector.
 */
real omega_to_zenith(vector omega) {
  
  real zenith;
  
  int N = num_elements(omega);
  
  if (N != 3) {
    print("Error: input vector omega must be of 3 dimensions");
  }

  zenith = pi() - acos(omega[3]);
    
  return zenith;
}

/**
 * Calculate the expected number of detected events from each source.
 */
real get_Nex(vector F, vector eps) {
  
  int K = num_elements(eps);
  real Nex = 0;
  
  for (k in 1:K) {
    Nex += F[k] * eps[k];
  }
  
  return Nex;
}


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

real[] generate_bin_edges(real lower_edge, real upper_edge, int nbins)
{

    real binedges[nbins+1];
    real binwidth = (upper_edge-lower_edge)/nbins;
    for (i in 1:nbins+1)
    {
        binedges[i] = lower_edge + (i-1)*binwidth;
    }
    return binedges;
}

int binary_search(real value, real[] binedges)
{
    int L = 1;
    int R = size(binedges);
    int m;
    if (value < binedges[1])
        return 0;
    else if(value > binedges[R])
        return R+1;
    else{
        while (L < R-1)
        {
            m = (L + R) / 2;
            if (binedges[m] < value)
                L = m;
            else if (binedges[m] > value)
                R = m;
            else
                return m;
        }
    }
    return L;
}
