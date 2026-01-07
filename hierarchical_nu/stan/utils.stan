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
				array[] vector integral_grid, real atmo_integ_val, real T, int Ns) {

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
			   array[] vector integral_grid, real T, int Ns) {

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
  vector[K] weights = F .* eps;
  return weights / sum(weights);
}

vector get_exposure_weights_from_Nex_et(array[] real Nex) {
  int K = num_elements(Nex);
  vector[K] weights = to_vector(Nex) / sum(Nex);
  return weights;
}
  
/**
 * Integral over power law without normalisation, i.e. \int_x1^x2 x^{-gamma}
 */
real Ngamma(real gamma, real n, real x0, real x1, real x2) {

  real idx = n + 1. - gamma;
  if (n + 1. == gamma) {
    return x0^(n+1.) * log(x2/x1);
  }
  else {
    return x0^gamma * (x2^idx - x1^idx) / idx;
  }
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
 * Convert from unit vector omega to declination of spherical coordinate system.
 * @param omega a 3D unit vector.
 */
real omega_to_dec(vector omega) {
  
  real dec;
  
  int N = num_elements(omega);
  
  if (N != 3) {
    print("Error: input vector omega must be of 3 dimensions");
  }

  dec = pi() / 2 - acos(omega[3]);
    
  return dec;
}

/**
 * Calculate the expected number of detected events from each source.
 */
real get_Nex(vector F, vector eps) {

  return sum(F .* eps);
}

vector get_Nex_vec(vector F, vector eps) {

  return F .* eps;
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
  array[N] real res;
  for(i in 1:N){
    res[i] = coeffs[i]*pow(x, N-i);
  }
  return sum(res);
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

array[] real generate_bin_edges(real lower_edge, real upper_edge, int nbins)
{

    array[nbins+1] real binedges;
    real binwidth = (upper_edge-lower_edge)/nbins;
    for (i in 1:nbins+1)
    {
        binedges[i] = lower_edge + (i-1)*binwidth;
    }
    return binedges;
}

int binary_search(real value, array[] real binedges)
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
            m = (L + R) %/% 2;
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


/**
 * Histogram rng, takes n, bins as arguments
 */
real histogram_rng(array[] real hist_array, array[] real hist_edges)
{
    array[size(hist_array)] real bin_width;
    array[size(hist_array)] real multiplied;
    vector[size(hist_array)] normalised;
    int index;
    for (i in 2:size(hist_edges)) {
        bin_width[(i-1)] = hist_edges[i] - hist_edges[i-1];
    }
    for (i in 1:size(hist_array)) {
      if (hist_array[i] > 0.) {
        multiplied[i] = hist_array[i] * bin_width[i];
      }
      else if (hist_array[i] == 0.) {
        multiplied[i] = 0.;
      }
    }
    for (i in 1:size(hist_array)) {
        normalised[i] = multiplied[i] / sum(multiplied);
    }
    index = categorical_rng(normalised);
    return uniform_rng(hist_edges[index], hist_edges[index+1]);
}

/**
 * Categorical rng, takes n, bins as arguments.
 * Like Histogram_rng, but returns the bin index instead of a sampled value.
 */

int hist_cat_rng(array[] real hist_array, array[] real hist_edges)
{
    array[size(hist_array)] real bin_width;
    array[size(hist_array)] real multiplied;
    vector[size(hist_array)] normalised;
    int index;
    for (i in 2:size(hist_edges)) {
        bin_width[(i-1)] = hist_edges[i] - hist_edges[i-1];
    }
    for (i in 1:size(hist_array)) {
      if (hist_array[i] > 0.) {
        multiplied[i] = hist_array[i] * bin_width[i];
      }
      else if (hist_array[i] == 0.) {
        multiplied[i] = 0.;
      }
    }
    for (i in 1:size(hist_array)) {
        normalised[i] = multiplied[i] / sum(multiplied);
    }
    return categorical_rng(normalised);
}

/**
 * Angular separation on a sphere
 */
real ang_sep(vector vec1, vector vec2)
{
    return acos(sum(vec1 .* vec2));
}
