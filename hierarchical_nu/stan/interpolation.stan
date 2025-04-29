/**
 * Stan functions for interpolation.
 * 
 * @author Francesca Capel
 * @date February 2019
 */

/**
 * Interpolate x from a given set of x and y values.
 * Prints warning if x is outside of the interpolation range.
 */
real interpolate(vector x_values, vector y_values, real x) {

  real x_left;
  real y_left;
  real x_right;
  real y_right;
  real dydx;
  
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];
  int i = 1;

  if (x > xmax || x < xmin) {

    /*
    print("Warning, x is outside of interpolation range!");
    print("Returning edge values.");
    print("x:", x);
    print("xmax", xmax);
    */
    
    if(x > xmax) {
      return y_values[Nx];
    }
    else if (x < xmin) {
      return y_values[1];
    }
  }
    
  if( x >= x_values[Nx - 1] ) {
    i = Nx - 1;
  }
  else {
    i = binary_search(x, to_array_1d(x_values));
  }

  x_left = x_values[i];
  y_left = y_values[i];
  x_right = x_values[i + 1];
  y_right = y_values[i + 1];
  
  dydx = (y_right - y_left) / (x_right - x_left);
    
  return y_left + dydx * (x - x_left);
}

real interpolate(array[] real x_values, array[] real y_values, real x) {

  real x_left;
  real y_left;
  real x_right;
  real y_right;
  real dydx;
  
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];
  int i = 1;

  if (x > xmax || x < xmin) {

    /*
    print("Warning, x is outside of interpolation range!");
    print("Returning edge values.");
    print("x:", x);
    print("xmax", xmax);
    */
    
    if(x > xmax) {
      return y_values[Nx];
    }
    else if (x < xmin) {
      return y_values[1];
    }
  }
    
  if( x >= x_values[Nx - 1] ) {
    i = Nx - 1;
  }
  else {
    i = binary_search(x, x_values);
  }

  x_left = x_values[i];
  y_left = y_values[i];
  x_right = x_values[i + 1];
  y_right = y_values[i + 1];
  
  dydx = (y_right - y_left) / (x_right - x_left);
    
  return y_left + dydx * (x - x_left);
}

/**
 *like interpolate, only return eponentiated values.
 */
real interpolate_log_y(vector x_values, vector log_y_values, real x) {

  real x_left;
  real y_left;
  real x_right;
  real y_right;
  real dydx;
  
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];
  int i = 1;

  if (x > xmax || x < xmin) {

    /*
    print("Warning, x is outside of interpolation range!");
    print("Returning edge values.");
    print("x:", x);
    print("xmax", xmax);
    */
    
    if(x > xmax) {
      return exp(log_y_values[Nx]);
    }
    else if (x < xmin) {
      return exp(log_y_values[1]);
    }
  }
    
  if( x >= x_values[Nx - 1] ) {
    i = Nx - 1;
  }
  else {
    i = binary_search(x, to_array_1d(x_values));
  }

  x_left = x_values[i];
  y_left = log_y_values[i];
  x_right = x_values[i + 1];
  y_right = log_y_values[i + 1];
  
  dydx = (y_right - y_left) / (x_right - x_left);
    
  return exp(y_left + dydx * (x - x_left));
}


real interpolate_2bins(vector x_values, vector y_values, real x) {

  real x_left = x_values[1];
  real y_left = y_values[1];
  real x_right = x_values[2];
  real y_right = y_values[2];

  if (x >= x_right || x <= x_left) {

    /*
    print("Warning, x is outside of interpolation range!");
    print("Returning edge values.");
    print("x:", x);
    print("xmax", xmax);
    */
    
    if(x >= x_right) {
      return y_right;
    }
    else if (x <= x_left) {
      return y_left;
    }
  }
  
  real dydx = (y_right - y_left) / (x_right - x_left);
    
  return y_left + dydx * (x - x_left);
}


real interp2d(real x, real y, array[] real xp, array[] real yp, array[,] real fp) {
  /*
  Interpolation on a 2d grid.
  xp and yp should be the points at which fp is evaluated.
  If some point (x, y) is outside the domain, the values along the
  respective boarder are returned.
  */
  int idx_y = binary_search(y, yp);
  int idx_yp1 = idx_y + 1;
  //safeguard against y values outside the defined range
  // interpolate will take care of the same issue in x direction
  if (idx_y == 0) {
    // return result from lowest slice
    return interpolate(to_vector(xp), to_vector(fp[:, 1]), x);
  }
  else if (idx_y >= size(yp)) {
    return interpolate(to_vector(xp), to_vector(fp[:, size(yp)]), x);
  }
  real y_vals_low = interpolate(to_vector(xp), to_vector(fp[:, idx_y]), x);
  real y_vals_high = interpolate(to_vector(xp), to_vector(fp[:, idx_yp1]), x);
  real val = interpolate(to_vector(yp[idx_y:idx_yp1]), [y_vals_low, y_vals_high]', y);
  return val;
}

real interp2d_reduced(int idx_x, real x, real y, array[] real xp, array[] real yp, array[,] real fp) {
  int idx_y = binary_search(y, yp);
  int idx_yp1 = idx_y + 1;
  int idx_xp1 = idx_x + 1;
  vector[2] x_slice = to_vector(xp[idx_x:idx_xp1]);
  //safeguard against y values outside the defined range
  // interpolate will take care of the same issue in x direction
  if (idx_y == 0) {
    // return result from lowest slice
    return interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, 1]), x);
  }
  else if (idx_y >= size(yp)) {
    return interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, size(yp)]), x);
  }
  real y_vals_low = interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, idx_y]), x);
  real y_vals_high = interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, idx_yp1]), x);
  real val = interpolate_2bins(to_vector(yp[idx_y:idx_yp1]), [y_vals_low, y_vals_high]', y);
  return val;
}
real interp2dlog_logy(real x, real y, array[] real xp, array[] real yp, array[,] real fp) {
  real logy = log(y);
  int idx_y = binary_search(logy, yp);
  int idx_yp1 = idx_y + 1;
  int idx_x = binary_search(x, xp);
  int idx_xp1 = idx_x + 1;
  vector[2] x_slice = to_vector(xp[idx_x:idx_xp1]);
  //safeguard against y values outside the defined range
  // interpolate will take care of the same issue in x direction
  if (idx_y == 0) {
    // return result from lowest slice
    return interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, 1]), x);
  }
  else if (idx_y >= size(yp)) {
    return interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, size(yp)]), x);
  }
  real y_vals_low = interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, idx_y]), x);
  real y_vals_high = interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, idx_yp1]), x);
  real val = interpolate_2bins(to_vector(yp[idx_y:idx_yp1]), [y_vals_low, y_vals_high]', logy);
  return exp(val);
}

real interp2dlog(real x, real y, array[] real xp, array[] real yp, array[,] real fp) {
  int idx_y = binary_search(y, yp);
  int idx_yp1 = idx_y + 1;
  int idx_x = binary_search(x, xp);
  int idx_xp1 = idx_x + 1;
  vector[2] x_slice = to_vector(xp[idx_x:idx_xp1]);
  //safeguard against y values outside the defined range
  // interpolate will take care of the same issue in x direction
  if (idx_y == 0) {
    // return result from lowest slice
    return interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, 1]), x);
  }
  else if (idx_y >= size(yp)) {
    return interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, size(yp)]), x);
  }
  real y_vals_low = interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, idx_y]), x);
  real y_vals_high = interpolate_2bins(x_slice, to_vector(fp[idx_x:idx_xp1, idx_yp1]), x);
  real val = interpolate_2bins(to_vector(yp[idx_y:idx_yp1]), [y_vals_low, y_vals_high]', y);
  return exp(val);
}