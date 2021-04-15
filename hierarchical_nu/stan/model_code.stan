functions
{
#include interpolation.stan
#include utils.stan
#include vMF.stan
real CascadesEffAreaHist(real value_0,real value_1)
{
real hist_array[50,10] = {{ 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.20923161, 0.24187892, 0.27208374, 0.29281419, 0.30402686, 0.30554038,
   0.30335889, 0.29688418, 0.29308346, 0.2900113 },
 { 0.74473308, 0.89162445, 1.02927674, 1.12448363, 1.17300404, 1.18429344,
   1.17119329, 1.15218079, 1.13219335, 1.12421716},
 { 0.94853599, 1.19758453, 1.44037029, 1.60727459, 1.68936543, 1.70458244,
   1.68693729, 1.65378366, 1.63225993, 1.63920831},
 { 1.13685911, 1.5245375 , 1.9168453 , 2.19611221, 2.34343281, 2.36646892,
   2.33072282, 2.27345599, 2.24211599, 2.27620467},
 { 1.30140378, 1.85358722, 2.44590527, 2.88316994, 3.122603  , 3.15958627,
   3.09454794, 3.0259923 , 2.99002632, 3.05938084},
 { 1.41582805, 2.14818137, 2.98117394, 3.64471084, 4.01885205, 4.10284442,
   4.03107955, 3.9036897 , 3.87254771, 3.98512649},
 { 1.47861047, 2.40274814, 3.50684977, 4.44331108, 4.98825561, 5.15758861,
   5.08963326, 4.94403588, 4.88663536, 5.04712644},
 { 1.48935068, 2.55357776, 3.98045106, 5.22342357, 6.0102042 , 6.304851  ,
   6.23392667, 6.0773463 , 6.03101235, 6.24593385},
 { 1.45177915, 2.64512029, 4.32553169, 5.91046853, 7.01397692, 7.50621913,
   7.5303505 , 7.35149386, 7.25949309, 7.51658279},
 { 1.3759477 , 2.63723613, 4.53046853, 6.49759293, 7.96548639, 8.70102443,
   8.81528319, 8.6761755 , 8.59714327, 8.91011962},
 { 1.28747848, 2.54732412, 4.58648501, 6.90767392, 8.78397299, 9.84975935,
  10.14716286,10.03165292, 9.98670018,10.38114873},
 { 1.19524003, 2.4148046 , 4.5261785 , 7.11173961, 9.42439271,10.88719471,
  11.48832812,11.46903561,11.4381968 ,11.8503445 },
 { 1.12089174, 2.25062516, 4.35692203, 7.13192791, 9.89172374,11.90169028,
  12.81396022,12.96176843,12.96096073,13.39585398},
 { 1.05873077, 2.08193287, 4.08693515, 6.95035117,10.15631407,12.76740608,
  14.22309879,14.59791015,14.6230266 ,14.96488177},
 { 1.03534606, 1.92680157, 3.75978035, 6.70389705,10.28580239,13.63599454,
  15.70808603,16.51796894,16.45469569,16.52354107},
 { 1.05270641, 1.79274667, 3.40733916, 6.31094325,10.35476071,14.5324817 ,
  17.58194757,18.81580519,18.64102609,18.23553639},
 { 1.13589402, 1.70080718, 3.09243753, 5.88564961,10.34835886,15.48630455,
  19.70575519,21.7825348 ,21.33679461,20.05547798},
 { 1.2934017 , 1.64733412, 2.79510484, 5.51219974,10.27463032,16.5084601 ,
  22.3765342 ,25.67915499,24.97449907,22.29576085},
 { 1.03186456, 1.11751162, 1.76833102, 3.56115619, 6.98291793,11.83679918,
  17.03571576,20.32411115,19.58984726,16.67185277},
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        }};
real hist_edge_0[51] = {1.00000000e+02,1.38038426e+02,1.90546072e+02,2.63026799e+02,
 3.63078055e+02,5.01187234e+02,6.91830971e+02,9.54992586e+02,
 1.31825674e+03,1.81970086e+03,2.51188643e+03,3.46736850e+03,
 4.78630092e+03,6.60693448e+03,9.12010839e+03,1.25892541e+04,
 1.73780083e+04,2.39883292e+04,3.31131121e+04,4.57088190e+04,
 6.30957344e+04,8.70963590e+04,1.20226443e+05,1.65958691e+05,
 2.29086765e+05,3.16227766e+05,4.36515832e+05,6.02559586e+05,
 8.31763771e+05,1.14815362e+06,1.58489319e+06,2.18776162e+06,
 3.01995172e+06,4.16869383e+06,5.75439937e+06,7.94328235e+06,
 1.09647820e+07,1.51356125e+07,2.08929613e+07,2.88403150e+07,
 3.98107171e+07,5.49540874e+07,7.58577575e+07,1.04712855e+08,
 1.44543977e+08,1.99526231e+08,2.75422870e+08,3.80189396e+08,
 5.24807460e+08,7.24435960e+08,1.00000000e+09};
real hist_edge_1[11] = {-1. ,-0.8,-0.6,-0.4,-0.2, 0. , 0.2, 0.4, 0.6, 0.8, 1. };
return hist_array[binary_search(value_0, hist_edge_0)][binary_search(value_1, hist_edge_1)];
}
real CascadesAngularResolution(real true_energy,vector true_dir,vector reco_dir)
{
vector[6] CascadesAngularResolutionPolyCoeffs = [-4.84839608e-01, 3.59082699e+00, 4.39765349e+01,-4.86964043e+02,
  1.50499694e+03,-1.48474342e+03]';
return vMF_lpdf(reco_dir | true_dir, eval_poly1d(log10(truncate_value(true_energy, 100.0, 100000000.0)),CascadesAngularResolutionPolyCoeffs));
}
real c_energy_res_mix(real x,vector means,vector sigmas,vector weights)
{
vector[4] result;
for (i in 1:4)
{
result[i] = (log(weights)[i]+lognormal_lpdf(x | means[i], sigmas[i]));
}
return log_sum_exp(result);
}
real CascadeEnergyResolution(real true_energy,real reco_energy)
{
real CascadesEnergyResolutionMuPolyCoeffs[4,4] = {{ 6.78744003e-02,-1.13104870e+00, 6.91260682e+00,-1.02860172e+01},
 { 8.62985130e-03,-1.35891099e-01, 1.64945738e+00,-1.13266450e+00},
 { 7.55687311e-03,-1.41696079e-01, 1.88717443e+00,-1.87493829e+00},
 { 1.08066459e-04, 8.60033932e-04, 9.79384946e-01, 4.84697609e-02}};
real CascadesEnergyResolutionSdPolyCoeffs[4,4] = {{-6.35102894e-03, 1.13980940e-01,-6.54702500e-01, 1.26738248e+00},
 {-1.22445964e-03, 2.07342567e-02,-1.10650640e-01, 2.11711378e-01},
 {-1.14645231e-03, 2.09051849e-02,-1.26572454e-01, 2.64462273e-01},
 { 6.69865237e-13,-1.11718576e-11, 5.87059333e-11, 9.99999991e-03}};
real mu_e_res[4];
real sigma_e_res[4];
vector[4] weights;
for (i in 1:4)
{
weights[i] = 1.0/4;
}
for (i in 1:4)
{
mu_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 30000.00000000001, 10000000.0)), to_vector(CascadesEnergyResolutionMuPolyCoeffs[i]));
sigma_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 30000.00000000001, 10000000.0)), to_vector(CascadesEnergyResolutionSdPolyCoeffs[i]));
}
return c_energy_res_mix(log10(reco_energy), to_vector(log(mu_e_res)), to_vector(sigma_e_res), weights);
}
real CascadesEffectiveArea(real true_energy,vector true_dir)
{
return CascadesEffAreaHist(true_energy, cos(pi() - acos(true_dir[3])));
}
real src_spectrum_logpdf(real E,real alpha,real e_low,real e_up)
{
real N;
real p;
if(alpha == 1.0)
{
N = (1.0/(log(e_up)-log(e_low)));
}
else
{
N = ((1.0-alpha)/((e_up^(1.0-alpha))-(e_low^(1.0-alpha))));
}
p = (N*pow(E, (alpha*-1)));
return log(p);
}
real diff_spectrum_logpdf(real E,real alpha,real e_low,real e_up)
{
real N;
real p;
if(alpha == 1.0)
{
N = (1.0/(log(e_up)-log(e_low)));
}
else
{
N = ((1.0-alpha)/((e_up^(1.0-alpha))-(e_low^(1.0-alpha))));
}
p = (N*pow(E, (alpha*-1)));
return log(p);
}
real flux_conv(real alpha,real e_low,real e_up)
{
real f1;
real f2;
if(alpha == 1.0)
{
f1 = (log(e_up)-log(e_low));
}
else
{
f1 = ((1/(1-alpha))*((e_up^(1-alpha))-(e_low^(1-alpha))));
}
if(alpha == 2.0)
{
f2 = (log(e_up)-log(e_low));
}
else
{
f2 = ((1/(2-alpha))*((e_up^(2-alpha))-(e_low^(2-alpha))));
}
return (f1/f2);
}
}
data
{
int N;
unit_vector[3] omega_det[N];
vector[N] Edet;
vector[N] kappa;
real Esrc_min;
real Esrc_max;
int Ns;
unit_vector[3] varpi[Ns];
vector[Ns] D;
vector[Ns+1] z;
int Ngrid;
vector[Ngrid] src_index_grid;
vector[Ngrid] diff_index_grid;
vector[Ngrid] integral_grid[Ns+1];
vector[Ngrid] E_grid;
vector[Ngrid] Pdet_grid[Ns+1];
real T;
real L_scale;
real F_diff_scale;
real F_tot_scale;
}
parameters
{
real<lower=0, upper=1e+60> L;
real<lower=0.0, upper=0.0001> F_diff;
real<lower=1, upper=4> src_index;
real<lower=1, upper=4> diff_index;
vector<lower=Esrc_min, upper=Esrc_max> [N] Esrc;
}
transformed parameters
{
real Fsrc;
vector[Ns+1] F;
vector[Ns+1] eps;
vector[Ns+1] lp[N];
vector[Ns+1] logF;
real<lower=0, upper=1> f;
real<lower=0> Ftot;
real Nex;
vector[N] E;
Fsrc = 0.0;
for (k in 1:Ns)
{
F[k] = L/ (4 * pi() * pow(D[k] * 3.086e+22, 2));
F[k]*=flux_conv(src_index, Esrc_min, Esrc_max);
Fsrc+=F[k];
}
F[Ns+1] = F_diff;
Ftot = (F_diff+Fsrc);
f = Fsrc / Ftot;
logF = log(F);
for (i in 1:N)
{
lp[i] = logF;
for (k in 1:Ns+1)
{
if(k < (Ns+1))
{
lp[i][k] += src_spectrum_logpdf(Esrc[i], src_index, Esrc_min, Esrc_max);
E[i] = Esrc[i] / ((1+z[k]));
lp[i][k] += vMF_lpdf(omega_det[i] | varpi[k], kappa[i]);
}
else if(k == (Ns+1))
{
lp[i][k] += diff_spectrum_logpdf(Esrc[i], diff_index, Esrc_min, Esrc_max);
E[i] = Esrc[i] / ((1+z[k]));
lp[i][k] += -2.5310242469692907;
}
lp[i][k] += CascadeEnergyResolution(E[i], Edet[i]);
lp[i][k] += log(interpolate(E_grid, Pdet_grid[k], E[i]));
}
}
eps = get_exposure_factor(src_index, diff_index, src_index_grid, diff_index_grid, integral_grid, T, Ns);
Nex = get_Nex(F, eps);
}
model
{
for (i in 1:N)
{
target += log_sum_exp(lp[i]);
}
target += -Nex;
L ~ normal(0, L_scale);
F_diff ~ normal(0, F_diff_scale);
Ftot ~ normal(F_tot_scale, (0.5*F_tot_scale));
src_index ~ normal(2.0, 2.0);
diff_index ~ normal(2.0, 2.0);
}
