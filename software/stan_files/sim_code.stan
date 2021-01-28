functions
{
#include utils.stan
#include vMF.stan
#include interpolation.stan
#include sim_functions.stan
real CascadesEffAreaHist(real value_0,real value_1)
{
real hist_array[50,10] = {{ 0.63933587, 0.74699076, 0.84415315, 0.91115512, 0.9434195 , 0.9526226 ,
   0.94245776, 0.92420666, 0.91262234, 0.90292905},
 { 0.75312694, 0.89986591, 1.03903315, 1.13992773, 1.18523385, 1.19512779,
   1.18558944, 1.16613617, 1.14683526, 1.13521031},
 { 0.86668994, 1.07171631, 1.2735945 , 1.40310027, 1.47384048, 1.48576319,
   1.46620855, 1.44214093, 1.41817148, 1.41988222},
 { 0.98820933, 1.25923488, 1.52123705, 1.70465396, 1.79688734, 1.8127543 ,
   1.79585598, 1.75837364, 1.73733324, 1.74783073},
 { 1.09429869, 1.44056537, 1.79056968, 2.03835659, 2.16482787, 2.19305007,
   2.16087646, 2.11401468, 2.08067401, 2.11080854},
 { 1.19647092, 1.6385902 , 2.09031435, 2.41759943, 2.59559062, 2.61650251,
   2.57308843, 2.51067956, 2.47742148, 2.51605643},
 { 1.28818452, 1.82647011, 2.39515069, 2.81357161, 3.0379943 , 3.08294166,
   3.01685881, 2.94909065, 2.91483817, 2.98306113},
 { 1.36308365, 2.00150268, 2.71182579, 3.25100182, 3.54864924, 3.59615912,
   3.53032646, 3.42431571, 3.40514701, 3.47563081},
 { 1.42434438, 2.16944097, 3.01875345, 3.6913052 , 4.06835016, 4.1516009 ,
   4.08754622, 3.96778443, 3.91636003, 4.04749926},
 { 1.46501711, 2.31878336, 3.31918235, 4.15257896, 4.63066475, 4.75837482,
   4.70553388, 4.55143041, 4.50651146, 4.64500107},
 { 1.48570103, 2.44800509, 3.60953808, 4.61057054, 5.20071081, 5.39452534,
   5.30402955, 5.16075833, 5.1101743 , 5.28554261},
 { 1.4924773 , 2.52906496, 3.89618125, 5.0656722 , 5.7863571 , 6.05304606,
   5.97780527, 5.81623624, 5.75598427, 5.97406881},
 { 1.4807675 , 2.59340023, 4.12604982, 5.48506006, 6.3907092 , 6.73131402,
   6.68355493, 6.52482022, 6.48148587, 6.69992242},
 { 1.45973425, 2.64374766, 4.31722793, 5.87204838, 6.95289258, 7.42156215,
   7.42791468, 7.25978499, 7.17707707, 7.42223696},
 { 1.41995373, 2.65391585, 4.43347491, 6.21345543, 7.49716037, 8.11406254,
   8.1797899 , 8.00457225, 7.91195588, 8.21121341},
 { 1.36877211, 2.63523327, 4.55009145, 6.55900729, 8.05323634, 8.81563625,
   8.93728358, 8.77860548, 8.71242469, 9.03219747},
 { 1.31928717, 2.5908894 , 4.58991594, 6.78727526, 8.53581672, 9.45310066,
   9.71765285, 9.61139485, 9.51436578, 9.89765037},
 { 1.26328218, 2.51827032, 4.58601699, 6.98431335, 8.94300039,10.11174716,
  10.42085247,10.33693256,10.30607634,10.69869428},
 { 1.21228083, 2.44792011, 4.54371981, 7.09316662, 9.29614525,10.6802837 ,
  11.21932132,11.17439373,11.16694532,11.54508155},
 { 1.16351122, 2.35402482, 4.47955415, 7.13889074, 9.64854924,11.30671618,
  12.01818996,12.02007764,11.99092452,12.41127302},
 { 1.12318372, 2.25769788, 4.36952198, 7.13019295, 9.89018746,11.86036594,
  12.77026956,12.91759393,12.91532639,13.37284986},
 { 1.08462693, 2.16207993, 4.21923882, 7.0526861 ,10.03775776,12.35846345,
  13.49874654,13.81220965,13.82612997,14.22496605},
 { 1.05158219, 2.06445884, 4.05685351, 6.93969285,10.19431461,12.86189222,
  14.39488329,14.75876209,14.80611133,15.1229516 },
 { 1.04038193, 1.96763407, 3.86702487, 6.78289188,10.28525027,13.3487905 ,
  15.21131168,15.8822746 ,15.79212598,16.03052558},
 { 1.03411561, 1.89216214, 3.67457179, 6.63130032,10.28894141,13.86858935,
  16.12638944,17.05741609,16.99217049,16.91292328},
 { 1.04452452, 1.8144775 , 3.46682629, 6.37184973,10.35581405,14.37225355,
  17.25455995,18.34809773,18.2527227 ,17.92464898},
 { 1.07607433, 1.74667345, 3.26914609, 6.14688837,10.34999682,14.90247159,
  18.35974573,19.91647616,19.56958117,18.96266118},
 { 1.13098901, 1.70274093, 3.10354625, 5.89321637,10.35730956,15.45701403,
  19.65364479,21.64562864,21.26012857,20.06535467},
 { 1.2070938 , 1.65809431, 2.92266618, 5.6775237 ,10.30375921,16.07126467,
  21.03936989,23.73097227,23.19429886,21.16503933},
 { 1.31670727, 1.64549403, 2.75928324, 5.45349351,10.2680503 ,16.62876047,
  22.78455798,26.27179056,25.51733177,22.60677157},
 { 1.48301192, 1.63914536, 2.61834626, 5.27933923,10.25672602,17.28163126,
  24.60271648,29.23541931,28.23802368,24.19707741},
 { 0.5270135 , 0.55368765, 0.85987952, 1.73218165, 3.44363392, 5.89115051,
   8.64887516,10.36626977, 9.93317012, 8.37561911},
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
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        }};
real hist_edge_0[51] = {3.00000000e+04,3.61059544e+04,4.34546648e+04,5.22990716e+04,
 6.29435965e+04,7.57546209e+04,9.11730963e+04,1.09729722e+05,
 1.32063211e+05,1.58942276e+05,1.91292086e+05,2.30226112e+05,
 2.77084450e+05,3.33479951e+05,4.01353730e+05,4.83041983e+05,
 5.81356394e+05,6.99680915e+05,8.42088241e+05,1.01347999e+06,
 1.21975541e+06,1.46801444e+06,1.76680208e+06,2.12640252e+06,
 2.55919308e+06,3.08007029e+06,3.70696258e+06,4.46144740e+06,
 5.36949388e+06,6.46235672e+06,7.77765190e+06,9.36065150e+06,
 1.12658419e+07,1.35587991e+07,1.63184461e+07,1.96397690e+07,
 2.36370869e+07,2.84479860e+07,3.42380563e+07,4.12065900e+07,
 4.95934420e+07,5.96872852e+07,7.18355466e+07,8.64563657e+07,
 1.04052987e+08,1.25231080e+08,1.50719589e+08,1.81395820e+08,
 2.18315640e+08,2.62749819e+08,3.16227766e+08};
real hist_edge_1[11] = {-1. ,-0.8,-0.6,-0.4,-0.2, 0. , 0.2, 0.4, 0.6, 0.8, 1. };
return hist_array[binary_search(value_0, hist_edge_0)][binary_search(value_1, hist_edge_1)];
}
real spectrum_rng(real alpha,real e_low,real e_up)
{
real uni_sample;
real norm;
norm = ((1-alpha)/((e_up^(1-alpha))-(e_low^(1-alpha))));
uni_sample = uniform_rng(0, 1);
return ((((uni_sample*(1-alpha))/norm)+(e_low^(1-alpha)))^(1/(1-alpha)));
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
vector CascadesAngularResolution_rng(real true_energy,vector true_dir)
{
vector[6] CascadesAngularResolutionPolyCoeffs = [-4.84839608e-01, 3.59082699e+00, 4.39765349e+01,-4.86964043e+02,
  1.50499694e+03,-1.48474342e+03]';
return vMF_rng(true_dir, eval_poly1d(log10(truncate_value(true_energy, 100.0, 100000000.0)),CascadesAngularResolutionPolyCoeffs));
}
real c_energy_res_mix_rng(vector means,vector sigmas,vector weights)
{
int index;
index = categorical_rng(weights);
return lognormal_rng(means[index], sigmas[index]);
}
real CascadeEnergyResolution_rng(real true_energy)
{
real CascadesEnergyResolutionMuPolyCoeffs[4,4] = {{ 6.76407765e-02,-1.12716962e+00, 6.89131295e+00,-1.02474116e+01},
 { 8.54785344e-03,-1.34538541e-01, 1.64205931e+00,-1.11928361e+00},
 { 7.55433894e-03,-1.41657645e-01, 1.88698773e+00,-1.87465399e+00},
 { 1.07971639e-04, 8.61965387e-04, 9.79372612e-01, 4.84957277e-02}};
real CascadesEnergyResolutionSdPolyCoeffs[4,4] = {{-6.26109056e-03, 1.12494521e-01,-6.46590841e-01, 1.25275718e+00},
 {-1.21679598e-03, 2.06090980e-02,-1.09975316e-01, 2.10508185e-01},
 {-1.14642242e-03, 2.09050768e-02,-1.26574688e-01, 2.64473098e-01},
 { 8.20302891e-11,-1.40009717e-09, 7.85263331e-09, 9.99998556e-03}};
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
return c_energy_res_mix_rng(to_vector(log(mu_e_res)), to_vector(sigma_e_res), weights);
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
real CascadesEnergyResolutionMuPolyCoeffs[4,4] = {{ 6.76407765e-02,-1.12716962e+00, 6.89131295e+00,-1.02474116e+01},
 { 8.54785344e-03,-1.34538541e-01, 1.64205931e+00,-1.11928361e+00},
 { 7.55433894e-03,-1.41657645e-01, 1.88698773e+00,-1.87465399e+00},
 { 1.07971639e-04, 8.61965387e-04, 9.79372612e-01, 4.84957277e-02}};
real CascadesEnergyResolutionSdPolyCoeffs[4,4] = {{-6.26109056e-03, 1.12494521e-01,-6.46590841e-01, 1.25275718e+00},
 {-1.21679598e-03, 2.06090980e-02,-1.09975316e-01, 2.10508185e-01},
 {-1.14642242e-03, 2.09050768e-02,-1.26574688e-01, 2.64473098e-01},
 { 8.20302891e-11,-1.40009717e-09, 7.85263331e-09, 9.99998556e-03}};
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
}
data
{
int Ns;
unit_vector[3] varpi[Ns];
vector[Ns] D;
vector[Ns+1] z;
real alpha;
real Edet_min;
real Esrc_min;
real Esrc_max;
real L;
real F_diff;
int Ngrid;
vector[Ngrid] alpha_grid;
vector[Ngrid] integral_grid[Ns+1];
real aeff_max;
real v_lim;
real T;
}
transformed data
{
vector[Ns+1] F;
simplex[Ns+1] w_exposure;
vector[Ns+1] eps;
int track_type;
int cascade_type;
real Ftot;
real Fs;
real f;
real Nex;
int N;
track_type = 0;
cascade_type = 1;
Fs = 0.0;
for (k in 1:Ns)
{
F[k] = L/ (4 * pi() * pow(D[k] * 3.086e+22, 2));
F[k]*=flux_conv(alpha, Esrc_min, Esrc_max);
Fs += F[k];
}
F[Ns+1] = F_diff;
Ftot = (Fs+F_diff);
f = Fs/Ftot;
print("f: ", f);
eps = get_exposure_factor(alpha, alpha_grid, integral_grid, T, Ns);
Nex = get_Nex(F, eps);
w_exposure = get_exposure_weights(F, eps);
N = poisson_rng(Nex);
print(w_exposure);
print(Ngrid);
print(Nex);
print(N);
}
generated quantities
{
int Lambda[N];
unit_vector[3] omega;
vector[N] Esrc;
vector[N] E;
vector[N] Edet;
real cosz[N];
real Pdet[N];
int accept;
int detected;
int ntrials;
simplex[2] prob;
unit_vector[3] event[N];
real Nex_sim;
vector[N] event_type;
Nex_sim = Nex;
for (i in 1:N)
{
Lambda[i] = categorical_rng(w_exposure);
accept = 0;
detected = 0;
ntrials = 0;
while((accept!=1))
{
if(Lambda[i] <= Ns)
{
omega = varpi[Lambda[i]];
}
else if(Lambda[i] == (Ns+1))
{
omega = sphere_lim_rng(1, v_lim);
}
cosz[i] = cos(omega_to_zenith(omega));
if(Lambda[i] <= (Ns+1))
{
Esrc[i] = spectrum_rng(alpha, Esrc_min, Esrc_max);
E[i] = (Esrc[i]/(1+z[Lambda[i]]));
}
Pdet[i] = (CascadesEffectiveArea(E[i], omega)/aeff_max);
Edet[i] = (10^CascadeEnergyResolution_rng(E[i]));
prob[1] = Pdet[i];
prob[2] = (1-Pdet[i]);
ntrials += 1;
if(ntrials< 1000000)
{
detected = categorical_rng(prob);
if((Edet[i] >= Edet_min) && ((detected==1)))
{
accept = 1;
}
}
else
{
accept = 1;
print("problem component: ", Lambda[i]);
;
}
}
event[i] = CascadesAngularResolution_rng(E[i], omega);
event_type[i] = cascade_type;
}
}
