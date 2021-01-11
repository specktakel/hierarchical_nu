functions
{
#include utils.stan
#include vMF.stan
#include interpolation.stan
#include sim_functions.stan
real CascadesEffAreaHist(real value_0,real value_1)
{
real hist_array[50,10] = {{  4.12296733,  4.84904949,  5.5159688 ,  5.98461963,  6.2264291 ,
    6.32061887,  6.29151569,  6.24204368,  6.24329538,  6.28597893},
 {  4.80970912,  5.80436843,  6.76031   ,  7.44312562,  7.81468778,
    7.91065805,  7.8829675 ,  7.81922906,  7.7740974 ,  7.87470415},
 {  5.51776154,  6.89651517,  8.24529477,  9.14176934,  9.62258856,
    9.73887223,  9.70135805,  9.63565683,  9.60939348,  9.77648883},
 {  6.24557864,  8.02245607,  9.76340998, 11.05020688, 11.71890126,
   11.90140345, 11.85421953, 11.66673361, 11.68036352, 11.9595978 },
 {  6.89677993,  9.15852133, 11.52702134, 13.16020515, 14.0922186 ,
   14.34602312, 14.19747765, 14.02563262, 14.0031119 , 14.41821042},
 {  7.47380763, 10.35116871, 13.32941093, 15.58301891, 16.75795015,
   16.98714862, 16.83158616, 16.55713859, 16.53995129, 17.06290653},
 {  8.04259851, 11.50034853, 15.28470428, 18.03434338, 19.61806089,
   20.05041451, 19.70870582, 19.4273524 , 19.39914553, 20.14608909},
 {  8.42575318, 12.54336005, 17.12065422, 20.74998752, 22.82103944,
   23.25746295, 23.01893536, 22.48551862, 22.58498322, 23.35260732},
 {  8.80265361, 13.54083836, 19.01812155, 23.50122543, 26.14473571,
   26.90099736, 26.59836402, 25.98037948, 25.96265337, 27.09307293},
 {  8.98256123, 14.4452487 , 20.92482469, 26.38465622, 29.51981181,
   30.63035382, 30.52535144, 29.74796913, 29.67039574, 30.9455894 },
 {  9.09718134, 15.13577111, 22.65172658, 29.149789  , 33.24113699,
   34.70102953, 34.28177107, 33.6870475 , 33.69121918, 35.01225526},
 {  9.11895546, 15.59907265, 24.30949025, 31.993905  , 36.82967489,
   38.82733539, 38.77900657, 37.8116203 , 37.75033926, 39.40195187},
 {  9.00401261, 15.96568042, 25.65624604, 34.44531543, 40.57637135,
   43.0682052 , 43.06107088, 42.3928246 , 42.27420527, 43.95651171},
 {  8.863984  , 16.24295866, 26.79931792, 36.89377039, 44.10021485,
   47.49134857, 47.9283423 , 47.04874884, 46.75559324, 48.65271708},
 {  8.57455292, 16.22519075, 27.4241613 , 38.93818118, 47.37650061,
   51.75018735, 52.43402173, 51.84001757, 51.46705059, 53.67239682},
 {  8.24606664, 16.07173518, 28.10786874, 40.9131219 , 50.81609825,
   56.07744687, 57.37073556, 56.72449591, 56.48920568, 58.63695092},
 {  7.96202504, 15.76394099, 28.23024104, 42.33421441, 53.83853495,
   60.25328115, 62.22745928, 61.757067  , 61.45460156, 64.19215952},
 {  7.59948524, 15.27734258, 28.18065291, 43.43990964, 56.18729736,
   64.02649028, 66.78489188, 66.49801193, 66.58882035, 69.16036553},
 {  7.30523783, 14.82100535, 27.85604638, 43.97802841, 58.47924795,
   67.8963467 , 71.69300824, 71.80818528, 71.83979779, 74.55814349},
 {  6.9885602 , 14.24578681, 27.46639226, 44.21372446, 60.46361308,
   71.53847092, 76.70354813, 77.14224757, 77.36665534, 79.90607097},
 {  6.73760966, 13.63674569, 26.65024732, 44.12855255, 61.79036704,
   75.09143531, 81.29224123, 82.95391021, 83.0047745 , 85.98608825},
 {  6.51026616, 13.03930535, 25.6552647 , 43.51288346, 62.9146572 ,
   78.02524141, 86.2484906 , 88.41299082, 88.79389606, 91.20928858},
 {  6.33437994, 12.4295668 , 24.72735226, 42.75408991, 63.51641884,
   81.32001708, 91.6881147 , 94.69098435, 95.15745839, 97.07410404},
 {  6.25333966, 11.87111609, 23.43206155, 41.81974708, 64.14989325,
   84.31660257, 96.95738112,101.70874261,101.45824311,102.78932066},
 {  6.24225036, 11.39324642, 22.3188391 , 40.6372603 , 64.10298915,
   87.36858437,102.7039233 ,109.26807882,109.19419702,108.50741203},
 {  6.27950247, 10.89280914, 20.97967491, 39.09561011, 64.40875919,
   90.51633544,109.57442985,117.6026228 ,117.13951067,115.13208807},
 {  6.51323624, 10.52342966, 19.81286005, 37.63961154, 64.25644245,
   93.74040859,116.77789301,127.62496078,125.85259725,122.01400893},
 {  6.84718529, 10.23173577, 18.73188456, 36.04303434, 64.31434596,
   97.13610512,124.72811145,138.40674932,136.68118169,128.93354052},
 {  7.3175193 ,  9.99213293, 17.65486847, 34.7091077 , 63.86523767,
  100.80940947,133.48502461,151.83081163,149.38255538,136.96508307},
 {  8.01365225,  9.89733782, 16.6091873 , 33.34762195, 63.47882148,
  104.12490313,144.43640107,168.13467798,164.43349712,146.4083255 },
 {  9.00617125,  9.86731814, 15.80237125, 32.13029151, 63.40102447,
  108.11757916,155.41963967,186.67489883,181.86961045,156.96366516},
 {  2.19785762,  2.31089037,  3.61816851,  7.3561966 , 14.73279198,
   25.42202955, 37.87556403, 45.44645943, 44.0483584 , 37.62063544},
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        },
 {  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        }};
real hist_edge_0[51] = {3.16227766e+04,3.80189396e+04,4.57088190e+04,5.49540874e+04,
 6.60693448e+04,7.94328235e+04,9.54992586e+04,1.14815362e+05,
 1.38038426e+05,1.65958691e+05,1.99526231e+05,2.39883292e+05,
 2.88403150e+05,3.46736850e+05,4.16869383e+05,5.01187234e+05,
 6.02559586e+05,7.24435960e+05,8.70963590e+05,1.04712855e+06,
 1.25892541e+06,1.51356125e+06,1.81970086e+06,2.18776162e+06,
 2.63026799e+06,3.16227766e+06,3.80189396e+06,4.57088190e+06,
 5.49540874e+06,6.60693448e+06,7.94328235e+06,9.54992586e+06,
 1.14815362e+07,1.38038426e+07,1.65958691e+07,1.99526231e+07,
 2.39883292e+07,2.88403150e+07,3.46736850e+07,4.16869383e+07,
 5.01187234e+07,6.02559586e+07,7.24435960e+07,8.70963590e+07,
 1.04712855e+08,1.25892541e+08,1.51356125e+08,1.81970086e+08,
 2.18776162e+08,2.63026799e+08,3.16227766e+08};
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
real CascadesEnergyResolutionMuPolyCoeffs[3,4] = {{ 8.48311815e-02,-1.40745871e+00, 8.39735975e+00,-1.29122823e+01},
 { 2.21533174e-02,-3.66019618e-01, 2.96495761e+00,-3.60384901e+00},
 { 2.32768758e-03,-4.24291672e-02, 1.26012779e+00,-5.56780567e-01}};
real CascadesEnergyResolutionSdPolyCoeffs[3,4] = {{-4.14191927e-03, 7.53090015e-02,-4.31439496e-01, 8.45584783e-01},
 { 1.31648641e-03,-2.44148960e-02, 1.55383237e-01,-3.02974555e-01},
 {-4.15523836e-04, 7.44664372e-03,-4.42253583e-02, 9.70242677e-02}};
real mu_e_res[3];
real sigma_e_res[3];
vector[3] weights;
for (i in 1:3)
{
weights[i] = 1.0/3;
}
for (i in 1:3)
{
mu_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionMuPolyCoeffs[i]));
sigma_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionSdPolyCoeffs[i]));
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
vector[3] result;
for (i in 1:3)
{
result[i] = (log(weights)[i]+lognormal_lpdf(x | means[i], sigmas[i]));
}
return log_sum_exp(result);
}
real CascadeEnergyResolution(real true_energy,real reco_energy)
{
real CascadesEnergyResolutionMuPolyCoeffs[3,4] = {{ 8.48311815e-02,-1.40745871e+00, 8.39735975e+00,-1.29122823e+01},
 { 2.21533174e-02,-3.66019618e-01, 2.96495761e+00,-3.60384901e+00},
 { 2.32768758e-03,-4.24291672e-02, 1.26012779e+00,-5.56780567e-01}};
real CascadesEnergyResolutionSdPolyCoeffs[3,4] = {{-4.14191927e-03, 7.53090015e-02,-4.31439496e-01, 8.45584783e-01},
 { 1.31648641e-03,-2.44148960e-02, 1.55383237e-01,-3.02974555e-01},
 {-4.15523836e-04, 7.44664372e-03,-4.42253583e-02, 9.70242677e-02}};
real mu_e_res[3];
real sigma_e_res[3];
vector[3] weights;
for (i in 1:3)
{
weights[i] = 1.0/3;
}
for (i in 1:3)
{
mu_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionMuPolyCoeffs[i]));
sigma_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionSdPolyCoeffs[i]));
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
real F_atmo;
int Ngrid;
vector[Ngrid] alpha_grid;
vector[Ngrid] integral_grid[Ns+1];
real atmo_integ_val;
real aeff_max;
real v_lim;
real T;
int N_atmo;
unit_vector[3] atmo_directions[N_atmo];
vector[N_atmo] atmo_energies;
simplex[N_atmo] atmo_weights;
}
transformed data
{
vector[Ns+2] F;
simplex[Ns+2] w_exposure;
vector[Ns+2] eps;
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
F[Ns+2] = F_atmo;
Ftot = ((Fs+F_diff)+F_atmo);
f = Fs/Ftot;
print("f: ", f);
eps = get_exposure_factor_atmo(alpha, alpha_grid, integral_grid, atmo_integ_val, T, Ns);
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
int atmo_index;
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
else if(Lambda[i] == (Ns+2))
{
atmo_index = categorical_rng(atmo_weights);
omega = atmo_directions[atmo_index];
}
cosz[i] = cos(omega_to_zenith(omega));
if(Lambda[i] <= (Ns+1))
{
Esrc[i] = spectrum_rng(alpha, Esrc_min, Esrc_max);
E[i] = (Esrc[i]/(1+z[Lambda[i]]));
}
else if(Lambda[i] == (Ns+2))
{
E[i] = atmo_energies[atmo_index];
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
