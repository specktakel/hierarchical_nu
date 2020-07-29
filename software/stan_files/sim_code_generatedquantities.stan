int Lambda[N];
unit_vector[3] omega;
vector[N] Esrc;
vector[N] E;
vector[N] Edet;
real cosz[N];
real Pdet[N];
int accept;
int ntrials;
simplex[2] prob;
unit_vector[3] event[N];
real Nex_sim;

Nex_sim = Nex;
for (i in 1:N)
{
Lambda[i] = categorical_rng(w_exposure);
accept = 0;
ntrials = 0;
while(accept != 1) {
if (Lambda[i] < Ns+1) {
omega = varpi[Lambda[i]];
}
else if (Lambda[i] == Ns+1) {
omega = sphere_rng(1);
};
cosz[i] = cos(omega_to_zenith(omega));
Esrc[i] = spectrum_rng(alpha, Emin * (1+z[Lambda[i]]) );
E[i] = Esrc[i]/ (1+z[Lambda[i]]);
if (cosz[i] >= 0.1) {
Pdet[i] = 0;
}
else {
Pdet[i] = NorthernTracksEffectiveArea(E[i], omega) / aeff_max;
};
prob[1] = Pdet[i];
prob[2] = 1-Pdet[i];
ntrials += 1;
if (ntrials < 10000) {
accept = categorical_rng(prob);
}
else {
accept = 1;
print("problem component: ", Lambda[i]);
};
};
event[i] = NorthernTracksAngularResolution_rng(E[i], omega);
Edet[i] = NorthernTracksEnergyResolution_rng(E[i]);
}
