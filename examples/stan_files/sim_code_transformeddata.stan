vector[Ns+1] F;
real FT;
real Fs;
real f;
simplex[Ns+1] w_exposure;
real Nex;
int N;
vector[Ns+1] eps;

for (k in 1:Ns)
{
F[k] = Q/ (4 * pi() * pow(D[k] * 3.086e+22, 2));
Fs += F[k];
}
F[Ns+1] = F0;
FT = Fs+FT;
f = Fs/FT;
eps = get_exposure_factor(T, Emin, alpha, alpha_grid, integral_grid, Ns);
Nex = get_Nex_sim(F, eps);
w_exposure = get_exposure_weights(F, eps);
N = poisson_rng(Nex);
print(w_exposure);
print(Ngrid);
print(Nex);
print(N);
