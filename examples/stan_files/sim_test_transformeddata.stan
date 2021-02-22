vector[Ns+1] F;
real FT;
real Fs;
real f;
simplex[Ns+1] w_exposure;
real Nex;
int N;
vector eps[Ns+1];
real Mpc_to_m;

Mpc_to_m = 3.086e22;
F[Ns+1] = F0;
FT = (Fs+F0);
f = Fs/FT;
eps = get_exposure_factor(T, Emin, alpha,alpha_grid, integral_grid, Ns);
Nex = get_Nex_sim(F, eps);
w_exposure = get_exposure_weights(F, eps);
N = poisson_rng(Nex);
for (k in 1:Ns)
{

F[k] = 4 * pi() * pow(D[k] * Mpc_to_m, 2);
Fs += F[k];
}
