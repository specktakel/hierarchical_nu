/**
* Power law pdf
* @date April 2023
*/

real powerlaw_logpdf(real E, real alpha, real e_low, real e_up)
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

real dbbpl_logpdf(real x, real gamma2, real x1, real x2, real x3, real x4)
{
//real x1 = 1.;   // bounds at which integration is cutoff in the flanks
//real x4 = 100.;
real gamma1 = -10.;   // indices at the flanks
real gamma3 = 10.;
real N2;
real N3;
real I1;
real I2;
real I3;
real I;
real p;
I1 = (pow(x2, -gamma1 + 1.) - pow(x1, -gamma1 + 1.)) / (1. - gamma1);
N2 = pow(x2, gamma2 - gamma1);
//I2 = (pow(x3, -gamma2) - pow(x2, -gamma2)) * N2 / (1. + gamma2);
if(gamma2 == 1.0)
{
I2 = log(x3 / x2) * N2;
}
else
{
I2 = (pow(x3, -gamma2 + 1.) - pow(x2, -gamma2 + 1.)) / (1. - gamma2) * N2;
}
N3 = pow(x3, gamma3 - gamma2) * N2;
I3 = (pow(x4, -gamma3 + 1.) - pow(x3, -gamma3 + 1.)) * N3 / (1. - gamma3);
I = I1 + I2 + I3;
if (x<x2) {
p = pow(x, -gamma1);
}
else if (x<x3) 
{
p = pow(x, -gamma2) * N2;
}
else
{
p = pow(x, -gamma3) * N3;
}
return log(p / I);
}
