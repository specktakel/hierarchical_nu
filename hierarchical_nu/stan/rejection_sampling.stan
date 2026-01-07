/**
* Functions for efficient rejection sampling
* of the effective area. 
*
* @author Francesca Capel
* @date August 2022
*/

/**
* Integrate a broken bounded power law.
*
* Returns total and relative weights of each section.
*/
vector bbpl_int(real x0, real x1, real x2, real gamma1, real gamma2) {

    vector[3] output;
    real int_first_seg;
    real int_second_seg;

    int_first_seg = (pow(x1, gamma1 + 1.0) 
        - pow(x0, gamma1 + 1.0)) / (gamma1 + 1.0);    
    int_second_seg = (pow(x1, gamma1 - gamma2) 
        * (pow(x2, gamma2 + 1.0) - pow(x1, gamma2 + 1.0))
        / (gamma2 + 1.0));

    output[3] = int_first_seg + int_second_seg;

    output[1] = int_first_seg / output[3];
    output[2] = int_second_seg / output[3];

    return output;
}

/**
* Sample from a bounded broken power law.
*
* Based on:
* https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb
* by J. M. Burgess (@grburgess).
*/
real bbpl_rng(real x0, real x1, real x2, real gamma1, real gamma2) {

    vector[3] output;
    real u;
    int seg;
    real sample;

    output = bbpl_int(x0, x1, x2, gamma1, gamma2);

    u = uniform_rng(0.0, 1.0);

    seg = bernoulli_rng(output[1]);

    if (seg == 1) {

        sample = pow(u * (pow(x1, gamma1 + 1.0) - pow(x0, gamma1 + 1.0))
            + pow(x0, gamma1 + 1.0),
            1.0 / (gamma1 + 1.0));

    }
    else if (seg == 0) {

        sample = pow(u * (pow(x2, gamma2 + 1.0)
                - pow(x1, gamma2 + 1.0))
            + pow(x1, gamma2 + 1.0),
            1.0 / (gamma2 + 1.0));

    }

    return sample;
}

/**
* PDF of a bounded broken power law.
*/
real bbpl_pdf(real x, real x0, real x1, real x2, real gamma1, real gamma2) {

    real output;
    real I1;
    real I2;
    real N;

    I1 = (pow(x1, gamma1 + 1.0) - pow(x0, gamma1 + 1.0)) / (gamma1 + 1.0);
    I2 = (
        pow(x1, gamma1 - gamma2)
        * (pow(x2, gamma2 + 1.0) - pow(x1, gamma2 + 1.0))
        / (gamma2 + 1.0)
    );

    N = 1.0 / (I1 + I2);

    if ((x <= x1) && (x >= x0)) {

        output = N * pow(x, gamma1);
    }
    else if ((x > x1) && (x <= x2)) {

        output = N * pow(x1, gamma1 - gamma2) * pow(x, gamma2);

    } 
    else {

        output = 0.0;

    }

    return output;
}

/**
* PDF of a bounded broken power law.
*/
real multiple_bbpl_pdf(real x, array[] real breaks, array[] real slopes, array[] real low_vals) {

    int N = size(breaks);
    int idx;
    real output;

    // check for x being outside of the breaks (including lower and upper limit of domain)
    if ((x < breaks[1]) || (x > breaks[N])) {
        return 0.0;
    }

    // find bin of x and return power law using the provided indices and normalisations
    idx = binary_search(x, breaks);
    output = low_vals[idx] * pow(x / breaks[idx], slopes[idx]);
    return output;
}

real multiple_bbpl_rng(array[] real breaks, array[] real slopes, vector weights) {
    int idx;
    real x1;
    real x0;
    real gamma;
    real u;
    real sample;
    real gammap1;

    idx = categorical_rng(weights);
    gamma = slopes[idx];
    gammap1 = gamma + 1.0;
    x0 = breaks[idx];
    x1 = breaks[idx+1];

    u = uniform_rng(0.0, 1.0);

    if (gamma == -1.) {
        sample = x0 * pow(x1 / x0, u);
    }
    else {
    sample = pow(u * (pow(x1, gammap1) - pow(x0, gammap1))
            + pow(x0, gammap1),
            1.0 / gammap1);
    }
    return sample;
}
