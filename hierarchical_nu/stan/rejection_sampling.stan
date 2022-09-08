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
* Returns total and elative weights of each section.
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