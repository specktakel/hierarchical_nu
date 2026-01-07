### Elaborate tests

The unit tests in the test directory should cover the unit test aspect. The tests collected here are supposed to check if things are working on a larger scale, using more CPUs for larger fits.

This includes, but is not limited to, fits to experimental data (e.g. TXS archival flare of 2014/15), fits to various simulated setups (e.g. single source and single season, single source over 10 years, adding background components...).
What this is not intended to cover is performance checks in the sense of "is my fit accurate enough for finding X source in the data". We are only looking for obvious flaws in the code, for example the sampler hitting max treedepth.