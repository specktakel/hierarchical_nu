import numpy as np
def get_poisson_confidence_intervals(counts):
        feldmann_cousins = { 0:(0.00, 1.29),
                             1:(0.37, 2.75),
                             2:(0.74, 4.25),
                             3:(1.10, 5.30),
                             4:(2.34, 6.78),
                             5:(2.75, 7.81),
                             6:(3.82, 9.28),
                             7:(4.25, 10.30),
                             8:(5.30, 11.32),
                             9:(6.33, 12.79),
                             10:(6.78, 13.81),
                             11:(7.81, 14.82),
                             12:(8.83, 16.29),
                             13:(9.28, 17.30),
                             14:(10.30, 18.32),
                             15:(11.32, 19.32),
                             16:(12.33, 20.80),
                             17:(12.79, 21.81),
                             18:(13.81, 22.82),
                             19:(14.82, 23.82),
                             20:(15.83, 25.30) }

        data_vals = counts
        data_errs=[]
        for vals in data_vals:
            val = vals
            if val == 0:
                data_errs.append( (0.0, feldmann_cousins[int(val)][1]) )
            elif (val > 0 and val < 21):
                #print feldmann_cousins[int(val)]
                data_errs.append( feldmann_cousins[int(val)] )
            else:
                data_errs.append( (val - np.sqrt(val), val + np.sqrt(val) ) )

        data_elow, data_eup = zip(*data_errs)
        data_elow = np.asarray(data_elow)
        data_eup = np.asarray(data_eup)

        data_elow = data_vals- data_elow
        data_eup = data_eup - data_vals

        return (data_elow, data_eup)

