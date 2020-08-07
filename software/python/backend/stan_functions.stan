real[] generate_bin_edges(real lower, real upper, int nbins)
    {

        real binedges[nbins+1];
        real binwidth = (upper-lower)/nbins;
        for (i in 1:nbins+1)
        {
            binedges[i] = lower + (i-1)*binwidth;
        }
        return binedges;
    }

    int binary_search(real value, real[] binedges)
    {
        int L = 1;
        int R = size(binedges);
        int m;
        if (value < binedges[1])
            return 0;
        else if(value > binedges[R])
            return R+1;
        else{
            while (L < R-1)
            {
                m = (L + R) / 2;
                if (binedges[m] < value)
                    L = m;
                else if (binedges[m] > value)
                    R = m;
                else
                    return m;
            }
        }
        return L;
    }