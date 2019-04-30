# Search for contained neutrino events at energies greater than 1 TeV in 2 years of data

## Introduction

The IceCube Neutrino Observatory was designed primarily to search for
high-energy (TeV--PeV) neutrinos produced in distant astrophysical objects. A
search for $\gtrsim 100$~TeV neutrinos interacting inside the instrumented volume
has recently provided evidence for an isotropic flux of such neutrinos. At lower
energies, IceCube collects large numbers of neutrinos from the weak
decays of mesons in cosmic-ray air showers. Here we present the results of a
search for neutrino interactions inside IceCube's instrumented volume between
1~TeV and 1~PeV in 641 days of data taken from 2010--2012, lowering the energy
threshold for neutrinos from the southern sky below 10 TeV for the first time,
far below the threshold of the previous high-energy analysis. Astrophysical
neutrinos remain the dominant component in the southern sky down to a deposited energy of 10 TeV.
From these data we derive new constraints on the diffuse astrophysical neutrino
spectrum, $\Phi_{\nu} = 2.06^{+0.4}_{-0.3} \times 10^{-18}
\left({E_{\nu}}/{10^5 \,\, \rm{GeV}} \right)^{-2.46 \pm 0.12} {\rm {GeV^{-1} \,
cm^{-2} \, sr^{-1} \, s^{-1}} } $ for $25 \,\, \text{TeV} < E_{\nu} < 1.4 \,\, \text{PeV}$,
as well as the strongest upper limit yet on
the flux of neutrinos from charmed-meson decay in the atmosphere, 1.52 times
the benchmark theoretical prediction used in previous IceCube results at 90\%
confidence.

Atmospheric and Astrophysical Neutrinos above 1 TeV Interacting in IceCube, IceCube Collaboration, Phys. Rev. D, 91(2):022001 (2015). doi: 10.1103/PhysRevD.91.022001.

## Data

The download includes the following files:

effective_area.nu_e.txt
effective_area.nu_mu.txt
effective_area.nu_tau.txt
-------------------------
Neutrino effective areas in m^2 for the entire event selection. To predict the
total number of events that would have appeared in this sample from your
favorite flux, integrate your flux model over the provided neutrino energy and
zenith angle bins to obtain a rate of events per m^2, multiply by the provided
effective areas, and take the sum over all neutrino energy and zenith angle
bins.

Note that the quoted effective area is an average for neutrinos and
antineutrinos. To obtain the correct number of events, multiply by the sum of
the neutrino and antineutrino fluxes.

effective_area.per_bin.nu_e.cc.cascade.txt.gz
effective_area.per_bin.nu_e.nc.cascade.txt.gz
effective_area.per_bin.nu_e_bar.cc.cascade.txt.gz
effective_area.per_bin.nu_e_bar.gr.cascade.txt.gz
effective_area.per_bin.nu_e_bar.gr.track.txt.gz
effective_area.per_bin.nu_e_bar.nc.cascade.txt.gz
effective_area.per_bin.nu_mu.cc.cascade.txt.gz
effective_area.per_bin.nu_mu.cc.track.txt.gz
effective_area.per_bin.nu_mu.nc.cascade.txt.gz
effective_area.per_bin.nu_mu_bar.cc.cascade.txt.gz
effective_area.per_bin.nu_mu_bar.cc.track.txt.gz
effective_area.per_bin.nu_mu_bar.nc.cascade.txt.gz
effective_area.per_bin.nu_tau.cc.cascade.txt.gz
effective_area.per_bin.nu_tau.cc.track.txt.gz
effective_area.per_bin.nu_tau.nc.cascade.txt.gz
effective_area.per_bin.nu_tau_bar.cc.cascade.txt.gz
effective_area.per_bin.nu_tau_bar.cc.track.txt.gz
effective_area.per_bin.nu_tau_bar.nc.cascade.txt.gz
---------------------------------------------------
Effective areas in m^2 for each bin shown in Fig. 8, separated by particle
type, interaction channel (charged-current deep-inelastic scattering (CC),
neutral current DIS (NC), or resonant anti-electron-neutrino/electron
scattering (GR)), and event signature (track or cascade, using the same
classification as the event table shown below). Using these tables instead of
the total effective areas given above allows you to predict the
deposited-energy spectrum observed in this analysis separately for the northern
and southern hemispheres.

Note that in contrast to the integrated tables above, these effective areas are
given separately for neutrinos and anti-neutrinos, and so should be multiplied
by a neutrino or anti-neutrino flux.

event_properties.txt
--------------------
Reconstructed deposited energies and declination for the 383 events with
successful energy reconstructions. Each of the error ranges given is 68%
confidence interval derived from Monte Carlo simulation, assuming the given
flux (conventional atmospheric neutrinos, the best-fit 1:1:1 E^-2.47
astrophysical neutrino flux, or the typical 1:1:1 E^-2 benchmark flux).

unfolding.txt
-------------
Data points from Fig. 12

energy_north_south.txt
----------------------
Models and data points from Fig. 8

NB: the energy is resonstructed deposited energy, not neutrino energy