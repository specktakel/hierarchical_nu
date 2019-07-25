//compile with: $CC effective_area_demo.c -lm -lhdf5 -o effective_area_demo

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <hdf5.h>

//An isotropic power law flux in units of per GeV cm^2 sr s
typedef struct{
	double normalization;
	double index;
} powerlawFlux;

double evaluatePowerlawFlux(const powerlawFlux flux, double neutrinoEnergy){
	//Note that we include here the correction due to (mis-simulating) the detector
	//optical efficiency. It isn't really part of the flux, but it depends on the
	//spectrum (in this case the spectral index), so this is convenient.
	double DOMEffCorr=pow(1.189944,-2.-flux.index);
	return(flux.normalization*DOMEffCorr
	       *pow(neutrinoEnergy/1.e5,flux.index));
}

double integratePowerlawFlux(const powerlawFlux flux, double minEnergy, double maxEnergy){
	assert(flux.index!=-1.0 && "Special case of E^{-1} not handled");
	double DOMEffCorr=pow(1.189944,-2.-flux.index);
	double intIndex=1.+flux.index;
	double norm=(flux.normalization*DOMEffCorr)/(intIndex*pow(1.e5,flux.index));
	return(norm*(pow(maxEnergy,intIndex)-pow(minEnergy,intIndex)));
}

//A quick and dirty multidimensional array
typedef struct{
	unsigned int rank;
	unsigned int* strides;
	double* data;
} multidim;

multidim alloc_multi(unsigned int rank, const unsigned int* strides){
	multidim m;
	m.rank=rank;
	m.strides=(unsigned int*)malloc(rank*sizeof(unsigned int));
	if(!m.strides){
		fprintf(stderr,"Failed to allocate memory");
		exit(1);
	}
	unsigned int size=1;
	for(unsigned int i=0; i<rank; i++){
		m.strides[i]=strides[i];
		size*=strides[i];
	}
	m.data=(double*)malloc(size*sizeof(double));
	if(!m.strides){
		fprintf(stderr,"Failed to allocate memory");
		exit(1);
	}
	return(m);
}

void free_multi(multidim m){
	free(m.strides);
	free(m.data);
}

double* index_multi(multidim m, unsigned int* indices){
	unsigned int idx=0;
	for(unsigned int i=0; i<m.rank; i++){
		idx*=m.strides[i];
		idx+=indices[i];
	}
	return(m.data+idx);
}

//Read a dataset into a buffer, asusming that the allocated size is correct.
//If anything goes wrong just bail out.
void readDataSet(hid_t container_id, const char* path, double* targetBuffer){
	hid_t dataset_id = H5Dopen(container_id, path, H5P_DEFAULT);
	herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, targetBuffer);
	if(status<0){
		fprintf(stderr,"Failed to read dataset '%s'\n",path);
		exit(1);
	}
	H5Dclose(dataset_id);
}

void readDoubleAttr(hid_t container_id, const char* path, const char* name, double* targetBuffer){
	hid_t attr_id = H5Aopen_by_name(container_id, path, name, H5P_DEFAULT, H5P_DEFAULT);
	herr_t status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, targetBuffer);
	if(status<0){
		fprintf(stderr,"Failed to read attribute '%s::%s'\n",path,name);
		exit(1);
	}
	H5Aclose(attr_id);
}

int main(void){
	
	const unsigned int neutrinoEnergyBins=280;
	const unsigned int cosZenithBins=11;
	const unsigned int energyProxyBins=50;
	//the edges used for the bins of the effective area histograms
	//these are the same for all effective areas
	double* trueEnergyEdges=(double*)malloc((neutrinoEnergyBins+1)*sizeof(double));
	double* cosZenithEdges=(double*)malloc((cosZenithBins+1)*sizeof(double));
	double* energyProxyEdges=(double*)malloc((energyProxyBins+1)*sizeof(double));
	
	const unsigned int histogramDims[3]={neutrinoEnergyBins,cosZenithBins,energyProxyBins};
	
	multidim effArea2010NuMu=alloc_multi(3,histogramDims);
	multidim effArea2010NuMuBar=alloc_multi(3,histogramDims);
	multidim effArea2010NuTau=alloc_multi(3,histogramDims);
	multidim effArea2010NuTauBar=alloc_multi(3,histogramDims);
	multidim effArea2011NuMu=alloc_multi(3,histogramDims);
	multidim effArea2011NuMuBar=alloc_multi(3,histogramDims);
	multidim effArea2011NuTau=alloc_multi(3,histogramDims);
	multidim effArea2011NuTauBar=alloc_multi(3,histogramDims);
	
	multidim effArea2010NuMu_Err=alloc_multi(3,histogramDims);
	multidim effArea2010NuMuBar_Err=alloc_multi(3,histogramDims);
	multidim effArea2010NuTau_Err=alloc_multi(3,histogramDims);
	multidim effArea2010NuTauBar_Err=alloc_multi(3,histogramDims);
	multidim effArea2011NuMu_Err=alloc_multi(3,histogramDims);
	multidim effArea2011NuMuBar_Err=alloc_multi(3,histogramDims);
	multidim effArea2011NuTau_Err=alloc_multi(3,histogramDims);
	multidim effArea2011NuTauBar_Err=alloc_multi(3,histogramDims);
	
	double livetime2010;
	double livetime2011;
	
	//make some arrays of aliases to make it easy to loop over these things
	multidim* effectiveAreas[2][4]={{&effArea2010NuMu,&effArea2010NuMuBar,
	                                 &effArea2010NuTau,&effArea2010NuTauBar},
	                                {&effArea2011NuMu,&effArea2011NuMuBar,
	                                 &effArea2011NuTau,&effArea2011NuTauBar}};
	multidim* effectiveAreaErrs[2][4]={{&effArea2010NuMu_Err,&effArea2010NuMuBar_Err,
	                                    &effArea2010NuTau_Err,&effArea2010NuTauBar_Err},
	                                   {&effArea2011NuMu_Err,&effArea2011NuMuBar_Err,
	                                    &effArea2011NuTau_Err,&effArea2011NuTauBar_Err}};
	double* livetimes[2]={&livetime2010,&livetime2011};
	
	//the expected distribution of astrophysical events
	//in the energy proxy observable
	double* expectation=(double*)malloc(energyProxyBins*sizeof(double));
	double* expectationUncertainty=(double*)malloc(energyProxyBins*sizeof(double));
	
	herr_t status=0;
	hid_t file_id = H5Fopen("effective_area.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
	
	readDoubleAttr(file_id,"/2010","experimental_livetime(seconds)",&livetime2010);
	readDoubleAttr(file_id,"/2011","experimental_livetime(seconds)",&livetime2011);
	
	readDataSet(file_id, "/2010/nu_mu/bin_edges_0", trueEnergyEdges);
	readDataSet(file_id, "/2010/nu_mu/bin_edges_1", cosZenithEdges);
	readDataSet(file_id, "/2010/nu_mu/bin_edges_2", energyProxyEdges);
	
	readDataSet(file_id, "/2010/nu_mu/area", effArea2010NuMu.data);
	readDataSet(file_id, "/2010/nu_mu_bar/area", effArea2010NuMuBar.data);
	readDataSet(file_id, "/2010/nu_tau/area", effArea2010NuTau.data);
	readDataSet(file_id, "/2010/nu_tau_bar/area", effArea2010NuTauBar.data);
	readDataSet(file_id, "/2011/nu_mu/area", effArea2011NuMu.data);
	readDataSet(file_id, "/2011/nu_mu_bar/area", effArea2011NuMuBar.data);
	readDataSet(file_id, "/2011/nu_tau/area", effArea2011NuTau.data);
	readDataSet(file_id, "/2011/nu_tau_bar/area", effArea2011NuTauBar.data);
	
	readDataSet(file_id, "/2010/nu_mu/area_uncertainty", effArea2010NuMu_Err.data);
	readDataSet(file_id, "/2010/nu_mu_bar/area_uncertainty", effArea2010NuMuBar_Err.data);
	readDataSet(file_id, "/2010/nu_tau/area_uncertainty", effArea2010NuTau_Err.data);
	readDataSet(file_id, "/2010/nu_tau_bar/area_uncertainty", effArea2010NuTauBar_Err.data);
	readDataSet(file_id, "/2011/nu_mu/area_uncertainty", effArea2011NuMu_Err.data);
	readDataSet(file_id, "/2011/nu_mu_bar/area_uncertainty", effArea2011NuMuBar_Err.data);
	readDataSet(file_id, "/2011/nu_tau/area_uncertainty", effArea2011NuTau_Err.data);
	readDataSet(file_id, "/2011/nu_tau_bar/area_uncertainty", effArea2011NuTauBar_Err.data);
	
	H5Fclose(file_id);
	
	printf("2010 livetime: %lf s\n",livetime2010);
	printf("2011 livetime: %lf s\n",livetime2011);
	
	//a test flux to evaluate
	powerlawFlux flux;
	//divide by two for per particle type flux, rather than per flavor
	flux.normalization=1.63e-18/2;
	flux.index=-2.22;
	
	//To compute the expected number of observed events we need to:
	//1a. Multiply each effective area by the average flux in each bin
	//1b. Multiply by the phase space in each bin (true energy and solid angle)
	//1c. Multiply by the livetime for which the effective area is relevant
	//2. Sum over the effective areas for all particle types and detector configurations
	//3. Sum over dimensions not of interest (true neutrino energy, possibly zenith angle)
	//In this case we will compute the expectation as a function of the energy
	//proxy only, so we will project out both true energy and zenith angle.
	
	memset(expectation, 0, energyProxyBins*sizeof(double));
	memset(expectationUncertainty, 0, energyProxyBins*sizeof(double));
	unsigned int indices[3];
	
	for(unsigned int i=0; i<energyProxyBins; i++){
		indices[2]=i;
		//for both years
		for(unsigned int y=0; y<2; y++){
			//for each particle type
			for(unsigned int p=0; p<4; p++){
				//for each true energy bin
				for(unsigned int e=0; e<neutrinoEnergyBins; e++){
					indices[0]=e;
					
					double enMin=trueEnergyEdges[e];
					double enMax=trueEnergyEdges[e+1];
					
					//for each cosine zenith angle bin
					for(unsigned int z=0; z<cosZenithBins; z++){
						indices[1]=z;
						
						double cosZenithMin=cosZenithEdges[z];
						double cosZenithMax=cosZenithEdges[z+1];
						
						//the product of the average flux in the bin and the
						//phase space in the bin is simply the integral of the
						//flux over the bin
						double fluxIntegral=
						  integratePowerlawFlux(flux,enMin,enMax) //energy intergal
						  *(cosZenithMax-cosZenithMin) //zenith integral
						  *2*M_PI; //azimuth integral
						
						double effectiveArea=*index_multi(*effectiveAreas[y][p],indices);
						effectiveArea*=1.0e4; //convert m^2 to cm^2
						expectation[i] += effectiveArea * fluxIntegral * *livetimes[y];
						
						//We can also compute the uncertainty due to limited simulation statistics.
						//This requires adding the error terms in quadrature.
						double effectiveAreaErr=*index_multi(*effectiveAreaErrs[y][p],indices);
						effectiveAreaErr*=1.0e4; //convert m^2 to cm^2
						effectiveAreaErr*=fluxIntegral * *livetimes[y];
						effectiveAreaErr*=effectiveAreaErr; //square!
						expectationUncertainty[i] += effectiveAreaErr;
					}
				}
			}
		}
	}
	
	//Print out the expected number of events in each energy proxy bin, with
	//statistical uncertainties.
	printf("Astrophysical flux expectation as a function of energy proxy\n");
	double total=0;
	for(unsigned int i=0; i<energyProxyBins; i++){
		printf(" %lf: %lf [%lf,%lf]\n",energyProxyEdges[i],expectation[i],
		       expectation[i]-sqrt(expectationUncertainty[i]),
		       expectation[i]+sqrt(expectationUncertainty[i]));
		total+=expectation[i];
	}
	printf("Total: %lf\n",total);
	
	//--------------------------------------------------------------------------
	
	//these share the same binning in the first two dimensions
	multidim convAtmosNuMu=alloc_multi(2,histogramDims);
	multidim convAtmosNuMuBar=alloc_multi(2,histogramDims);
	
	multidim convDOMEffCorrection2010=alloc_multi(3,histogramDims);
	multidim convDOMEffCorrection2011=alloc_multi(3,histogramDims);
	
	multidim* convAtmosFlux[2]={&convAtmosNuMu,&convAtmosNuMuBar};
	multidim* convDOMEffCorrection[2]={&convDOMEffCorrection2010,&convDOMEffCorrection2011};
	
	file_id = H5Fopen("conventional_flux.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
	
	readDataSet(file_id, "/nu_mu/integrated_flux", convAtmosNuMu.data);
	readDataSet(file_id, "/nu_mu_bar/integrated_flux", convAtmosNuMuBar.data);
	readDataSet(file_id, "/detector_correction/2010", convDOMEffCorrection2010.data);
	readDataSet(file_id, "/detector_correction/2011", convDOMEffCorrection2011.data);
	
	H5Fclose(file_id);
	
	memset(expectation, 0, energyProxyBins*sizeof(double));
	memset(expectationUncertainty, 0, energyProxyBins*sizeof(double));
	
	for(unsigned int i=0; i<energyProxyBins; i++){
		indices[2]=i;
		//for both years
		for(unsigned int y=0; y<2; y++){
			//Note that here we include only nu_mu and nu_mu_bar,
			//tau contributions are assumed to be zero
			for(unsigned int p=0; p<2; p++){
				//for each true energy bin
				for(unsigned int e=0; e<neutrinoEnergyBins; e++){
					indices[0]=e;
					
					double enMin=trueEnergyEdges[e];
					double enMax=trueEnergyEdges[e+1];
					
					//for each cosine zenith angle bin
					for(unsigned int z=0; z<cosZenithBins; z++){
						indices[1]=z;
						
						double cosZenithMin=cosZenithEdges[z];
						double cosZenithMax=cosZenithEdges[z+1];
						
						//TODO: comment on flux integral
						double fluxIntegral=*index_multi(*convAtmosFlux[p],indices);
						
						double effectiveArea=*index_multi(*effectiveAreas[y][p],indices);
						effectiveArea*=1.0e4; //convert m^2 to cm^2
						effectiveArea*=*index_multi(*convDOMEffCorrection[y],indices);
						expectation[i] += effectiveArea * fluxIntegral * *livetimes[y];
						
						//We can also compute the uncertainty due to limited simulation statistics.
						//This requires adding the error terms in quadrature.
						double effectiveAreaErr=*index_multi(*effectiveAreaErrs[y][p],indices);
						effectiveAreaErr*=1.0e4; //convert m^2 to cm^2
						effectiveAreaErr*=*index_multi(*convDOMEffCorrection[y],indices);
						effectiveAreaErr*=fluxIntegral * *livetimes[y];
						effectiveAreaErr*=effectiveAreaErr; //square!
						expectationUncertainty[i] += effectiveAreaErr;
					}
				}
			}
		}
	}
	
	printf("\n");
	
	//Print out the expected number of events in each energy proxy bin, with
	//statistical uncertainties.
	printf("Conventional atmospheric flux expectation as a function of energy proxy\n");
	total=0;
	for(unsigned int i=0; i<energyProxyBins; i++){
		printf(" %lf: %lf [%lf,%lf]\n",energyProxyEdges[i],expectation[i],
		       expectation[i]-sqrt(expectationUncertainty[i]),
		       expectation[i]+sqrt(expectationUncertainty[i]));
		total+=expectation[i];
	}
	printf("Total: %lf\n",total);
	
	free(trueEnergyEdges);
	free(cosZenithEdges);
	free(energyProxyEdges);
	free_multi(effArea2010NuMu);
	free_multi(effArea2010NuMuBar);
	free_multi(effArea2010NuTau);
	free_multi(effArea2010NuTauBar);
	free_multi(effArea2011NuMu);
	free_multi(effArea2011NuMuBar);
	free_multi(effArea2011NuTau);
	free_multi(effArea2011NuTauBar);
	free_multi(effArea2010NuMu_Err);
	free_multi(effArea2010NuMuBar_Err);
	free_multi(effArea2010NuTau_Err);
	free_multi(effArea2010NuTauBar_Err);
	free_multi(effArea2011NuMu_Err);
	free_multi(effArea2011NuMuBar_Err);
	free_multi(effArea2011NuTau_Err);
	free_multi(effArea2011NuTauBar_Err);
	free_multi(convDOMEffCorrection2010);
	free_multi(convDOMEffCorrection2011);
	free(expectation);
	free(expectationUncertainty);
	return(0);
}