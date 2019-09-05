#!/usr/bin/env python

import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import sklearn.gaussian_process as gp
from scipy.optimize import curve_fit

def GPfit(lcData, kernel, fit=True):
    time = lcData[:,0].reshape(-1,1)
    mag = lcData[:,1]
    sigmasq = lcData[:,2]*lcData[:,2]
    
    model = gp.GaussianProcessRegressor(kernel=kernel, alpha=sigmasq, n_restarts_optimizer=10)
    if fit:
        model.fit(time, mag)
    return model

def GPsamples(tsample, model, nsample):
    return np.squeeze(model.sample_y(tsample.reshape(-1,1), nsample))

def PlotSamples(tsample, samples):
    nsample = samples.shape[1]
    for n in range(nsample):
        plt.plot(tsample, samples[:,n], '.')

    plt.show()

def modelSigma(tsample, model):
    magPred, sigma = model.predict(tsample.reshape(-1,1), return_std=True)

    return sigma

def PlotPredict(tsample, model):
    magPred, sigma = model.predict(tsample.reshape(-1,1), return_std=True)

    plt.fill_between(tsample, magPred - 2.0*sigma, magPred + 2.0*sigma, alpha=0.2, color='k')

    plt.show()

def mean_function_parallax(t,F_S,F_B,t0,u0,tE,phi_pi,pi_E):
    global tc, tp, beta, Omega_0, eps, a

    """PSPL model"""
    #Define some constants
    tau = (t-t0)/tE 
    t_tp = t-tp
    t_tc = t-tc

    sin_phi_pi = np.sin(phi_pi)
    cos_phi_pi = np.cos(phi_pi)
    
    u = np.sqrt((tau*sin_phi_pi+u0*cos_phi_pi+a*pi_E*(1-eps*np.cos(Omega_0*t_tp))*np.sin(Omega_0*t_tp))**2+\
               (tau*cos_phi_pi-u0*sin_phi_pi+a*np.sin(beta)*pi_E*(1-eps*np.cos(Omega_0*t_tc))*np.cos(Omega_0*t_tc+2*eps*np.sin(Omega_0*t_tp)))**2)
   # A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))
    return F_S*(u**2 + 2)/(u*np.sqrt(u**2 + 4))+F_B

def mean_function(t,u0,t0,tE,DeltaF,Fbase):
    """PSPL model"""
    u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
    A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))
    return DeltaF*A(u) + Fbase

def magnitudes_to_fluxes(m, sig_m, zero_point=22.):
    """
    Given the mean and the standard deviation of a magnitude, 
    assumed to be normally distributed, and a reference magnitude m0, this 
    function returns the mean and the standard deviation of a Flux, 
    which is log-normally distributed.
    """

    # Calculate the mean and std. deviation for lnF which is assumed to be 
    # normally distributed 
    e = np.exp(1)
    mu_lnF = (zero_point - m)/(2.5*np.log10(e)) 
    sig_lnF = sig_m/(2.5*np.log10(e))

    # If lnF is normally distributed, F is log-normaly distributed with a mean
    # and root-variance given by
    mu_F = np.exp(mu_lnF + 0.5*sig_lnF**2)
    sig_F = np.sqrt((np.exp(sig_lnF**2) - 1)*np.exp(2*mu_lnF + sig_lnF**2))

    return mu_F, sig_F    

def fluxes_to_magnitudes(F, sig_F, zero_point=22):
    """
    Does the same thing as `magnitudes_to_fluxes` except in reverse.
    """
    e = np.exp(1)
    sig_m = 2.5*np.log10(e)*np.sqrt(np.log(sig_F**2/F**2 + 1))
    mu_m = zero_point - 2.5*np.log10(e)*(np.log(F) -\
         0.5*np.log(1 + sig_F**2/F**2)) 

    return mu_m, sig_m

def constructTsample(lc, maxGap):
    # where observed times are spaced greater than threshold, pick one or more points in the gap s.t. spread
    # is less than threshold.
    # NOTE: doesn't behave exactly correctly, but close enough for now
    tobs = lc[:,0]
    tsample = tobs.copy()
    
    tmin = np.amin(tobs)
    tmax = np.amax(tobs)
    tcand = np.arange(tmin, tmax, maxGap)
    for t in tcand:
        tdiff = np.abs(t - tsample)
        tdiffMin = np.amin(tdiff)
        if tdiffMin < maxGap:
            continue
        idx = np.argmin(tdiff)
        tsample = np.insert(tsample, idx + 1, t)
#        print('inserted', t, 'at ', idx + 1)
        
    return tsample

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def calcParallaxParams(lcName):
    global tc, tp, beta, Omega_0, eps, a
    
    # read in the ogle catalog and Earth orbit params
    stars = np.loadtxt('ogle3_stars.csv', dtype=str)
    dates = np.loadtxt('peri_equi.csv', skiprows=1, delimiter=',')

    # parse lcName into field and star number
    # example lcName: MLLC/photBLG195.4.I.231598.dat
    reLC = re.compile('.*phot(BLG\d+\.\d)\.I\.(\d+).dat')
    matchLC = re.search(reLC, lcName)
    fieldId = matchLC.group(1)
    starId = matchLC.group(2)
    print(lcName, fieldId, starId)

    # extract info
    fieldValues = stars[:,3]
    starIdValues = stars[:,4]
    idx = np.where((fieldValues==fieldId) & (starIdValues == starId))
    if len(idx) == 1:
        star = stars[idx[0][0],:]
        print(star)        
    else:
        print('Star not found, or duplicated')
        return None

    ra = star[1]
    dec = star[2]
    OGLE_t0 = float(star[5])
    OGLE_tE = float(star[8])
    OGLE_u0 = float(star[11])

    rahr,ramin,rasec = ra.split(':')
    rahms = rahr+'h'+ramin+'m'+rasec+'s'

    decdeg,decmin,decsec = dec.split(':')
    decdms = decdeg+'d'+decmin+'m'+decsec+'s'

    coord = SkyCoord(rahms, decdms)
    beta = coord.geocentrictrueecliptic.lat.rad
    
    eps = 0.0167086342
    a=1.0000010178
    Tyear=365.256363004
    Omega_0= 2 * np.pi/Tyear
    
    #compute tp and tc based on pspl mean t0 value
    t0_jd=OGLE_t0+2450000.
    t_0_day = Time(t0_jd,format='jd')
    t_0_yr = np.round(t_0_day.jyear)
    t_base = Time(t_0_yr,format='jyear')
    t_base_hjd_2450000 = t_base.jd-2450000.
    dates_jd = dates[:,8]
    dates_jd_equi = dates[:,9]
    tp = find_nearest(dates_jd,t0_jd)-2450000.
    verneq_day = find_nearest(dates_jd_equi,t0_jd)-tp
    tc = coord.ra.rad/(2*np.pi)*Tyear+verneq_day+t_base_hjd_2450000-2450000.

    print(tc, tp, beta, Omega_0, eps)

def fitML(t, lcMags, lcSigma):
    # convert mags to fluxes
    # fit mean_function_parallax
    flux, sigmaFlux = magnitudes_to_fluxes(lcMags, lcSigma)

    u0_guess = 1.0
    u0_lb = 0
    u0_ub = 100
    
    t0_guess = 0.5*(np.amin(t) + np.amax(t))
    t0_lb = np.amin(t)
    t0_ub = np.amax(t)
    
    tE_guess = 100.0
    tE_lb = 5
    tE_ub = 2000
    
    Fbase_guess = np.mean(flux)
    Fbase_lb = 0.8*Fbase_guess
    Fbase_ub = 1.2*Fbase_guess
    
    DeltaF_guess = 0.1 * Fbase_guess
    DeltaF_lb = 0
    DeltaF_ub = 100*Fbase_guess
    
    guess = np.array([u0_guess, t0_guess, tE_guess, Fbase_guess, DeltaF_guess])
    lb = np.array([u0_lb, t0_lb, tE_lb, Fbase_lb, DeltaF_lb])
    ub = np.array([u0_ub, t0_ub, tE_ub, Fbase_ub, DeltaF_ub])
    

    params, cov = curve_fit(mean_function, t, flux, p0=guess, sigma=sigmaFlux, bounds=(lb,ub))
    return params

def fitMLp(t, lcMags, lcSigma):
    # convert mags to fluxes
    # fit mean_function_parallax
    flux, sigmaFlux = magnitudes_to_fluxes(lcMags, lcSigma)

    u0_guess = 1.0
    u0_lb = 0
    u0_ub = 100
    
    t0_guess = 0.5*(np.amin(t) + np.amax(t))
    t0_lb = np.amin(t)
    t0_ub = np.amax(t)
    
    tE_guess = 100.0
    tE_lb = 5
    tE_ub = 5000
    
    F_B_guess = np.mean(flux)
    F_B_lb = 0.8*F_B_guess
    F_B_ub = 1.2*F_B_guess
    
    F_S_guess = 0.1 * F_B_guess
    F_S_lb = 0
    F_S_ub = 100*F_B_guess

    phi_pi_guess = 0
    phi_pi_lb = 0
    phi_pi_ub = 2.0*np.pi

    pi_E_guess = 0
    pi_E_lb = 0
    pi_E_ub = 1
    
    guess = np.array([F_S_guess, F_B_guess, t0_guess, u0_guess, tE_guess, phi_pi_guess, pi_E_guess])
    lb = np.array([F_S_lb, F_B_lb, t0_lb, u0_lb, tE_lb, phi_pi_lb, pi_E_lb])
    ub = np.array([F_S_ub, F_B_ub, t0_ub, u0_ub, tE_ub, phi_pi_ub, pi_E_ub])

    try:
        params, cov = curve_fit(mean_function_parallax, t, flux, p0=guess, sigma=sigmaFlux, bounds=(lb,ub))
    except RuntimeError:
        return None
        
    return params

def writeIteration(outFile, fitParams, t, lcMags, lcSigma):
    print('# ', file=outFile, end='')
    for p in fitParams:
        print(p, file=outFile, end=' ')
    print(file=outFile)
    
    modelMags = np.zeros_like(t)
        
    for i, tpt in enumerate(t):
        modelFlux = mean_function_parallax(tpt, fitParams[0], fitParams[1], fitParams[2], fitParams[3], fitParams[4], fitParams[5], fitParams[6])
        modelMags[i], sig = fluxes_to_magnitudes(modelFlux, 0.)
        print(tpt, lcMags[i], lcSigma[i], modelMags[i], file=outFile)

    outFile.close()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lc', help = 'lc data file')
    parser.add_argument('--out', help = 'output file')
    parser.add_argument('--nfit', help = 'number of fits to do', type=int)
    parser.add_argument('--maxgap', help = 'maximum time gap in days', type=float)

    args = parser.parse_args()

    if args.lc:
        lcFile = args.lc
    else:
        print('No LC file specified')
        raise

    if args.out:
        outFileName = args.out
    else:
        outFileName = 'test.out'


    if args.nfit:
        nFit = args.nfit
    else:
        nFit = 1


    if args.maxgap:
        maxGap = args.maxgap
    else:
        maxGap = 5.0


    print('lc: {} maxgap: {} nfit: {}'.format(lcFile, maxGap, nFit))
    
    lc = np.loadtxt(lcFile)

    tSample = constructTsample(lc, maxGap)

    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RationalQuadratic() + gp.kernels.WhiteKernel()
    model = GPfit(lc, kernel)
    calcParallaxParams(lcFile)

    lcMags = GPsamples(tSample, model, nFit)
    
    for n in range(nFit):
        lcSigma = modelSigma(tSample, model)
        fitParams = fitMLp(tSample, lcMags[:,n], lcSigma)
        # write fit params and lcSample to output file
        if fitParams is not None:
            outFile = open(outFileName+'.'+str(n), 'w')
            writeIteration(outFile, fitParams, tSample, lcMags[:,n], lcSigma)
            outFile.close()
        
        
        

    



