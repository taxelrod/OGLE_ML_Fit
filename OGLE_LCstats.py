import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def robustBaseline(lcData):

    x = lcData[:,0]
    x = sm.add_constant(x) # x is constant plus time
    y = lcData[:,1]
    sigma = lcData[:,2]
    scale = np.mean(sigma)
    
#    resrlm=sm.RLM(y,x, M=sm.robust.norms.TrimmedMean(c=scale*0.33), missing='drop').fit()
    resrlm=sm.RLM(y,x).fit()
    return resrlm

def trimMLevent(lcData):
    x = np.ones_like(lcData[:,0])
    y = lcData[:,1]
    resrlm = sm.RLM(y, x).fit()
    meanBaseline = resrlm.params
    meanSigma = np.mean(lcData[:,2])
    mlPeak = np.amin(y)
    mlThresh = meanBaseline - 5*meanSigma
    print(mlPeak, mlThresh)
    iMask = np.where(y<mlThresh)
    lcDataNew = np.copy(lcData)
    lcDataNew[iMask,1] = np.nan
    return lcDataNew

def summarizeLCDir(outFileName, matchPattern=None):
    fOut = open(outFileName, 'w')

    print('# name avg slope delta', file=fOut)

    if matchPattern is not None:
        pat = re.compile(matchPattern)
    else:
        pat = re.compile('.*.dat')
        
    for f in os.scandir():
        if pat.match(f.name):
            print(f.name)
            try:
                lcData = np.loadtxt(f.name)
            except ValueError:
                continue
            resrlm = robustBaseline(lcData)
            avg, slope = resrlm.params
            deltaT = np.amax(lcData[:,0]) - np.amin(lcData[:,0])
            print(f.name, avg, slope, slope*deltaT, file=fOut)

    fOut.close()

def buildLCMatrix(matchPattern=None):
    if matchPattern is not None:
        pat = re.compile(matchPattern)
    else:
        pat = re.compile('.*.dat')
        
    First = True
    for f in os.scandir():
        if pat.match(f.name):
            print(f.name)
            try:
                lcData = np.loadtxt(f.name)
            except ValueError:
                continue
            resrlm = robustBaseline(lcData)
            lcDataSubtracted = np.copy(lcData)
            lcDataSubtracted[:,1] -= resrlm.fittedvalues
            print(f.name, np.median(lcDataSubtracted[:,1]))
            if First:
                lcAll = np.copy(lcDataSubtracted)
                First = False
            else:
                lcAll = np.vstack((lcAll, lcDataSubtracted))

    return lcAll
    
def calcLCMatrixStats(lcAll):
    tuniq = np.unique(lcAll[:,0])  # unique times
    stats = np.zeros((len(tuniq),3))

    i = 0
    for t in tuniq:
        idx = np.where(lcAll[:,0] == t)
        if len(idx[0]) > 2:
            lcVals = lcAll[idx[0],1]
            meanDev = np.mean(lcVals)
            stdDev = np.std(lcVals)
            stats[i,0] = t
            stats[i,1] = meanDev
            stats[i,2] = stdDev
            print(t, len(idx[0]), meanDev, stdDev)
            i += 1


    return np.delete(stats, range(i, len(tuniq)), axis=0)

def plotLC(lcFileName, pp=None):
    try:
        lcData = np.loadtxt(lcFileName)
    except ValueError:
        print('Error opening lcFileName')

    plt.figure()
    plt.plot(lcData[:,0], lcData[:,1], '.')

    resrlm = robustBaseline(lcData)
    avg, slope = resrlm.params

    fitLC = avg + lcData[:,0]*slope

    plt.plot(lcData[:,0], fitLC)
    plt.title(lcFileName)
    if pp is not None:
        plt.savefig(pp, format='pdf')
    else:
        plt.show()


def plotLCData(lcData, pp=None):
    plt.figure()
    plt.plot(lcData[:,0], lcData[:,1], '.')

    avg, slope = robustBaseline(lcData)

    fitLC = avg + lcData[:,0]*slope

    plt.plot(lcData[:,0], fitLC)
    if pp is not None:
        plt.savefig(pp, format='pdf')
    else:
        plt.show()


    
        
