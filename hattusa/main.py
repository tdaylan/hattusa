import datetime
import os, sys
import time as timemodu

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import scipy
import scipy.signal
from scipy import signal
from scipy.optimize import rosen, shgo

import fleck
import lightkurve

import celerite
from celerite import terms

import healpy as hp

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

import miletos
import tdpy
from tdpy import summgene
import ephesus


def retr_lcurflar(meantime, indxtime, listtimeflar, listamplflar, listscalrise, listscalfall):
    
    if meantime.size == 0:
        raise Exception('')
    
    lcur = np.zeros_like(meantime)
    numbflar = len(listtimeflar)
    for k in np.arange(numbflar):
        lcur += retr_lcurflarsing(meantime, listtimeflar[k], listamplflar[k], listscalrise[k], listscalfall[k])

    return lcur


def retr_lcurflarsing(meantime, timeflar, amplflar, scalrise, scalfall):
    
    numbtime = meantime.size
    if numbtime == 0:
        raise Exception('')
    indxtime = np.arange(numbtime)
    indxtimerise = np.where(meantime < timeflar)[0]
    indxtimefall = np.setdiff1d(indxtime, indxtimerise)
    lcur = np.empty_like(meantime)
    lcur[indxtimerise] = np.exp((meantime[indxtimerise] - timeflar) / scalrise)
    lcur[indxtimefall] = np.exp(-(meantime[indxtimefall] - timeflar) / scalfall)
    lcur *= amplflar / np.amax(lcur) 
    
    return lcur


def plot_totl(gdat, k, lcurmodl, lcurmodlevol, lcurmodlspot, dictpara):
    
    strgtitlstar = 'N=%d, u1=%.3g, u2=%.3g, i=%.3g deg, P=%.3g day' % (gdat.numbspot, dictpara['ldc1'], dictpara['ldc2'], \
                                                                            dictpara['incl'], dictpara['prot'])
    strgtitlstar += ', $\\rho$=%.3g' % dictpara['shea']
    protlati = retr_protlati(dictpara['prot'], dictpara['shea'], dictpara['lati'])
    
    liststrgtitlspot = []
    liststrgtitlstarspot = []
    strgtitlstarspot = strgtitlstar
    strgtitltotl = strgtitlstar
    gdat.listcolrspot = ['g', 'r', 'orange', 'cyan', 'm', 'brown', 'violet', 'olive']
    for i in range(gdat.numbspot):
        strgtitlspot = '$P_s$=%.3g day; $\\theta$=%.3g deg, $\\phi$=%.3g deg, $R_s$=%.3g' % \
                                                    (protlati[i], dictpara['lati'][i], dictpara['lngi'][i], dictpara['rrat'][i])
        if gdat.boolevol:
            strgtitlspot += ', $T_{s}$=%.3g day, $\\sigma_s$=%.3g day' % (dictpara['timecent'][i], dictpara['timestdv'][i])
        strgtitlstarspot = strgtitlstar + '\n' + strgtitlspot
        liststrgtitlspot.append(strgtitlspot)
        liststrgtitlstarspot.append(strgtitlstarspot)
        strgtitltotl += strgtitlspot
    
    figr, axis = plt.subplots(figsize=(8, 4))
    for i in range(gdat.numbspot):
        axis.plot(gdat.time, lcurmodlevol[i, :], color=gdat.listcolrspot[i], lw=3)
        axis.plot(gdat.time, lcurmodlspot[i, :], color=gdat.listcolrspot[i], ls='--', alpha=0.3, lw=3)
    ## raw data
    axis.plot(gdat.timethis, gdat.lcurdataunbd, color='grey', ls='', marker='o', ms=0.5, rasterized=True)
    # binned data
    if gdat.boolbind:
        axis.errorbar(gdat.timebind, gdat.lcurdatabind, yerr=gdat.lcurdatastdvbind, color='k', ls='', marker='o', ms=2)
    # model
    axis.plot(gdat.time, lcurmodl, color='b', lw=2)
    axis.set_xlabel('Time [days]')
    axis.set_ylabel('Relative flux')
    axis.text(0.5, 1.3, strgtitlstar, ha='center', transform=axis.transAxes)
    for i in range(gdat.numbspot):
        axis.text(0.5, 1.3 - 0.07 * (i + 1), liststrgtitlspot[i], color=gdat.listcolrspot[i], ha='center', transform=axis.transAxes)
    
    #axis.set_title(strgtitltotl)
    path = gdat.pathpopl + 'lcurmodltotl%s_samp%06d_ns%02d.pdf' % (gdat.strgextn, k, gdat.numbspot)
    plt.subplots_adjust(top=0.7)
    print('Writing to {path}...')
    plt.savefig(path)
    plt.close()


def retr_lpri(para, gdat):
    
    # calculate the model light curve given the parameters
    dictpara = pars_para(gdat, para)
    
    # ensure the radii are ordered
    lpri = 0.
    for i in range(gdat.numbspot - 1):
        if dictpara['rrat'][i] < dictpara['rrat'][i+1]:
            lpri = -np.inf
    
    return lpri


def retr_llik(para, gdat):
    
    # calculate the model light curve given the parameters
    lcurmodl, lcurmodlevol, lcurmodlspot = retr_modl(gdat, para)
    
    # calculate the log-likelihood
    llik = -0.5 * np.sum((gdat.lcurdata - lcurmodl)**2 / gdat.lcurdatavari)
    
    return llik


def pars_para(gdat, para):
    
    dictpara = dict()
    
    dictpara['ldc1'] = para[0]
    dictpara['ldc2'] = para[1]
    dictpara['prot'] = para[2]
    dictpara['incl'] = para[3]
    dictpara['shea'] = para[4]
    dictpara['cons'] = para[5]
    
    indx = gdat.numbparastar + np.arange(gdat.numbspot) * gdat.numbparaspot
    
    dictpara['lati'] = para[indx+0]
    dictpara['lngi'] = para[indx+1]
    dictpara['rrat'] = para[indx+2]
    if gdat.boolevol:
        dictpara['timecent'] = para[indx+3]
        dictpara['timestdv'] = para[indx+4]
    
    if (dictpara['lati'] > 90).any() or (dictpara['lati'] < -90).any():
        print('pars_para()')
        print('dictpara')
        print(dictpara)
        print('dictpara[lati]')
        print(dictpara['lati'])
        raise Exception('')

    return dictpara


def retr_modl(gdat, para):
    
    # parse the parameter vector
    ## rotation period
    dictpara = pars_para(gdat, para)

    ldcv = [dictpara['ldc1'], dictpara['ldc2']]
    
    lcurmodlspot = np.empty((gdat.numbspot, gdat.numbtime))
    lcurmodlevol = np.empty((gdat.numbspot, gdat.numbtime))
    for i in range(gdat.numbspot):

        # rotation period at the spot's latitude
        protlati = retr_protlati(dictpara['prot'], dictpara['shea'], dictpara['lati'][i])
    
        # construct the phase grid for this spot
        phas = 360. * gdat.time / protlati
        
        # construct the fleck star object 
        gdat.objtstar = fleck.Star(spot_contrast=gdat.contspot, phases=phas, u_ld=ldcv)
    
        # forward-model the light curve
        lcurmodlspot[i, :] = gdat.objtstar.light_curve(dictpara['lngi'][i] * u.deg, dictpara['lati'][i] * u.deg, \
                                                                                        dictpara['rrat'][i], dictpara['incl'] * u.deg)[:, 0]

        if gdat.boolevol:
            # functional form of the spot evolution
            funcevol = np.exp(-0.5 * (gdat.time - dictpara['timecent'][i])**2 / dictpara['timestdv'][i]**2)
            
            # calculate the light curve of the spot, subject to evolution 
            lcurmodlevol[i, :] = 1. - (1. - lcurmodlspot[i, :]) * funcevol
        else:
            lcurmodlevol = lcurmodlspot

    # average the light curves from different spots
    lcurmodl = np.sum(lcurmodlevol, 0) + dictpara['cons'] - lcurmodlevol.shape[0] + 1.

    return lcurmodl, lcurmodlevol, lcurmodlspot

    
def retr_protlati(prot, shea, lati):
    """Return the rotation period at a list of latitudes given the shear"""
    
    protlati = prot * (1. + shea * np.sin(lati * np.pi / 180.)**2)

    return protlati


def plot_moll(gdat, lati, lngi, rrat):
    
    numbpixl = hp.nside2npix(gdat.numbside)
    m = np.zeros(numbpixl)
    numbspot = lati.size
    indxspot = np.arange(numbspot)
    for k in indxspot:
        t = np.radians(lati[k] + 90)
        p = np.radians(lngi[k])
        spot_vec = hp.ang2vec(t, p)
        ipix_spots = hp.query_disc(nside=gdat.numbside, vec=spot_vec, radius=rrat[k])
        m[ipix_spots] = gdat.contspot
    cmap = plt.cm.Greys
    cmap.set_under('w')
    hp.mollview(m, cbar=False, title="", cmap=cmap, hold=True, max=1.0, notext=True, flip='geo')
    hp.graticule(color='silver')


def retr_noisredd(time, logtsigm, logtrhoo):
    
    # set up a simple celerite model
    objtkern = celerite.terms.Matern32Term(logtsigm, logtrhoo)
    objtgpro = celerite.GP(objtkern)
    
    # simulate K datasets with N points
    objtgpro.compute(time)
    
    #y = objtgpro.sample(size=1)
    y = objtgpro.sample()
    
    return y[0]


def init( \
         # type of population
         typepopl='sc17', \
         # type of data
         typedata='real', \

         # data
         ## Boolean flag to bin the data
         boolbind=False, \
         boolbdtr=True, \
         durakernbdtrmedi=5., \
         
         strgexpr='TESS', \
        
         # mock data
         ## number of light curves to draw
         numbplotdraw=100, \
         # Boolean flag to write the generated mock data to disc
         boolwrit=False, \

         # data processing
         # Boolean flag to bin the light curve
         boolbinnlcur=False, \
         # Boolean flag to fit data
         boolfitt=False, \

         # model
         # Boolean flag to evolve the spots over time
         boolevol=False, \

         verbtype=1, \

         boolplotpcur=True, \
        ):
    
    # global object to be passed into the sampler
    gdat = tdpy.util.gdatstrt()

    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('hattusa pipeline initialized at %s...' % gdat.strgtimestmp)
    
    # global light curve parameters
    
    # fix seed for data generation
    np.random.seed(42)
        
    gdat.numbparastar = 6
    gdat.contspot = 0.7

    dictparaspot = dict()
    ## spot contrast
    dictparaspot['contspot'] = gdat.contspot
    ## number of parameters per star
    dictparaspot['numbparastar'] = gdat.numbparastar
    
    gdat.numbparastartrue = gdat.numbparastar + 1

    # number of parameters per spot
    gdat.numbparaspot = 3
    if gdat.boolevol:
        gdat.numbparaspot += 2
    
    # paths
    gdat.pathbase = os.environ['DATA'] + '/hattusa/'
    if gdat.typedata == 'mock':
        strgdata = '_mock'
    else:
        strgdata = ''
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathpopl = gdat.pathbase + gdat.typepopl + strgdata + '/'
    gdat.pathimagpopl = gdat.pathpopl + 'imag/'
    gdat.pathdatapopl = gdat.pathpopl + 'data/'
    
    # make folders
    for attr, valu in gdat.__dict__.items():
        if attr.startswith('path'):
            os.system('mkdir -p %s' % valu)
    
    dictmileinpt = dict()

    dictmileinpt['pathbasetarg'] = gdat.pathpopl
    dictmileinpt['boolsrchflar'] = True
    if gdat.boolfitt:
        ## number of samples per walker
        gdat.numbsampwalk = 1000
        ## number of samples to be burned in and not plotted
        gdat.numbsampburnwalk = 0
        ## number of samples to be burned in and plotted
        gdat.numbsampburnwalkseco = int(0.9 * gdat.numbsampwalk)
        
        # list of spot multiplicities to fit
        listindxnumbspot = np.arange(3, 5)
   
        # number of total samples after burn-in
        gdat.numbsamp = (gdat.numbsampwalk - gdat.numbsampburnwalkseco) * numbwalk
        gdat.indxsamp = np.arange(gdat.numbsamp)
    
    # target selection
    if typepopl == 'sc17':
        pathdatatess = os.environ['TESS_DATA_PATH'] + '/data/'
        path = pathdatatess + 'listtargtsec/all_targets_S%03d_v1.csv' % 17
        objt = pd.read_csv(path, skiprows=5)
        gdat.listticitarg = objt['TICID'].values
        gdat.listticitarg = gdat.listticitarg[5:]

    if gdat.typedata == 'mock':
        gdat.numbtarg = 5
    else:
        gdat.numbtarg = len(gdat.listticitarg)
        
    gdat.indxtarg = np.arange(gdat.numbtarg)

    if gdat.typedata == 'mock':
        
        # number of spots
        maxmnumbspot = 4
        listnumbspot = np.random.randint(1, maxmnumbspot, size=gdat.numbtarg)
        
        # limb darkening
        listldc1 = 0.3 + 0.1 * np.random.randn(gdat.numbtarg)
        listldc2 = 0.3 + 0.1 * np.random.randn(gdat.numbtarg)
        
        # rotational period
        listprot = np.random.rand(gdat.numbtarg) * 20. + 20.
        
        # inclinations
        listincl = np.random.rand(gdat.numbtarg) * 90. - 90.
        
        # rate of differential rotation (shear)
        listshea = np.random.rand(gdat.numbtarg) * 0.9 + 0.1
        
        paratrue = np.empty((gdat.numbtarg, maxmnumbspot * gdat.numbparaspot + gdat.numbparastartrue))
        paratrue[:, 0] = listnumbspot
        paratrue[:, 1] = listldc1
        paratrue[:, 2] = listldc2
        paratrue[:, 3] = listprot
        paratrue[:, 4] = listincl
        paratrue[:, 5] = listshea
        paratrue[:, 6] = 0.
        
        # time axis
        gdat.numbtime = 1080
        minmtime = 0.
        maxmtime = 180. * 6. * 1. / 6.
        gdat.time = np.linspace(minmtime, maxmtime, gdat.numbtime)
        
        gdat.lcurdata = np.empty((gdat.numbtarg, gdat.numbtime))
        gdat.lcurdatastdv = np.empty((gdat.numbtarg, gdat.numbtime))
        liststrgtarg = []
        for k in gdat.indxtarg:
            liststrgtarg.append('targ%08d' % k)
            gdat.numbspot = listnumbspot[k]

            indxspot = np.arange(listnumbspot[k])
            
            # latitude
            listlati = np.random.rand(listnumbspot[k]) * 180. - 90.
        
            # longitude
            listlngi = np.random.rand(listnumbspot[k]) * 360.
        
            # radius
            listrrat = np.random.rand(listnumbspot[k]) * 0.15 + 0.05
        
            if gdat.boolevol:
                # central time of evolution
                listtimecent = np.random.rand(listnumbspot[k]) * (maxmtime - minmtime) + minmtime
        
                # width of time evolution
                listtimestdv = listprot[k] * (np.random.rand(listnumbspot[k]) * 2. + 1.)
            
            for i in range(gdat.numbspot):
                indx = gdat.numbparastartrue + np.arange(gdat.numbspot) * gdat.numbparaspot
                paratrue[k, indx+0] = listlati
                paratrue[k, indx+1] = listlngi
                paratrue[k, indx+2] = listrrat
                if gdat.boolevol:
                    paratrue[k, indx+3] = listtimecent
                    paratrue[k, indx+4] = listtimestdv
            
            # get model light curve and its components
            lcurmodl, lcurmodlevol, lcurmodlspot = retr_modl(gdat, paratrue[k, 1:])

            # add white noise to the overall light curve to get the synthetic data
            gdat.lcurdata[k, :] = lcurmodl + np.random.randn(gdat.numbtime) * 1e-4
            
            # add red noise
            sigm = np.random.rand() * 0.0005
            rhoo = np.random.rand() * 9. + 1.
            logtsigm = np.log(sigm)
            logtrhoo = np.log(rhoo)
            noisredd = retr_noisredd(gdat.time, logtsigm, logtrhoo)
            gdat.lcurdata[k, :] += noisredd
            gdat.lcurdatastdv[k, :] = gdat.lcurdata[k, :] * 1e-3
        
            if k % 100 == 0:
                print('{k} light curves have been generated.')
        
        # write to FITS file
        if boolwrit:
            hdunprim = fits.PrimaryHDU()
            hduntrue = fits.ImageHDU(gdat.lcurdata)
            hdunlcur = fits.ImageHDU(paratrue)
            listhdun = fits.HDUList([hdunprim, hduntrue, hdunlcur])
            strgtimestmp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = gdat.pathdata + 'lcur_{strgtimestmp}.fits'
            print('Writing to {path}')
            listhdun.writeto(path, overwrite=True)
    
    gdat.boolexectmat = True

    for k in gdat.indxtarg:
        
        if gdat.typedata =='mock':
            if k < numbplotdraw:
                dictpara = pars_para(gdat, paratrue[k, :])
                plot_totl(gdat, k, lcurmodl, lcurmodlevol, lcurmodlspot, dictpara)
        
        # call miletos to analyze data
        dictmileoutp = miletos.init( \
                                ticitarg=gdat.listticitarg[k], \
                                typemodl='flar', \
                                boolclip=False, \
                                listtypemodlinfe=['spot'], \
                                boolexectmat=gdat.boolexectmat, \
                                **dictmileinpt, \
                               )
        
        
