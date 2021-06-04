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

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

import miletos
import tdpy
from tdpy import summgene
import ephesus

import pcat


def retr_lcurmodl_flarsing(meantime, timeflar, amplflar, scalrise, scalfall):
    
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


def plot_totl(gdat, k, dictpara):
    
    strgtitlstar = 'N=%d, u1=%.3g, u2=%.3g, i=%.3g deg, P=%.3g day' % (gdat.listnumbspot[k], dictpara['ldc1'], dictpara['ldc2'], \
                                                                            dictpara['incl'], dictpara['prot'])
    strgtitlstar += ', $\\rho$=%.3g' % dictpara['shea']
    protlati = retr_protlati(dictpara['prot'], dictpara['shea'], dictpara['lati'])
    
    liststrgtitlspot = []
    liststrgtitlstarspot = []
    strgtitlstarspot = strgtitlstar
    strgtitltotl = strgtitlstar
    gdat.listcolrspot = ['g', 'r', 'orange', 'cyan', 'm', 'brown', 'violet', 'olive']
    for i in range(gdat.listnumbspot[k]):
        strgtitlspot = '$P_s$=%.3g day; $\\theta$=%.3g deg, $\\phi$=%.3g deg, $R_s$=%.3g' % \
                                                    (protlati[i], dictpara['lati'][i], dictpara['lngi'][i], dictpara['rrat'][i])
        if gdat.boolevolspot:
            strgtitlspot += ', $T_{s}$=%.3g day, $\\sigma_s$=%.3g day' % (dictpara['timecent'][i], dictpara['timestdv'][i])
        strgtitlstarspot = strgtitlstar + '\n' + strgtitlspot
        liststrgtitlspot.append(strgtitlspot)
        liststrgtitlstarspot.append(strgtitlstarspot)
        strgtitltotl += strgtitlspot
    
    figr, axis = plt.subplots(figsize=(8, 4))
    for i in range(gdat.listnumbspot[k]):
        axis.plot(gdat.timethis, gdat.lcurmodlevol[k][i, :], color=gdat.listcolrspot[i], lw=2)
        axis.plot(gdat.timethis, gdat.lcurmodlspot[k][i, :], color=gdat.listcolrspot[i], ls='--', alpha=0.3, lw=2)
    ## raw data
    axis.plot(gdat.timethis, gdat.lcurdata[k], color='grey', ls='', marker='o', ms=0.5, rasterized=True)
    # model
    axis.plot(gdat.timethis, gdat.lcurmodl[k], color='b', lw=3)
    axis.set_xlabel('Time [days]')
    axis.set_ylabel('Relative flux')
    
    axis.text(0.5, 1.3, strgtitlstar, ha='center', transform=axis.transAxes)
    for i in range(gdat.listnumbspot[k]):
        axis.text(0.5, 1.3 - 0.1 * (i + 1), liststrgtitlspot[i], color=gdat.listcolrspot[i], ha='center', transform=axis.transAxes)
    
    path = gdat.pathimagpopl + 'lcurmodltotl%s_targ%06d_ns%02d.pdf' % (gdat.strgextn, k, gdat.listnumbspot[k])
    plt.subplots_adjust(top=0.7)
    print('Writing to %s...' % path)
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
    lcurmodl, lcurmodlevol, lcurmodlspot, lcurmodlflar = retr_lcurmodl_spotflar(gdat, para)
    
    # calculate the log-likelihood
    llik = -0.5 * np.sum((gdat.lcurdatathis - lcurmodl)**2 / gdat.lcurdatavarithis)
    
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
    if gdat.boolevolspot:
        dictpara['timecent'] = para[indx+3]
        dictpara['timestdv'] = para[indx+4]
    
    cntr = np.amax(indx) + 5
    indx = gdat.numbparastar + cntr + np.arange(gdat.numbflar) * gdat.numbparaflar
    
    dictpara['timeflar'] = para[indx+0]
    dictpara['amplflar'] = para[indx+1]
    dictpara['scalrise'] = para[indx+2]
    dictpara['scalfall'] = para[indx+3]
    

    if (dictpara['lati'] > 90).any() or (dictpara['lati'] < -90).any():
        print('pars_para()')
        print('dictpara')
        print(dictpara)
        print('dictpara[lati]')
        print(dictpara['lati'])
        raise Exception('')

    return dictpara


def retr_lcurmodl_spotflar(gdat, para):
    
    # parse the parameter vector
    ## rotation period
    dictpara = pars_para(gdat, para)

    ldcv = [dictpara['ldc1'], dictpara['ldc2']]
    
    lcurmodlspot = np.empty((gdat.numbspot, gdat.timethis.size))
    lcurmodlevol = np.empty((gdat.numbspot, gdat.timethis.size))
    lcurmodlflar = np.empty(gdat.timethis.size)
    for i in range(gdat.numbspot):

        # rotation period at the spot's latitude
        protlati = retr_protlati(dictpara['prot'], dictpara['shea'], dictpara['lati'][i])
    
        # construct the phase grid for this spot
        phas = 360. * gdat.timethis / protlati
        
        # construct the fleck star object 
        gdat.objtstar = fleck.Star(spot_contrast=gdat.contspot, phases=phas, u_ld=ldcv)
    
        # forward-model the light curve
        lcurmodlspot[i, :] = gdat.objtstar.light_curve(dictpara['lngi'][i] * u.deg, dictpara['lati'][i] * u.deg, \
                                                                                        dictpara['rrat'][i], dictpara['incl'] * u.deg)[:, 0]

        if gdat.boolevolspot:
            # functional form of the spot evolution
            funcevol = np.exp(-0.5 * (gdat.time - dictpara['timecent'][i])**2 / dictpara['timestdv'][i]**2)
            
            # calculate the light curve of the spot, subject to evolution 
            lcurmodlevol[i, :] = 1. - (1. - lcurmodlspot[i, :]) * funcevol
        else:
            lcurmodlevol = lcurmodlspot

    # average the light curves from different spots
    lcurmodl = np.sum(lcurmodlevol, 0) + dictpara['cons'] - lcurmodlevol.shape[0] + 1.
    
    numbflar = len(dictpara['timeflar'])
    lcurmodlflar = np.zeros_like(lcurmodl)
    for i in range(numbflar):
        lcurmodlflar += retr_lcurmodl_flarsing(gdat.timethis, dictpara['timeflar'][i], \
                                        dictpara['amplflar'][i], dictpara['scalrise'][i], dictpara['scalfall'][i])
    
    lcurmodl += lcurmodlflar

    return lcurmodl, lcurmodlevol, lcurmodlspot, lcurmodlflar

    
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


def init( \
         # type of population
         typepopl='sc17', \
         # type of data
         typedata='real', \

         # data
         ## Boolean flag to bin the data
         boolbdtr=True, \
         durakernbdtrmedi=5., \
         # type of the simulated phenomenon for mock data
         typemodltrue='spotflar', \

         strgexpr='TESS', \
        
         # mock data
         ## number of light curves to draw
         numbplotdraw=100, \
         # Boolean flag to write the generated mock data to disc
         boolwrit=False, \

         # data processing
         # Boolean flag to bin the light curve
         boolbinnlcur=False, \

         # model
         # Boolean flag to evolve the spots over time
         boolevolspot=False, \

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
    
    gdat.numbparastartrue = gdat.numbparastar + 2

    # number of parameters per flare
    gdat.numbparaflar = 4
    # number of parameters per spot
    gdat.numbparaspot = 3
    if gdat.boolevolspot:
        gdat.numbparaspot += 2
    
    # paths
    gdat.pathbase = os.environ['DATA'] + '/hattusa/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathpopl = gdat.pathbase + gdat.typepopl + '_' + gdat.typedata + '/'
    gdat.pathimagpopl = gdat.pathpopl + 'imag/'
    gdat.pathdatapopl = gdat.pathpopl + 'data/'
    
    # make folders
    for attr, valu in gdat.__dict__.items():
        if attr.startswith('path'):
            os.system('mkdir -p %s' % valu)
    
    dictmileinpt = dict()

    dictmileinpt['pathbasetarg'] = gdat.pathpopl
    dictmileinpt['boolsrchflar'] = True
        
    # list of spot multiplicities to fit
    listindxnumbspot = np.arange(3, 5)
   
    # target selection
    if typepopl == 'sc17':
        pathdatatess = os.environ['TESS_DATA_PATH'] + '/data/'
        path = pathdatatess + 'listtargtsec/all_targets_S%03d_v1.csv' % 17
        objt = pd.read_csv(path, skiprows=5)
        gdat.listticitarg = objt['TICID'].values
        gdat.listticitarg = gdat.listticitarg[5:]
    elif typepopl == 'tyr1':
        gdat.listticitarg = np.array([25118964])
    elif typepopl == 'simp':
        pass

    if gdat.typedata == 'mock':
        gdat.numbtarg = 5
    else:
        gdat.numbtarg = len(gdat.listticitarg)
    
    gdat.strgextn = '%s_%s' % (gdat.typedata, gdat.typepopl)
    print('gdat.numbtarg')
    print(gdat.numbtarg)

    gdat.indxtarg = np.arange(gdat.numbtarg)
    
    gdat.time = [[] for k in gdat.indxtarg]
    if gdat.typedata == 'mock':
        
        # time axis
        minmtime = 0.
        maxmtime = 30.
        difftime = 2. / 60. / 24.
            
        gdat.lcurdata = [[] for k in gdat.indxtarg]
        gdat.lcurdatastdv = [[] for k in gdat.indxtarg]
        gdat.lcurmodl = [[] for k in gdat.indxtarg]
        
        for k in gdat.indxtarg:
            gdat.time[k] = np.arange(minmtime, maxmtime, difftime)
            
        if gdat.typemodltrue == 'spotflar':
            # number of spots
            maxmnumbflar = 20
            gdat.listnumbflar = np.random.randint(1, maxmnumbflar, size=gdat.numbtarg)
            
            # number of spots
            maxmnumbspot = 4
            gdat.listnumbspot = np.random.randint(1, maxmnumbspot, size=gdat.numbtarg)
            
            # limb darkening
            listldc1 = 0.3 + 0.1 * np.random.randn(gdat.numbtarg)
            listldc2 = 0.3 + 0.1 * np.random.randn(gdat.numbtarg)
            
            # rotational period
            listprot = np.random.rand(gdat.numbtarg) * 20. + 20.
            
            # inclinations
            listincl = np.random.rand(gdat.numbtarg) * 90. - 90.
            
            # rate of differential rotation (shear)
            listshea = np.random.rand(gdat.numbtarg) * 0.9 + 0.1
            
            numbparatrue = maxmnumbspot * gdat.numbparaspot + maxmnumbflar * gdat.numbparaflar + gdat.numbparastartrue
            paratrue = np.empty((gdat.numbtarg, numbparatrue))
            paratrue[:, 0] = gdat.listnumbspot
            paratrue[:, 1] = gdat.listnumbflar
            paratrue[:, 2] = listldc1
            paratrue[:, 3] = listldc2
            paratrue[:, 4] = listprot
            paratrue[:, 5] = listincl
            paratrue[:, 6] = listshea
            paratrue[:, 7] = 0.
            
            gdat.lcurmodlspot = [[] for k in gdat.indxtarg]
            gdat.lcurmodlevol = [[] for k in gdat.indxtarg]
            gdat.lcurmodlflar = [[] for k in gdat.indxtarg]
            for k in gdat.indxtarg:
                
                gdat.timethis = gdat.time[k]
                
                gdat.numbspot = gdat.listnumbspot[k]
                gdat.numbflar = gdat.listnumbflar[k]
                gdat.lcurmodlspot[k] = np.empty((gdat.numbspot, gdat.time[k].size))
                gdat.lcurmodlevol[k] = np.empty((gdat.numbspot, gdat.time[k].size))

                indxspot = np.arange(gdat.listnumbspot[k])
                
                # latitude
                listlati = np.random.rand(gdat.listnumbspot[k]) * 180. - 90.
            
                # longitude
                listlngi = np.random.rand(gdat.listnumbspot[k]) * 360.
            
                # radius
                listrrat = np.random.rand(gdat.listnumbspot[k]) * 0.15 + 0.05
            
                if gdat.boolevolspot:
                    # central time of evolution
                    listtimecent = np.random.rand(gdat.listnumbspot[k]) * (maxmtime - minmtime) + minmtime
            
                    # width of time evolution
                    listtimestdv = listprot[k] * (np.random.rand(gdat.listnumbspot[k]) * 2. + 1.)
                
                # time stamps of the flares
                listtimeflar = minmtime + (maxmtime - minmtime) * np.random.rand(gdat.listnumbflar[k])
            
                # amplitudes of the flares
                listamplflar = tdpy.icdf_powr(np.random.rand(gdat.listnumbflar[k]), 1e-6, 1e-2, -2.)
            
                # rise time scales of the flares
                listscalrise = pcat.icdf_gaus(np.random.rand(gdat.listnumbflar[k]), 4. / 24. / 60., 0.4 / 24. / 60.)
            
                # fall time scales of the flares
                listscalfall = pcat.icdf_gaus(np.random.rand(gdat.listnumbflar[k]), 40. / 24. / 60., 4. / 24. / 60.)
                
                cntr = 8
                for i in range(gdat.numbspot):
                    indx = np.arange(gdat.numbspot) * gdat.numbparaspot
                    paratrue[k, cntr+indx+0] = listlati
                    paratrue[k, cntr+indx+1] = listlngi
                    paratrue[k, cntr+indx+2] = listrrat
                    cntr += 3
                    if gdat.boolevolspot:
                        paratrue[k, cntr+indx+3] = listtimecent
                        paratrue[k, cntr+indx+4] = listtimestdv
                        cntr += 2
                for i in range(gdat.numbflar):
                    indx = cntr + np.arange(gdat.numbflar) * gdat.numbparaflar
                    paratrue[k, cntr+indx+0] = listtimeflar
                    paratrue[k, cntr+indx+1] = listamplflar
                    paratrue[k, cntr+indx+2] = listscalrise
                    paratrue[k, cntr+indx+3] = listscalfall
                
                # get model light curve and its components
                gdat.lcurmodl[k], gdat.lcurmodlevol[k], gdat.lcurmodlspot[k], gdat.lcurmodlflar[k] = retr_lcurmodl_spotflar(gdat, paratrue[k, 1:])

        for k in gdat.indxtarg:
            
            # add white noise to the overall light curve to get the synthetic data
            gdat.lcurdata[k] = gdat.lcurmodl[k] + np.random.randn(gdat.time[k].size) * 1e-4
            
            # add red noise
            sigm = np.random.rand() * 0.0005
            rhoo = np.random.rand() * 9. + 1.
            logtsigm = np.log(sigm)
            logtrhoo = np.log(rhoo)
            #noisredd = retr_noisredd(gdat.time, logtsigm, logtrhoo)
            #gdat.lcurdata[k, :] += noisredd
            gdat.lcurdatastdv[k] = gdat.lcurdata[k] * 1e-3
        
        print('%d light curves have been generated.' % gdat.numbtarg)
        
        # write to FITS file
        if boolwrit:
            hdunprim = fits.PrimaryHDU()
            hduntrue = fits.ImageHDU(gdat.lcurdata)
            hdunlcur = fits.ImageHDU(paratrue)
            listhdun = fits.HDUList([hdunprim, hduntrue, hdunlcur])
            strgtimestmp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = gdat.pathdata + 'lcur_{strgtimestmp}.fits'
            print('Writing to %s...' % path)
            listhdun.writeto(path, overwrite=True)
    
    gdat.boolexectmat = True
    
    gdat.strgtarg = np.empty(gdat.numbtarg, dtype=object)
    for k in gdat.indxtarg:
        
        if gdat.typedata == 'mock':
            if k < numbplotdraw:
                gdat.numbspot = gdat.listnumbspot[k]
                gdat.numbflar = gdat.listnumbflar[k]
                dictpara = pars_para(gdat, paratrue[k, 2:])
                plot_totl(gdat, k, dictpara)
            
                gdat.strgtarg[k] = 'targ%06d' % k
                
                # plot
                dictmodl = dict()
                dictmodl['modltotl'] = {'lcur': gdat.lcurmodl[k], 'time': gdat.time[k]}
                for i in np.arange(gdat.numbspot):
                    dictmodl['modlevolsp%02d' % i] = {'lcur': gdat.lcurmodlevol[k][i, :], 'time': gdat.time[k]}
                    dictmodl['modlspotsp%02d' % i] = {'lcur': gdat.lcurmodlspot[k][i, :], 'time': gdat.time[k]}
                dictmodl['modlflar'] = {'lcur': gdat.lcurmodlflar[k], 'time': gdat.time[k]}
                strgextn = '%s_%s' % (gdat.typedata, gdat.strgtarg[k])
                ephesus.plot_lcur(gdat.pathimagpopl, dictmodl=dictmodl, timedata=gdat.time[k], lcurdata=gdat.lcurdata[k], strgextn=strgextn)
        
        if gdat.typedata == 'real':
            listarrytser = None
        
            ticitarg = gdat.listticitarg[k]
            
            labltarg = None
        else:
            arry = np.empty((gdat.time[k].size, 3))
            arry[:, 0] = gdat.time[k] 
            arry[:, 1] = gdat.lcurdata[k]
            arry[:, 2] = gdat.lcurdatastdv[k]
            listarrytser = dict()
            listarrytser['raww'] = [[[[]]], [[[]]]]
            listarrytser['raww'][0][0][0] = arry

            if gdat.typepopl == 'simp':
                ticitarg = None
                labltarg = 'Mock target %d' % k
            else:
                ticitarg = gdat.listticitarg[k]
                labltarg = None

        dictlcurtessinpt = dict()
        dictlcurtessinpt['boolspoconly'] = True
        
        # call miletos to analyze data
        dictmileoutp = miletos.init( \
                                ticitarg=ticitarg, \
                                listarrytser=listarrytser, \
                                labltarg=labltarg, \
                                listtypeanls=['flar', 'spot'], \
                                boolclip=False, \
                                #timescalbdtrspln=0.2, \
                                bdtrtype='medi', \
                                durakernbdtrmedi=0.05, \
                                listtypemodlinfe=['spot'], \
                                boolexectmat=gdat.boolexectmat, \
                                dictlcurtessinpt=dictlcurtessinpt, \
                                **dictmileinpt, \
                               )
        
        
