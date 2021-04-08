from datetime import datetime
import os, sys
import time as timemodu

import numpy as np
import matplotlib.pyplot as plt

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

import tdpy.mcmc
from tdpy.util import summgene
import tdpy.util
import ephesus.util

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
    path = gdat.pathimag + 'lcurmodltotl%s_samp%06d_ns%02d.pdf' % (gdat.strgextn, k, gdat.numbspot)
    plt.subplots_adjust(top=0.7)
    print(f'Writing to {path}...')
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
        gdat.objtstar = fleck.Star(spot_contrast=gdat.cont, phases=phas, u_ld=ldcv)
    
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
        m[ipix_spots] = gdat.cont
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
         ticitarg=None, \
         labltarg=None, \
         strgtarg=None, \
         
         # data
         ## Boolean flag to bin the data
         boolbind=False, \
         boolbdtr=True, \
         durakernbdtrmedi=5., \
         
         strgexpr='TESS', \
        
         # output
         boolwrit=False, \
         
         # data processing
         # Boolean flag to bin the light curve
         boolbinnlcur=False, \

         # model
         boolevol=False, \

         # mock data
         numbplotdraw=100, \

         # Boolean flag to make data
         boolfitt=False, \

         verbtype=1, \

         boolplotpcur=True, \
        ):
    
    # global object to be passed into the sampler
    gdat = tdpy.util.gdatstrt()

    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # global light curve parameters
    # spot contrast
    gdat.cont = 0.7
    
    # fix seed
    np.random.seed(42)
    
    # fudge factor for inflating the uncertainties
    gdat.factstdvfudg = 1e1

    # number of parameters per star
    gdat.numbparastar = 6
    gdat.numbparastartrue = gdat.numbparastar + 1

    # number of parameters per spot
    gdat.numbparaspot = 3
    if gdat.boolevol:
        gdat.numbparaspot += 2
    
    # paths
    gdat.pathbase = os.environ['DATA'] + '/hattusa/'
    if strgtarg is not None:
        pathtarg = gdat.pathbase + strgtarg + '/'
        gdat.pathimag = pathtarg + 'imag/'
        gdat.pathdata = gdat.pathbase + 'data/'
        os.system('mkdir -p %s' % pathtarg)
    else:
        pathtarg = None
        gdat.pathimag = gdat.pathbase + 'imag/mock/'
        gdat.pathdata = gdat.pathbase + 'data/mock/'
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimag)
    
    if ticitarg is None:
        gdat.datatype = 'mock'
    else:
        gdat.datatype = 'inpt'

    if ticitarg is not None:
        # string representing the target
        if strgtarg is None:
            strgtarg = 'TIC%d' % ticitarg
        # label representing the target
        if labltarg is None:
            labltarg = 'TIC %d' % ticitarg
    
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
    
    if gdat.datatype == 'mock':
        gdat.numbtarg = 100
    else:
        gdat.numbtarg = len(gdat.listticitarg)
        
    gdat.indxtarg = np.arange(gdat.numbtarg)

    if gdat.datatype == 'mock':
        
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
                print(f'{k} light curves have been generated.')
        
        # write to FITS file
        if boolwrit:
            hdunprim = fits.PrimaryHDU()
            hduntrue = fits.ImageHDU(gdat.lcurdata)
            hdunlcur = fits.ImageHDU(paratrue)
            listhdun = fits.HDUList([hdunprim, hduntrue, hdunlcur])
            strgtimestmp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = gdat.pathdata + f'lcur_{strgtimestmp}.fits'
            print(f'Writing to {path}')
            listhdun.writeto(path, overwrite=True)
        
    for k in gdat.indxtarg:
        
        # get data
        if gdat.datatype == 'inpt':
            coord = SkyCoord.from_name(liststrgtarg[k])
            pdcsap = lightkurve.search_lightcurvefile(coord, mission=strgexpr).download_all().PDCSAP_FLUX.stitch()
            
            # parse data
            gdat.time = pdcsap.time
            gdat.lcurdata = pdcsap.flux
            gdat.lcurdatastdv = pdcsap.flux_err * gdat.factstdvfudg
            gdat.lcurdatavari = gdat.lcurdatastdv**2

            # get rid of NaNs
            indx = np.where(np.isfinite(gdat.lcurdata) & np.isfinite(gdat.lcurdatavari))[0]
            gdat.time = gdat.time[indx]
            gdat.lcurdatathis = gdat.lcurdata[indx]
            gdat.lcurdatastdvthis = gdat.lcurdatastdv[indx]
            gdat.lcurdatavarithis = gdat.lcurdatavari[indx]
            
            # number of time bins
            gdat.numbtime = gdat.time.size
            
            gdat.minmtime = np.amin(gdat.time)
            gdat.maxmtime = np.amax(gdat.time)
            
            # find flares
            maxmcorr, gdat.listindxtimeposimaxm, timetmpt = ephesus.util.find_flar(gdat.time, gdat.lcurdatathis, \
                                                                                            strgextn=gdat.strgextn, pathimag=gdat.pathimag, \
                                                                                                               boolplot=True, verbtype=gdat.verbtype)
            
            numbkern = len(maxmcorr)
            indxkern = np.arange(numbkern)
            
            # mask out flares
            listindxtimemask = []
            for k in indxkern:
                for indxtime in gdat.listindxtimeposimaxm[k]:
                    indxtimemask = np.arange(indxtime - 60, indxtime + 60)
                    listindxtimemask.append(indxtimemask)
            indxtimemask = np.concatenate(listindxtimemask)
            indxtimemask = np.unique(indxtimemask)
            indxtimegood = np.setdiff1d(np.arange(gdat.time.size), indxtimemask)
            gdat.time = gdat.time[indxtimegood]
            gdat.lcurdata = gdat.lcurdata[indxtimegood]
            gdat.lcurdatastdv = gdat.lcurdatastdv[indxtimegood]
            gdat.numbtime = gdat.time.size
        
        else:
            strgtarg = 'mock%04d' % k
            gdat.timethis = gdat.time
            gdat.lcurdatathis = gdat.lcurdata[k, :]
            gdat.lcurdatastdvthis = gdat.lcurdatastdv[k, :]
        gdat.strgextn = '%s_%s' % (strgtarg, strgexpr)
            
            
        # normalize the light curve
        gdat.medilcurdata = np.median(gdat.lcurdata)
        gdat.lcurdatastdv /= gdat.medilcurdata
        gdat.lcurdata /= gdat.medilcurdata
        
        # plot light curve without the model
        ephesus.util.plot_lcur(gdat.pathimag, timedata=gdat.timethis, lcurdata=gdat.lcurdatathis, strgextn=gdat.strgextn)
        
        gdat.timeunbd = np.copy(gdat.timethis)
        gdat.lcurdataunbd = np.copy(gdat.lcurdatathis)
        gdat.lcurdatastdvunbd = np.copy(gdat.lcurdatastdvthis)
        # bin down the data
        if gdat.boolbind:
            facttime = 200
            print(f'Binning down the data along the time axis by a factor of {facttime} points...')
            gdat.numbtimedown = gdat.numbtime // facttime
            gdat.indxtimedown = np.arange(gdat.numbtimedown)
            gdat.lcurdown = np.empty(gdat.numbtimedown)
            gdat.lcurdownstdv = np.empty(gdat.numbtimedown)
            gdat.meantimedown = np.empty(gdat.numbtimedown)
            for t in gdat.indxtimedown:
                indx = np.arange(t * facttime, (t+1) * facttime)
                gdat.lcurdown[t] = np.mean(gdat.lcurdata[indx])
                gdat.lcurdownstdv[t] = np.sqrt((np.mean(gdat.lcurdatastdv[indx])**2 + np.std(gdat.lcurdata[indx])**2)) / np.sqrt(indx.size)
                gdat.meantimedown[t] = np.mean(gdat.time[indx]) 
            
            gdat.timebind = gdat.meantimedown
            gdat.lcurdatabind = gdat.lcurdown
            gdat.lcurdatastdvbind = gdat.lcurdownstdv
            
            gdat.numbtime = gdat.time.size
        
        else:
            gdat.timebind = None
            gdat.lcurdatabind = None
            gdat.lcurdatastdvbind = None
        
        if gdat.boolbinnlcur:
            gdat.timeused = gdat.timebind
            gdat.lcurdataused = gdat.lcurdatabind
            gdat.lcurdatastdvused = gdat.lcurdatastdvbind
        else:
            gdat.timeused = gdat.timeunbd
            gdat.lcurdataused = gdat.lcurdataunbd
            gdat.lcurdatastdvused = gdat.lcurdatastdvunbd
        
        gdat.lcurdatavari = gdat.lcurdatastdvused**2
        
        if gdat.datatype =='mock':
            if k < numbplotdraw:
                dictpara = pars_para(gdat, paratrue[k, :])
                plot_totl(gdat, k, lcurmodl, lcurmodlevol, lcurmodlspot, dictpara)
        # sample from the posterior
        # plot light curve without the model
        ephesus.util.plot_lcur(gdat.pathimag, timedata=gdat.time, lcurdata=gdat.lcurdataused, \
                       timedatabind=gdat.timebind, lcurdatabind=gdat.lcurdataused, lcurdatastdvbind=gdat.lcurdatastdvused, strgextn=gdat.strgextn)
        
        arrylcur = np.empty((gdat.numbtime, 3))
        arrylcur[:, 0] = gdat.time
        arrylcur[:, 1] = gdat.lcurdataused
        arrylcur[:, 2] = gdat.lcurdatastdvused
        
        ephesus.util.plot_lspe(gdat.pathimag, arrylcur, strgextn=gdat.strgextn)
        
        if boolplotpcur:
            epoc = np.mean(gdat.time)
            listperi = ephesus.util.plot_lspe(gdat.pathimag, arrylcur, strgextn=gdat.strgextn)
            peri = listperi[0]
            print('Phase-folding with the period %g...' % peri)
            ephesus.util.plot_pcur(gdat.pathimag, arrylcur, booltime=True, epoc=epoc, peri=peri, strgextn=gdat.strgextn)

        if gdat.boolfitt:
            # for each spot multiplicity, fit the spot model
            for gdat.numbspot in listindxnumbspot:
                
                print('gdat.numbspot')
                print(gdat.numbspot)

                # list of parameter labels and units
                listlablpara = [['$u_1$', ''], ['$u_2$', ''], ['$P$', 'days'], ['$i$', 'deg'], ['$\\rho$', ''], ['$C$', '']]
                # list of parameter scalings
                listscalpara = ['self', 'self', 'self', 'self', 'self', 'self']
                # list of parameter minima
                listminmpara = [-1., -1., 0.2,   0.,  0.,-1e-1]
                # list of parameter maxima
                listmaxmpara = [ 3.,  3., 0.4, 89.9, 0.6, 1e-1]
                
                for numbspottemp in range(gdat.numbspot):
                    listlablpara += [['$\\theta_{%d}$' % numbspottemp, 'deg'], ['$\\phi_{%d}$' % numbspottemp, 'deg'], ['$R_{%d}$' % numbspottemp, '']]
                    listscalpara += ['self', 'self', 'self']
                    listminmpara += [-90.,   0.,  0.]
                    listmaxmpara += [ 90., 360., 0.4]
                    if gdat.boolevol:
                        listlablpara += [['$T_{s;%d}$' % numbspottemp, 'day'], ['$\\sigma_{s;%d}$' % numbspottemp, '']]
                        listscalpara += ['self', 'self']
                        listminmpara += [gdat.minmtime, 0.1]
                        listmaxmpara += [gdat.maxmtime, 20.]
                        
                listminmpara = np.array(listminmpara)
                listmaxmpara = np.array(listmaxmpara)
                listmeangauspara = None
                liststdvgauspara = None
                
                # number of parameters
                numbpara = len(listlablpara)
                # number of walkers
                numbwalk = max(20, 2 * numbpara)
                    
                numbdata = gdat.lcurdataused.size
                
                # number of degrees of freedom
                gdat.numbdoff = numbdata - numbpara
                
                indxpara = np.arange(numbpara)

                listpost = tdpy.mcmc.samp(gdat, gdat.pathimag, gdat.numbsampwalk, gdat.numbsampburnwalk, gdat.numbsampburnwalkseco, retr_llik, \
                        listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, numbdata, boolpool=True, \
                                #retr_lpri=retr_lpri, \
                                strgextn=gdat.strgextn, samptype='emce')

                # plot light curve
                figr, axis = plt.subplots(figsize=(8, 4))
                # plot samples from the posterior
                ## the sample indices which will be plotted
                gdat.numbsampplot = 10
                indxsampplot = np.random.choice(gdat.indxsamp, size=gdat.numbsampplot, replace=False)
                indxsampplot = np.sort(indxsampplot)
                listlcurmodl = np.empty((gdat.numbsampplot, gdat.numbtime))
                listlcurmodlevol = np.empty((gdat.numbsampplot, gdat.numbspot, gdat.numbtime))
                listlcurmodlspot = np.empty((gdat.numbsampplot, gdat.numbspot, gdat.numbtime))
                for kk, k in enumerate(indxsampplot):
                    # calculate the model light curve for this parameter vector
                    listlcurmodl[kk, :], listlcurmodlevol[kk, :, :], listlcurmodlspot[kk, :, :] = retr_modl(gdat, listpost[k, :])
                    axis.plot(gdat.time, listlcurmodl[kk, :], color='b', alpha=0.1)
                
                # plot components of each sample
                for kk, k in enumerate(indxsampplot):
                    dictpara = pars_para(gdat, listpost[k, :])
                    plot_totl(gdat, k, listlcurmodl[kk, :], listlcurmodlevol[kk, :, :], listlcurmodlspot[kk, :, :], dictpara)

                # plot map
                figr, axis = plt.subplots(figsize=(8, 4))
                gdat.numbside = 2**10
                
                lati = np.empty((gdat.numbsamp, gdat.numbspot))
                lngi = np.empty((gdat.numbsamp, gdat.numbspot))
                rrat = np.empty((gdat.numbsamp, gdat.numbspot))
                for n in gdat.indxsamp:
                    dictpara = pars_para(gdat, listpost[n, :])
                    lati[n, :] = dictpara['lati']
                    lngi[n, :] = dictpara['lngi']
                    rrat[n, :] = dictpara['rrat']
                lati = np.median(lati, 0)
                lngi = np.median(lngi, 0)
                rrat = np.median(rrat, 0)

                print('lati')
                print(lati)
                print('lngi')
                print(lngi)
                print('rrat')
                print(rrat)
                plot_moll(gdat, lati, lngi, rrat)
                
                #for k in indxsampplot:
                #    lati = listpost[k, 1+0*gdat.numbparaspot+0]
                #    lngi = listpost[k, 1+0*gdat.numbparaspot+1]
                #    rrat = listpost[k, 1+0*gdat.numbparaspot+2]
                #    plot_moll(gdat, lati, lngi, rrat)

                for sp in ['right', 'top']:
                    axis.spines[sp].set_visible(False)

                path = gdat.pathimag + f'smap%s_ns%02d.pdf' % (strgtarg, gdat.numbspot)
                print(f'Writing to {path}...')
                plt.savefig(path)
                plt.close()


def cnfg_cplxrota():

    listtici = [206544316]
   
    numbtarg = len(listtici)
    indxtarg = np.arange(numbtarg)
    listticitarg = []
    for k in indxtarg:
        ticitarg = listtici[k]
        listticitarg.append(ticitarg)

    init(listticitarg=listticitarg)


def cnfg_mockcplxrota():
   
    boolfitt = False
    init(boolfitt=boolfitt)


if __name__ == "__main__":
    
    globals().get(sys.argv[1])()
