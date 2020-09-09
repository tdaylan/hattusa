import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

def plot_lcur(time, lcur, strgextn, titl=''):
    if strgextn != '':
        strgextn = '_' + strgextn
    figr, axis = plt.subplots(figsize=(8, 4))
    axis.plot(time, lcur, color='k', ls='', marker='o', ms=1)
    axis.set_xlabel('Time [days]')
    axis.set_ylabel('Flux')
    axis.set_title(titl)
    namehost = os.uname()[1]
    if namehost == 'TansusUniverse3':
        pathbase = '/Users/tdaylan/Documents/work/data/tdgu/fdla/'
    else:
        pathbase = '/home/tansu_daylan/'
    plt.close()


seed = 42  # any number, a random starting value
np.random.seed(seed)

# list of models
liststrgmodl = ['rfor', 'conv']

# number of time points
numbtime = 500  # TESS ~15000 points, downsized to speed up code

# number of data samples
numbdata = 20000  # 100 lightcurves
numbtran = 18000

# so for this amplitude level, 1e-5 noise equates to a >5 sigma detection
# when we include red noise the detection will worsen


# generate time
# time (between 0 and 27.3 days which is 2*TESS orbit, 1 TESS sector)
time = np.linspace(0., 27.3, numbtime)

# generate data
# data cube shape
shapdatacube = (numbdata, numbtime)

# signal
# lightcurves with random periods between 1 and 9 days
peri = np.random.rand(numbdata) * 9. + 1.
# random phases
phas = np.random.rand(numbdata) * 2. * np.pi
amplsine = -3. + 2 * np.random.rand(numbdata)
amplsine = 10**(amplsine)
sign = amplsine[:, None] * np.sin(2. * np.pi * time[None, :] / peri[:, None] + phas[:, None])

# white noise
dictdata = dict()
dictdata['amplnoiswhit'] = -4. + 2 * np.random.rand(numbdata)
dictdata['amplnoiswhit'] = 10**(dictdata['amplnoiswhit'])
noiswhit = dictdata['amplnoiswhit'][:, None] * np.random.randn(numbtime * numbdata).reshape(shapdatacube)

# red noise 
numbnoisredd = 10
noisredd = np.zeros(shapdatacube)
# amplitude
amplnoisredd = -4. + 2 * np.random.rand(numbdata * numbnoisredd).reshape((numbdata, numbnoisredd))
amplnoisredd = 10**(amplnoisredd)
# phase
phasnois = 2. * np.pi * np.random.rand(numbdata * numbnoisredd).reshape((numbdata, numbnoisredd))
# periods
perinois = -2. + 2.4 * np.random.rand(numbdata * numbnoisredd).reshape((numbdata, numbnoisredd))
perinois = 10**(perinois)
for k in range(numbnoisredd):
    noisredd += amplnoisredd[:, k, None] * np.sin(2. * np.pi * time[None, :] / perinois[:, k, None] + phasnois[:, k, None])

# generate data
data = (1. + sign + noisredd)
data[numbtran:, :] += noiswhit[numbtran:, :]

# break the labels and data into training and test data
datatran = data[:numbtran, :]
datatest = data[numbtran:, :]  # last 100 elements of the array
# parameters
peritran = peri[:numbtran]
peritest = peri[numbtran:]

# run random forest (RF)
# define RF object
objt = RandomForestRegressor(random_state=seed)

# train
# random forest
print('Fitting the random forest...')
objt.fit(datatran, peritran)
# CNN
print('Fitting the CNN...')
pass # call CNN training function here

# predict
# random forest (RF)
print('Predicting using the random forest model...')
predrfor = objt.predict(data)
# CNN
print('Predicting using the CNN model...')
predconv = np.zeros_like(predrfor) # call CNN prediction function here
listpred = [predrfor, predconv]

# RF
# absolute residual
abrerfor = abs((listpred[0] - peri) / peri)
# chsq
chsqrfor = (listpred[0] - peri)**2 / peri**2
# CNN
# absolute residual
abreconv = abs((listpred[1] - peri) / peri)
# chsq
chsqconv = (listpred[1] - peri)**2 / peri**2

# list of labels for features
listlablfeat = ['$|R_r|$', '$\chi^2_r$', 'White Noise Amplitude', 'Signal amplitude']

# list of file names for features
liststrgfeat = ['abrerfor', 'chsqrfor', 'amplnoiswhit', 'amplsign']

# list of scalings for features
listscalfeat = ['logt', 'logt', 'logt', 'logt']

# plot features
print('Plotting features...')
# for all data (b == 0) and only test data (b == 1)
for b in range(2):

    # all data
    if b == 0:
        numbinit = 0
        strgscop = 'totl'
    # test data only
    if b == 1:
        numbinit = numbtran
        strgscop = 'test'

    # define feature vector
    listfeat = [abrerfor, chsqrfor, dictdata['amplnoiswhit'], amplsine]

    listfeat = list(listfeat)
    listfeat = [listfeattemp[numbinit:] for listfeattemp in listfeat]

    # for RF and CNN
    for a in range(2):

        if a == 1:
            continue

        for k, featfrst in enumerate(listfeat):
            print('Working on ' + listlablfeat[k] + '...')

            # histogram plot
            figr, axis = plt.subplots(figsize=(8, 4))
            axis.hist(listfeat[k], 40)
            axis.set_xlabel(listlablfeat[k])
            axis.set_ylabel('N')
            axis.set_yscale('log')
            if listscalfeat[k] == 'logt':
                axis.set_xscale('log')
            path = f'/Users/tdaylan/Documents/work/data/tdgu/fdla/hist{liststrgfeat[k]}_{liststrgmodl[a]}_{strgscop}.pdf'
            print(f'Writing to {path}...')
            plt.savefig(path)
            plt.close()

            for l, featseco in enumerate(listfeat):
                if k >= l:
                    continue
                print('Working on ' + listlablfeat[l] + '...')

                # scatter plot
                figr, axis = plt.subplots(figsize=(8, 4))
                axis.scatter(featfrst, featseco, s=2, color='k')
                axis.set_xlim([np.amin(featfrst), np.amax(featfrst)])
                axis.set_ylim([np.amin(featseco), np.amax(featseco)])
                axis.set_xlabel(listlablfeat[k])
                axis.set_ylabel(listlablfeat[l])
                if listscalfeat[k] == 'logt':
                    axis.set_xscale('log')
                if listscalfeat[l] == 'logt':
                    axis.set_yscale('log')
                # note that even for the highest noise level in amplitude the error is not
                # super high because RF predicts for the period really well.
                path = f'/Users/tdaylan/Documents/work/data/tdgu/fdla/scat_{liststrgfeat[k]}_{liststrgfeat[l]}_{liststrgmodl[a]}_{strgscop}.pdf'
                print(f'Writing to {path}...')
                plt.savefig(path)
                plt.close()


# Making a plot of mock lightcurves with the predicted (regressed) periods
# cites true period and random-forest predicted period for numbplot regressions
print('Making light curve plots...')
numbplot = 30
strgextn = ''
for k in range(numbplot):
    titl = f'White noise ampl: {dictdata["amplnoiswhit"][numbtran+k]:g}, True P: {peri[numbtran+k]:g} \n'
    titl += f'RF-Predicted P {listpred[0][k]:g}, CNN-Predicted P {listpred[1][k]:g}'
    plot_lcur(time, data[numbtran+k, :], strgextn=strgextn, titl=titl)
