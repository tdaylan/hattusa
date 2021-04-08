import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from tdpy.util import summgene

pathbase = os.environ['DATA'] + '/hattusa/'
pathdata = pathbase + 'data/'
pathimag = pathbase + 'imag/'

path = pathdata + 'asassn_2021-01-21-21_49_54.csv'
arry = pd.read_csv(path, sep=',', low_memory=False)

# write the RA and DEC to a new file
numbtarg = len(arry['RAJ2000'])
arryoutp = np.empty((numbtarg, 2))
arryoutp[:, 0] = arry['RAJ2000'].values
arryoutp[:, 1] = arry['DEJ2000'].values

print(arry.columns)
for name in arry.columns:
    print(name)
    summgene(arry[name].values)
    print('')

path = pathdata + 'asassn_coor.csv'
print('Writing to %s...' % path)
#np.savetxt(path, arryoutp, delimiter=',')

# read WTVT output
path = pathdata + 'wtv-asassn_coor.csv'
arrytess = np.loadtxt(path, skiprows=62, delimiter=',')
listrasc = arrytess[:, 0]
listdecl = arrytess[:, 1]
boolobsd = arrytess[:, 2:]

# start at Sector 40 with index 39
boolobsdtotl = boolobsd[:, 39:].any(1)

path = pathdata + 'asassn_finl.csv'
print('Writing to %s...' % path)
#np.savetxt(path, arrytess[boolobsdtotl, :2])

indxtargobsd = np.where(boolobsd[:, 39:].any(1))

vmag = arry['Mean Vmag'].values[indxtargobsd]
ampl = arry['Amplitude'].values[indxtargobsd]
peri = arry['Period'].values[indxtargobsd]

figr, axis = plt.subplots(1, 2, figsize=(8, 4))
axis[0].scatter(vmag, ampl, s=0.5)
axis[1].scatter(vmag, peri, s=0.5)

axis[0].set_xlabel('V magnitude')
axis[0].set_ylabel('Amplitude')
axis[1].set_xlabel('V magnitude')
axis[1].set_ylabel('Period [days]')

path = pathimag + 'plot.pdf'
#plt.subplots_adjust(top=0.7)
print('Writing to %s...' % path)
plt.savefig(path)
plt.close()


path = pathdata + 'MAST_Crossmatch_TIC.csv'
print('Reading from %s...' % path)
arry = pd.read_csv(path, sep=',', low_memory=False, header=4)
path = pathdata + 'asassn_save.csv'
print('Writing to %s...' % path)
np.savetxt(path, arry['MatchID'].values, delimiter=',', fmt='%d')

