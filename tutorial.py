import sys

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(10)
figrsizeydob = [8., 4.]
figr, axis = plt.subplots(figsize=figrsizeydob)
axis.plot(x, x)
axis.set_ylabel('Flux')
axis.set_xlabel('Time [BJD]')
plt.savefig('/Users/tdaylan/Desktop/test1.pdf')
plt.close()
            
import hattusa

def cnfg_cplxrota():

    listtici = [206544316]
   
    numbtarg = len(listtici)
    indxtarg = np.arange(numbtarg)
    listticitarg = []
    for k in indxtarg:
        ticitarg = listtici[k]
        listticitarg.append(ticitarg)

    init(listticitarg=listticitarg)


def cnfg_mock():
   
    boolfitt = False
    hattusa.init( \
                 typepopl='simp', \
                 typedata='mock', \
                )


def cnfg_tyr1():
   
    hattusa.init( \
                 # population of stars in the TESS Year 1 flare paper (Guenther+2019)
                 typepopl='tyr1', \
                )


globals().get(sys.argv[1])()

