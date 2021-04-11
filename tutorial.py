import sys

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


def cnfg_mockcplxrota():
   
    boolfitt = False
    hattusa.init( \
                 boolfitt=boolfitt, \
                )


globals().get(sys.argv[1])()

