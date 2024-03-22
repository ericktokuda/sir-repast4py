#!/usr/bin/env python3
"""
"""

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import pandas as pd

##########################################################
def main(indir, outdir):
    info(inspect.stack()[0][3] + '()')
    # doneflag = pjoin(outdir, 'DONE')
        #if os.path.exists(doneflag):
        #  info('We may want to know if it ended!')
    #  return    

    # for i in os.listdir('')
    df = pd.read_csv(pjoin(indir, 'agent_pos.csv'))
    breakpoint()
    


    # info('For Aiur!')

    open(doneflag, 'a').write(time.ctime())

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indir', default='./output/', help='Input dir')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output dir')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.indir, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
