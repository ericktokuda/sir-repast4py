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
    pospath = pjoin(indir, 'agent_pos.csv')
    plot_positions(pospath, outdir)

##########################################################
def plot_positions(pospath, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(pospath)

    for t in np.arange(df.tick.min(), df.tick.max() + .01, 1.0):
        df2 = df.loc[df.tick == t]
        tint = int((t * 10) // 10)
        outpath = pjoin(outdir, '{:02d}.png'.format(tint))
        W = 640; H = 480
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        for atype in df.agent_type.unique():
            df3 = df2.loc[df2.agent_type == atype]
            ax.scatter(df3.posx, df3.posy)
        # ax.set_xticklabels()
        # ax.set_yticklabels()
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.axis('off')
        plt.savefig(outpath)
        plt.close()

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
