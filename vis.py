#!/usr/bin/env python3
"""
"""

import argparse
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import *

PALETTE = {STATE.S: 'green', STATE.I: 'red', STATE.R: 'blue'}
LABELS = {STATE.S: 'S', STATE.I: 'I', STATE.R: 'R'}
plt.style.use('ggplot')

##########################################################
def main(indir, outdir):
    info(inspect.stack()[0][3] + '()')
    pospath = pjoin(indir, 'agent_pos.csv')
    plot_positions(pospath, outdir)
    plot_counts(pospath, outdir)

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

        for sirstate, colour in enumerate(['green', 'red', 'blue']):
            df3 = df2.loc[df2.sirstate == sirstate]
            ax.scatter(df3.posx, df3.posy, c=colour)

        # ax.set_xticklabels()
        # ax.set_yticklabels()
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        # ax.axis('off')
        plt.tick_params(left=False, right=False , labelleft=False ,
                labelbottom=False, bottom=False)
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()

##########################################################
def plot_counts(pospath, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(pospath)

    outpath = pjoin(outdir, 'counts.png')
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    for sirstate, colour in PALETTE.items():
        df2 = df.loc[df.sirstate == sirstate]
        df3 = df2.groupby('tick').count()
        ax.plot(df3.index.astype(int), df3.sirstate, c=colour,
                label=LABELS[sirstate])

    ax.set_xlabel('Steps')
    ax.set_ylabel('Count')
    plt.legend()
    plt.savefig(outpath, bbox_inches='tight')
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
