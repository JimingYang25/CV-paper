"""Reproduce the lesson's original geometric illustration (no model weights)."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,3,figsize=(13.6,4.8),
                             gridspec_kw={'width_ratios':[1,1,1.1]})
    fig.patch.set_facecolor('#f6f8fc')
    colors = ['#146ca4','#de6745','#269b77','#845fc0']
    shapes = [(8,12),(4,6)]
    reference = np.array([0.5,0.5])
    for axis,(h,w) in zip(axes[:2],shapes):
        axis.set_facecolor('white')
        axis.set_xlim(0,1); axis.set_ylim(1,0); axis.set_aspect('equal')
        for x in np.linspace(0,1,w+1):
            axis.axvline(x,color='#dce2ed',lw=0.65,zorder=0)
        for y in np.linspace(0,1,h+1):
            axis.axhline(y,color='#dce2ed',lw=0.65,zorder=0)
        for head,color in enumerate(colors):
            angle = head*np.pi/2
            direction = np.array([np.cos(angle),np.sin(angle)])
            for radius in (1,2):
                point = reference + radius*direction/np.array([w,h])
                axis.annotate('',xy=point,xytext=reference,
                              arrowprops={'arrowstyle':'->','color':color,'lw':1.2})
                axis.scatter(*point,s=65,color=color,zorder=3)
        axis.scatter(*reference,marker='*',s=200,color='#17283e',zorder=5)
        axis.set_title(f'Level {1 if w==12 else 2}: H={h}, W={w}',
                       fontsize=13,fontweight='bold',pad=13)
        axis.set_xlabel('normalized x')
        axis.set_ylabel('normalized y')
        axis.set_xticks([0,.5,1]); axis.set_yticks([0,.5,1])
    axis = axes[2]
    axis.set_xlim(-.18,1.28); axis.set_ylim(1.25,-.2); axis.set_aspect('equal')
    axis.set_facecolor('white')
    for x,y,label in [(0,0,'1'),(1,0,'3'),(0,1,'5'),(1,1,'7')]:
        axis.scatter(x,y,s=550,color='#e2eaf7',edgecolors='#146ca4',zorder=2)
        axis.text(x,y,label,ha='center',va='center',fontweight='bold',zorder=3)
    axis.plot([0,1,1,0,0],[0,0,1,1,0],color='#bcc9da')
    axis.scatter(.25,.75,color='#de6745',s=90,zorder=4)
    axis.annotate('sample (0.25, 0.75)\nvalue = 4.5',xy=(.25,.75),
                  xytext=(.46,.49),fontsize=10,
                  arrowprops={'arrowstyle':'->','color':'#de6745'})
    axis.set_xticks([0,.25,1]); axis.set_yticks([0,.75,1])
    axis.set_xlabel('feature pixel-center x')
    axis.set_ylabel('feature pixel-center y')
    axis.set_title('Bilinear interpolation',fontsize=13,fontweight='bold',pad=13)
    fig.suptitle('Deformable attention: one reference, learned sparse samples',
                 fontsize=17,fontweight='bold',color='#17283e',y=.98)
    fig.text(.5,.025,'Left: schematic initialization, 4 heads and 2 points per head. '
             'The same pixel offset spans more image area at a coarser level.',
             ha='center',fontsize=10,color='#3d5068')
    fig.subplots_adjust(left=.06,right=.98,top=.82,bottom=.20,wspace=.42)
    output = Path(__file__).resolve().parents[2]/'assets'/'deformable-attention'
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'sampling-geometry.svg',facecolor=fig.get_facecolor())
    # Local visual QA only; publication uses the vector figure.
    fig.savefig(output/'sampling-geometry.png',dpi=140,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output/'sampling-geometry.svg')


if __name__ == '__main__':
    main()
