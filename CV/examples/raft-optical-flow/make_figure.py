"""Original figure from the executed synthetic experiment and exact counts."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from demo import run


def pyramid_elements(height, width, levels=4, stride=8):
    h,w = height//stride,width//stride
    source = h*w
    total = 0
    for _ in range(levels):
        total += source*h*w
        h,w = h//2,w//2
    return total


def main():
    _,response,_,_ = run()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'svg.fonttype':'none'})
    fig,axes = plt.subplots(1,3,figsize=(15,5.4))
    fig.patch.set_facecolor('#f7f9fc')
    axes[0].set_xlim(3,12); axes[0].set_ylim(9,2); axes[0].set_aspect('equal')
    axes[0].set_xticks(range(4,13,2)); axes[0].set_yticks(range(2,10,2))
    axes[0].grid(color='#dfe5ee')
    axes[0].scatter(7,6,s=100,c='#156da5',label='source (7,6)',zorder=4)
    axes[0].scatter(9,5,s=100,c='#e57939',label='target (9,5)',zorder=4)
    axes[0].annotate('',xy=(9,5),xytext=(7,6),
                     arrowprops={'arrowstyle':'->','lw':2,'color':'#17283e'})
    axes[0].text(6.8,4.35,'flow = (+2, -1)',fontweight='bold')
    axes[0].set_title('1. Forward flow in feature pixels',fontweight='bold')
    axes[0].set_xlabel('x: right'); axes[0].set_ylabel('y: down')
    axes[0].legend(loc='lower left',fontsize=9)
    im=axes[1].imshow(response,cmap='viridis',origin='upper')
    axes[1].scatter(7,6,marker='+',s=120,c='white',linewidths=2)
    axes[1].scatter(9,5,marker='o',s=130,facecolors='none',edgecolors='#ffb66e',lw=2)
    axes[1].add_patch(Rectangle((4.5,3.5),5,5,fill=False,edgecolor='white',lw=1.5))
    axes[1].set_title('2. Correlation map for one source',fontweight='bold')
    axes[1].set_xlabel('target x'); axes[1].set_ylabel('target y')
    fig.text(.385,.075,'White box: radius-2 lookup at zero flow.\n'
             'Orange circle: true correspondence.',fontsize=9,color='#384b63')
    fig.colorbar(im,ax=axes[1],fraction=.04,pad=.035,label='dot product / sqrt(D)')
    sizes=np.array([128,192,256,384,512,768,1024])
    mib=np.array([pyramid_elements(int(s),int(s))*2/2**20 for s in sizes])
    axes[2].plot(sizes,mib,'o-',label='FP16 correlation pyramid',color='#156da5')
    axes[2].plot(sizes,2*mib,'s--',label='FP32 correlation pyramid',color='#e57939')
    axes[2].set_yscale('log'); axes[2].set_xlabel('square input side (pixels)')
    axes[2].set_ylabel('stored correlation only (MiB)')
    axes[2].set_title('3. Correlation storage vs. resolution',fontweight='bold')
    axes[2].grid(alpha=.25); axes[2].legend(fontsize=8,loc='upper left')
    fig.text(.735,.075,'Exact counts: stride 8, four levels.\n'
             'Excludes features, hidden states\nand runtime buffers.',
             fontsize=9,color='#384b63')
    fig.suptitle('RAFT: reusable matching evidence, local queries, recurrent refinement',
                 fontsize=15,fontweight='bold',color='#17283e',y=.99)
    fig.subplots_adjust(left=.06,right=.985,top=.82,bottom=.30,wspace=.50)
    out=Path(__file__).resolve().parents[2]/'assets'/'raft-optical-flow'
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'correlation-and-memory.svg',facecolor=fig.get_facecolor())
    # The PNG is for local visual inspection only.
    fig.savefig(out/'correlation-and-memory.png',dpi=145,facecolor=fig.get_facecolor())
    plt.close(fig)
    for h,w in [(256,256),(480,640),(720,1280),(1088,1920)]:
        n=pyramid_elements(h,w)
        print(f'{h}x{w}: {n} elements; FP16 {n*2/2**20:.3f} MiB; '
              f'FP32 {n*4/2**20:.3f} MiB')


if __name__=='__main__':
    main()
