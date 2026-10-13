"""Code-aligned architecture diagrams (editable SVG, vector PDF, 300 dpi PNG).

Schematic only: no invented radiograph or measured performance is drawn.
"""
import argparse
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PALETTE = dict(detector='#edf3f8', visual='#e3f1ed', texture='#fff0d7',
               detail='#eee7f5', score='#e8ecfa', loss='#fff4f1')


def box(ax, x, y, w, h, text, kind='visual', fontsize=10):
    frame=FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.10',
        facecolor=PALETTE[kind], edgecolor='#536572', linewidth=1.1)
    ax.add_patch(frame)
    # Design at near-final IEEE two-column width, not an 18-inch poster that
    # becomes unreadable when reduced. Wrap labels inside their own boxes.
    fontsize=min(7.3,fontsize*.72)
    scale=7.1/18.*72
    while True:
        columns=max(8,int((w-.12)*scale/(fontsize*.50)))
        wrapped='\n'.join('\n'.join(textwrap.wrap(line,columns,break_long_words=False,
                         break_on_hyphens=False)) for line in text.split('\n'))
        if len(wrapped.split('\n'))*fontsize*1.22 <= (h-.08)*scale or fontsize <= 6.0:
            break
        fontsize-=.2
    label=ax.text(x+w/2, y+h/2, wrapped, ha='center', va='center',
                  fontsize=fontsize, linespacing=1.22, color='#182935')
    label._diagram_box=frame


def arrow(ax, points, dashed=False, color='#526779'):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=1.2, linestyle='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
        mutation_scale=12, linewidth=1.2, color=color,
        linestyle='--' if dashed else '-'))


def canvas(width, height, title):
    fig, ax = plt.subplots(figsize=(7.1, height/width*7.1))
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(.3, height-.20, title, fontsize=9.5, weight='bold', color='#182935')
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.99)
    return fig, ax


def overview():
    fig, ax = canvas(18, 7.9, 'TQ-SymNet: detector and candidate-quality scoring')
    ax.add_patch(FancyBboxPatch((2.1, 4.3), 10.5, 2.7,
        boxstyle='round,pad=.08', facecolor='#fafbfc', edgecolor='#9da8af',
        linestyle='--', linewidth=1.))
    ax.text(2.3, 6.7, 'Stage I: train detector   |   Stage II: freeze detector',
            fontsize=11, color='#54656f')
    box(ax, .3, 5.0, 1.4, 1.0, 'CXR\nimage', 'detector')
    box(ax, 2.35, 4.9, 1.9, 1.2, 'TransNeXt\nbackbone', 'detector')
    box(ax, 4.9, 4.9, 1.9, 1.2, 'CrossScale\nneck: P2-P5', 'detector')
    box(ax, 7.4, 4.65, 2.4, 1.7,
        'Sym classification\nCrossScale regression\n+ Sym adapter', 'detector', 9.2)
    box(ax, 10.35, 4.8, 2.0, 1.4,
        'Decode boxes\n1000 / level\nmax p > .05\nno NMS', 'detector', 9.1)
    for a, b in [((1.7,5.5),(2.35,5.5)), ((4.25,5.5),(4.9,5.5)),
                 ((6.8,5.5),(7.4,5.5)), ((9.8,5.5),(10.35,5.5))]:
        arrow(ax, [a,b])
    box(ax, 2.35, 1.2, 3.1, 1.35,
        'Shared native-image memories\nLBP59 / Gabor\ndetail encoder', 'texture', 10)
    box(ax, 6.65, 1.0, 3.65, 1.75,
        'Candidate quality\nP2-P5 + texture\n+ native side detail\none quality logit zQ', 'visual', 10.2)
    box(ax, 11.25, 1.1, 2.15, 1.55,
        'Logit fusion\n(1-a)zSym + a zQ\nsigmoid(zTB)\na = 0.5', 'score', 9.2)
    box(ax, 14.0, 1.1, 1.8, 1.55,
        'Gaussian\nSoft-NMS\nsigma = 0.7', 'score', 10)
    box(ax, 16.35, 1.1, 1.3, 1.55,
        'Final boxes\n+ scores\ncap: 100', 'score', 9.6)
    arrow(ax, [(1.,5.),(1.,1.9),(2.35,1.9)])
    ax.text(1.15,3.25,'Native CXR', rotation=90, fontsize=9, color='#526779')
    arrow(ax, [(5.45,1.9),(6.65,1.9)])
    arrow(ax, [(5.85,4.9),(5.85,3.6),(7.6,3.6),(7.6,2.75)])
    ax.text(6.2,3.78,'P2-P5 memories',fontsize=9,color='#526779')
    arrow(ax, [(11.35,4.8),(11.35,3.2),(9.3,3.2),(9.3,2.75)])
    ax.text(9.5,3.37,'Fixed box coordinates',fontsize=9,color='#526779')
    arrow(ax, [(10.3,1.85),(11.25,1.85)])
    arrow(ax, [(13.4,1.85),(14.,1.85)])
    arrow(ax, [(15.8,1.85),(16.35,1.85)])
    arrow(ax, [(8.6,6.35),(8.6,7.1),(12.85,7.1),(12.85,2.65)])
    ax.text(10.1,7.23,'Subtype logits zA, zO',fontsize=9,color='#526779')
    ax.text(.35,.45,'Coordinates are unchanged. Memories are encoded once per image. Stage II trains only the quality evaluator.',
            fontsize=10,color='#526779')
    ax.text(13.,3.0,'zSym = max(zA,zO)',fontsize=8.5,color='#526779')
    return fig


def quality_detail():
    fig, ax = canvas(18, 9.8, 'Candidate-quality evaluator: geometry, texture and native detail')
    box(ax,.3,7.8,2.25,1.0,'Box geometry\nXYXY / center / size',fontsize=10)
    box(ax,3.0,7.7,3.,1.25,'Geometry + tokens\n4 broad + 24 side',fontsize=10)
    arrow(ax,[(2.55,8.3),(3.,8.3)])
    box(ax,.3,5.75,2.25,1.05,'P2 / P3 / P4 / P5\nindependent K,V', 'detector')
    box(ax,3.,5.7,3.,1.15,'Per-level attention\nGaussian + side bias\n96D / 6 heads',fontsize=9.8)
    arrow(ax,[(4.5,7.7),(4.5,6.85)])
    arrow(ax,[(2.55,6.3),(3.,6.3)])
    box(ax,6.6,5.7,2.6,1.15,'Level fusion\nretain role tokens\nvisual query V',fontsize=10)
    arrow(ax,[(6.,6.3),(6.6,6.3)])
    box(ax,10.05,7.8,3.0,1.05,'LBP59 memory\nbin embedding + gate','texture',10)
    box(ax,10.05,5.7,3.0,1.05,'Learnable Gabor\ntexture memory','texture',10)
    box(ax,13.8,7.8,3.2,1.05,'Independent LBP\nattention (Q = V)','texture',10)
    box(ax,13.8,5.7,3.2,1.05,'Independent Gabor\nattention (Q = V)','texture',10)
    arrow(ax,[(13.05,8.32),(13.8,8.32)])
    arrow(ax,[(13.05,6.22),(13.8,6.22)])
    arrow(ax,[(7.9,6.85),(7.9,9.15),(15.4,9.15),(15.4,8.85)])
    arrow(ax,[(9.2,6.0),(9.6,6.0),(9.6,7.25),(15.4,7.25),(15.4,6.75)])
    ax.text(10.5,9.27,'Same visual query for both texture reads',fontsize=9,color='#526779')
    box(ax,11.65,3.7,3.4,1.15,'V / LBP / Gabor fusion\nroles + geometry MLP\ndescriptor d',fontsize=10)
    arrow(ax,[(17.,8.32),(17.4,8.32),(17.4,4.3),(15.05,4.3)])
    arrow(ax,[(15.4,5.7),(15.4,4.85),(14.4,4.85)])
    arrow(ax,[(7.9,5.7),(7.9,4.3),(11.65,4.3)])
    box(ax,.3,3.6,2.25,1.15,'Native CXR encoder\nshared D2 / D4','detail',9.8)
    box(ax,3.1,3.55,3.3,1.2,'24 side tokens\ninner / outer positions\n9 samples each','detail',9.8)
    arrow(ax,[(2.55,4.18),(3.1,4.18)])
    arrow(ax,[(12.3,3.7),(12.3,3.0),(4.75,3.0),(4.75,3.55)])
    ax.text(7.15,3.15,'Descriptor-conditioned queries',fontsize=9,color='#526779')
    box(ax,7.0,1.35,3.25,1.15,'Local D2 / D4 read\nall side roles\nresidual update','detail',10)
    arrow(ax,[(6.4,4.18),(6.65,4.18),(6.65,1.92),(7.,1.92)])
    box(ax,11.15,1.35,2.6,1.15,'Linear quality head\none logit zQ','score',11)
    arrow(ax,[(10.25,1.92),(11.15,1.92)])
    box(ax,14.6,1.25,3.0,1.35,'Training only\nBCE + 0.1 rank\nquality-only rank','loss',10)
    arrow(ax,[(14.6,1.92),(13.75,1.92)],dashed=True,color='#b17658')
    ax.text(.4,.5,'t(u) = mean of sigmoid((u - threshold)/0.03), thresholds = 0.50, 0.55, ..., 0.95.',
            fontsize=10,color='#526779')
    ax.text(.4,.18,'sigmoid(zQ) is a threshold surrogate, not literal IoU. GT is used only for training and offline analysis.',
            fontsize=9.5,color='#526779')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='paper_figures')
    args = parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                         'svg.fonttype':'none','savefig.dpi':300})
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    for name, build in [('framework',overview),('quality_network',quality_detail)]:
        fig = build()
        for text in fig.axes[0].texts[1:]:
            text.set_fontsize(min(text.get_fontsize(),7.3))
        # Measure rendered labels, not just approximate character lengths.
        for text in fig.axes[0].texts:
            frame=getattr(text,'_diagram_box',None)
            if frame is None:continue
            for _ in range(30):
                fig.canvas.draw()
                renderer=fig.canvas.get_renderer()
                bounds=frame.get_window_extent(renderer)
                extent=text.get_window_extent(renderer)
                if (extent.x0 >= bounds.x0+2 and extent.x1 <= bounds.x1-2
                        and extent.y0 >= bounds.y0+2 and extent.y1 <= bounds.y1-2):break
                text.set_fontsize(text.get_fontsize()*.96)
                if text.get_fontsize()<5.5:
                    raise RuntimeError('Shorten diagram label: '+text.get_text())
        for suffix in ('png','svg','pdf'):
            fig.savefig(out / (name+'.'+suffix), bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print('Saved two code-aligned diagrams to',out.resolve())


if __name__ == '__main__':
    main()
