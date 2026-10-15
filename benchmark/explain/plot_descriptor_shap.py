"""Render standalone SHAP panel sets as editable-text SVG and vector PDF."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot(data_path, output_dir):
    d = json.loads(Path(data_path).read_text(encoding='utf-8')); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .5, 'savefig.bbox': 'tight'})
    colors = ['#287EAB', '#B83B50']; cmap = LinearSegmentedColormap.from_list('signed', [colors[0], '#F1EFF2', colors[1]])
    names = [f'{i+1:02d} {f["label"]}' for i, f in enumerate(d['features'])]
    def forest(ax, rows, labels, kind='importance'):
        y = np.arange(len(rows))
        for j, f in enumerate(rows):
            a = f[kind]; ax.errorbar(a['mean'], j, xerr=[[a['mean']-a['lo']], [a['hi']-a['mean']]], fmt='o', color=f.get('color', colors[1]), ms=3, lw=.7, capsize=2)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel('Mean absolute SHAP value')
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.7), gridspec_kw={'width_ratios': [1.1, 1]}, layout='constrained')
    for cls, color, delta, title in [('healthy', colors[0], -.13, 'Healthy'), ('mcao', colors[1], .13, 'MCAO')]:
        vals = [f[cls] for f in d['features']]; means = np.array([v['mean'] for v in vals])
        ax[0].errorbar(means, np.arange(len(vals))+delta, xerr=[means-np.array([v['lo'] for v in vals]), np.array([v['hi'] for v in vals])-means], fmt='o', ms=3, color=color, lw=.7, capsize=2, label=title)
    ax[0].set_yticks(range(len(names)), names); ax[0].invert_yaxis(); ax[0].axvline(0, color='.7', lw=.5)
    ax[0].set(xlabel='Mean signed SHAP value', title='g  Class-specific descriptor contributions'); ax[0].legend(frameon=False)
    im = ax[1].imshow(d['corr'], cmap=cmap, vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(names)), [f'{i+1:02d}' for i in range(len(names))]); ax[1].set_yticks(range(len(names)), [f'{i+1:02d}' for i in range(len(names))])
    ax[1].set_title('h  Descriptor dependencies'); fig.colorbar(im, ax=ax[1], orientation='horizontal', label='Spearman ρ', shrink=.75)
    for ext in ['svg', 'pdf']: fig.savefig(out / ('Fig2_gh_descriptor_SHAP.'+ext))
    plt.close(fig)
    fig, axs = plt.subplots(3, 2, figsize=(7.1, 9), layout='constrained'); axes = axs.ravel()
    forest(axes[0], d['features'], names); axes[0].set_title('a  Global descriptor importance')
    for j, f in enumerate(d['features']):
        x = np.array([v['x'] for v in f['points']]); col = [v['percentile'] for v in f['points']]
        jitter = .26*np.sin(np.arange(len(x))*2.399963)
        sm = axes[1].scatter(x, j+jitter, c=col, vmin=0, vmax=1, cmap=cmap, s=3, linewidths=0)
    axes[1].set_yticks(range(len(names)), names); axes[1].invert_yaxis(); axes[1].axvline(0, color='.7', lw=.5)
    axes[1].set(xlabel='SHAP value for P(MCAO)', title='b  Individual descriptor contributions')
    fig.colorbar(sm, ax=axes[1], orientation='horizontal', label='Descriptor percentile among training donors', shrink=.75)
    forest(axes[2], d['groups'], [f['key'] for f in d['groups']]); axes[2].set(xlabel='Mean absolute group SHAP', title='c  Joint descriptor-family attribution')
    allv=[]
    for f in d['groups']:
        x=[v['x'] for v in f['points']]; y=[v['y'] for v in f['points']]; allv+=x+y
        axes[3].scatter(x, y, color=f['color'], s=4, label=f['key'])
    lo, hi = min(allv), max(allv); pad=(hi-lo)*.05
    axes[3].plot([lo-pad, hi+pad], [lo-pad, hi+pad], color='.7', lw=.5)
    axes[3].set(xlabel='Sum of individual SHAP', ylabel='Joint group SHAP', title='d  Effect of joint grouping'); axes[3].legend(frameon=False, fontsize=5.5)
    for a, case, panel in zip(axes[4:], d['cases'], ['e', 'f']):
        previous=case['baseline']; labs=['Reference']+[f['label'] for f in case['features']]+['Prediction']
        a.plot(previous, 0, 'o', color='#52647F', ms=3)
        for j, f in enumerate(case['features'], 1):
            nextvalue=previous+f['shap']; a.barh(j, abs(f['shap']), left=min(previous, nextvalue), height=.55, color=colors[1 if f['shap']>=0 else 0]); previous=nextvalue
        a.plot(case['probability'], len(labs)-1, 'o', color='#52647F', ms=3)
        a.set_yticks(range(len(labs)), labs); a.invert_yaxis(); a.set(xlabel='P(MCAO)', title=f'{panel}  {case["id"]}')
    for ext in ['svg','pdf']: fig.savefig(out/(f'Supplementary_Fig{d["supplementary_figure"]}_descriptor_SHAP.'+ext))
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',required=True,type=Path);ap.add_argument('--output-dir',required=True,type=Path)
    a=ap.parse_args();plot(a.data,a.output_dir)
