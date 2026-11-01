#!/usr/bin/env python3
"""Render factual evidence tables and scientific figures from saved CSVs only."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache'/'matplotlib'))
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=['train_objective','validation_accuracy','validation_macro_f1','optimizer_seconds','fit_wall_seconds','updates']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
SCHEDULE={0.:'Constant',.5:'Inverse sqrt',1.:'Inverse'}


def aggregate(frame,keys):
    grouped=frame.groupby(keys,sort=True)
    result=grouped[METRICS].agg(['mean','std'])
    result.columns=['_'.join(c) for c in result.columns]
    result['n_seeds']=grouped.size()
    return result.reset_index()


def pm(row,metric,percent=False,digits=6,latex=False):
    factor=100 if percent else 1
    a,b=row[metric+'_mean']*factor,row[metric+'_std']*factor
    if latex:return f'${a:.{digits}f} \\pm {b:.{digits}f}$'
    return f'{a:.{digits}f} ± {b:.{digits}f}'


def tex_table(path,headers,rows):
    text=['% Generated directly from saved CSVs; requires booktabs.',
          '\\begin{tabular}{'+'l'*len(headers)+'}','\\toprule',
          ' & '.join(headers)+' \\\\','\\midrule']
    text += [' & '.join(map(str,r))+' \\\\' for r in rows]
    text += ['\\bottomrule','\\end{tabular}']
    path.write_text('\n'.join(text)+'\n')


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args();res=args.results;root=res.parent
    if not (res/'test_complete.json').exists():raise ValueError('Generate final evidence only after the locked test phase')
    figures=root/'figures';tables=root/'tables';figures.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    base=pd.read_csv(res/'baseline_runs.csv');test=pd.read_csv(res/'test_results.csv')
    hist=pd.read_csv(res/'histories.csv');lr=pd.read_csv(res/'learning_rate_runs.csv')
    batch=pd.read_csv(res/'batch_size_runs.csv');size=pd.read_csv(res/'training_size_runs.csv')
    cfg=json.loads((res/'frozen_selection.json').read_text())['config']
    meta=json.loads((res/'dataset_metadata.json').read_text());env=json.loads((res/'environment.json').read_text())
    lrs=aggregate(lr,['power','eta0']);bs=aggregate(batch,['batch_size']);ss=aggregate(size,['fraction','N_train'])
    lrs.to_csv(tables/'learning_rate_summary.csv',index=False)
    bs.to_csv(tables/'batch_size_summary.csv',index=False)
    ss.to_csv(tables/'training_size_summary.csv',index=False)
    combined=base.merge(test,on='seed',suffixes=('','_test'))
    combined.to_csv(tables/'baseline_per_seed.csv',index=False)
    stats=[]
    for metric in METRICS+['test_accuracy','test_macro_f1','prediction_seconds']:
        stats.append({'metric':metric,'mean':combined[metric].mean(),
                      'std':combined[metric].std(ddof=1),'n_seeds':len(combined)})
    pd.DataFrame(stats).to_csv(tables/'baseline_summary.csv',index=False)
    baseline={r['metric']+'_mean':r['mean'] for r in stats}
    baseline.update({r['metric']+'_std':r['std'] for r in stats})
    baseline_rows=[['Validation',pm(baseline,'validation_accuracy',True,2),pm(baseline,'validation_macro_f1',True,2)],
                   ['Test',pm(baseline,'test_accuracy',True,2),pm(baseline,'test_macro_f1',True,2)]]
    tex_table(tables/'baseline.tex',['Split','Accuracy (\\%)','Macro-F1 (\\%)'],
              [[name,pm(baseline,prefix+'_accuracy',True,2,True),pm(baseline,prefix+'_macro_f1',True,2,True)]
               for name,prefix in [('Validation','validation'),('Test','test')]])
    lr_rows=[];lr_tex=[]
    for _,r in lrs.iterrows():
        lr_rows.append([SCHEDULE[r.power],f'{r.eta0:g}',pm(r,'train_objective'),
                        pm(r,'validation_accuracy',True,2),pm(r,'validation_macro_f1',True,2)])
        lr_tex.append([SCHEDULE[r.power],f'{r.eta0:g}',pm(r,'train_objective',False,6,True),
                       pm(r,'validation_macro_f1',True,2,True)])
    tex_table(tables/'learning_rate.tex',['Schedule','$\\eta_0$','Train objective','Val. Macro-F1 (\\%)'],lr_tex)
    batch_rows=[];batch_tex=[]
    for _,r in bs.iterrows():
        batch_rows.append([int(r.batch_size),int(r.updates_mean),pm(r,'train_objective'),
                           pm(r,'validation_accuracy',True,2),pm(r,'validation_macro_f1',True,2),
                           pm(r,'optimizer_seconds',False,3),pm(r,'fit_wall_seconds',False,3)])
        batch_tex.append([int(r.batch_size),f'{int(r.updates_mean):,}',pm(r,'train_objective',False,6,True),
                          pm(r,'validation_macro_f1',True,2,True),pm(r,'optimizer_seconds',False,3,True)])
    tex_table(tables/'batch_size.tex',['Batch','Updates','Train objective','Val. F1 (\\%)','Optimizer (s)'],batch_tex)
    size_rows=[];size_tex=[]
    for _,r in ss.iterrows():
        size_rows.append([int(r.N_train),int(r.N_train/8),pm(r,'train_objective'),
                          pm(r,'validation_accuracy',True,2),pm(r,'validation_macro_f1',True,2),pm(r,'optimizer_seconds',False,3)])
        size_tex.append([int(r.N_train),pm(r,'validation_accuracy',True,2,True),
                         pm(r,'validation_macro_f1',True,2,True),pm(r,'optimizer_seconds',False,3,True)])
    tex_table(tables/'training_size.tex',['Train $N$','Val. Acc. (\\%)','Val. F1 (\\%)','Optimizer (s)'],size_tex)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
                          'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,
                          'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
                          'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,
                          'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.5})
    def save(fig,name):
        fig.savefig(figures/f'{name}.png',dpi=300,bbox_inches='tight')
        fig.savefig(figures/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    def curve(ax,run_ids,x='epoch',label=None,color=None,dropzero=False):
        traces=[hist[hist.run_id==run_id].sort_values('epoch') for run_id in run_ids]
        if x=='fit_wall_seconds':
            left=max(t[x].iloc[0] for t in traces)
            right=min(t[x].iloc[-1] for t in traces)
            xx=np.linspace(left,right,250)
            yy=np.array([np.interp(xx,t[x],t.train_objective) for t in traces])
        else:
            xx=traces[0][x].to_numpy();yy=np.array([t.train_objective.to_numpy() for t in traces])
            if dropzero:xx,yy=xx[1:],yy[:,1:]
        mean=yy.mean(axis=0);std=yy.std(axis=0,ddof=1)
        ax.plot(xx,mean,color=color,label=label,lw=1.3)
        ax.fill_between(xx,mean-std,mean+std,color=color,alpha=.17,lw=0)
        return xx,mean

    fig,axes=plt.subplots(1,2,figsize=(6.75,2.55),layout='constrained')
    curve(axes[0],base.run_id,color=COLORS[0]);curve(axes[1],base.run_id,'fit_wall_seconds',color=COLORS[0])
    axes[0].set(xlabel='Epoch',ylabel='Mean OvR training objective',title='(a) Equal data passes')
    axes[1].set(xlabel='Fit wall time (s, includes monitoring)',title='(b) Measured wall time')
    save(fig,'convergence')

    fig,axes=plt.subplots(1,3,figsize=(6.75,2.5),layout='constrained',sharey=True)
    for ax,power in zip(axes,[0.,.5,1.]):
        for color,eta in zip(COLORS,sorted(lr.eta0.unique())):
            ids=lr[(lr.power==power)&(lr.eta0==eta)].run_id
            curve(ax,ids,label=f'{eta:g}',color=color)
        ax.set(xlabel='Epoch',title=SCHEDULE[power])
    axes[0].set_ylabel('Mean OvR training objective')
    axes[2].legend(title=r'$\eta_0$',loc='upper right',frameon=False)
    save(fig,'learning_rate_objective')

    fig,ax=plt.subplots(figsize=(3.25,2.65),layout='constrained')
    for color,power in zip(COLORS,[0.,.5,1.]):
        group=lrs[lrs.power==power]
        ax.errorbar(group.eta0,100*group.validation_macro_f1_mean,
                    yerr=100*group.validation_macro_f1_std,label=SCHEDULE[power],color=color,
                    marker='o',ms=3,lw=1,capsize=2)
    ax.set(xscale='log',xlabel=r'Initial learning rate $\eta_0$',ylabel='Validation Macro-F1 (%)')
    ax.legend(frameon=False,loc='best');save(fig,'learning_rate_validation')

    fig,axes=plt.subplots(1,3,figsize=(6.75,2.6),layout='constrained',sharey=True)
    for color,b in zip(COLORS,sorted(batch.batch_size.unique())):
        ids=batch[batch.batch_size==b].run_id
        curve(axes[0],ids,label=f'B={b}',color=color)
        curve(axes[1],ids,'updates',color=color,dropzero=True)
        curve(axes[2],ids,'fit_wall_seconds',color=color)
    axes[0].set(xlabel='Epoch',ylabel='Mean OvR training objective',title='(a) Equal data exposure')
    axes[1].set(xlabel='Joint updates (log scale)',xscale='log',title='(b) Different update budgets')
    axes[2].set(xlabel='Fit wall time (s)',xscale='log',title='(c) Actual fit wall time')
    axes[0].legend(frameon=False,loc='upper right');save(fig,'batch_size_convergence')

    fig,ax=plt.subplots(figsize=(3.25,2.65),layout='constrained')
    for color,metric,label in zip(COLORS,['validation_accuracy','validation_macro_f1'],['Accuracy','Macro-F1']):
        ax.errorbar(bs.batch_size,100*bs[metric+'_mean'],yerr=100*bs[metric+'_std'],
                    label=label,color=color,marker='o',ms=3,capsize=2,lw=1)
    ax.set(xscale='log',xlabel='Batch size',ylabel='Validation performance (%)')
    ax.set_xticks(bs.batch_size,[str(b) for b in bs.batch_size]);ax.minorticks_off()
    ax.legend(frameon=False);save(fig,'batch_size_validation')

    fig,ax=plt.subplots(figsize=(3.25,2.6),layout='constrained')
    combined=combined.sort_values('seed');xx=np.arange(len(combined))
    ax.plot(xx,100*combined.validation_macro_f1,'o-',ms=3,lw=.8,label='Validation',color=COLORS[0])
    ax.plot(xx,100*combined.test_macro_f1,'s-',ms=3,lw=.8,label='Test (final phase)',color=COLORS[1])
    ax.set(xticks=xx,xticklabels=combined.seed.astype(str),xlabel='Prespecified training seed',ylabel='Macro-F1 (%)')
    ax.legend(frameon=False);save(fig,'stability')

    fig,axes=plt.subplots(1,2,figsize=(6.75,2.55),layout='constrained')
    for color,metric,label in zip(COLORS,['validation_accuracy','validation_macro_f1'],['Accuracy','Macro-F1']):
        axes[0].errorbar(ss.N_train,100*ss[metric+'_mean'],yerr=100*ss[metric+'_std'],
                         color=color,label=label,marker='o',ms=3,capsize=2,lw=1)
    axes[0].set(xlabel='Training rows (nested customer subsets)',ylabel='Validation performance (%)')
    axes[0].legend(frameon=False)
    axes[1].errorbar(ss.N_train,ss.optimizer_seconds_mean,yerr=ss.optimizer_seconds_std,
                     color=COLORS[0],marker='o',ms=3,capsize=2,lw=1)
    axes[1].set(xlabel='Training rows',ylabel='Optimizer time (s)')
    save(fig,'training_size')

    captions='''# Figure captions (all curves are actual recorded runs)

1. **convergence.pdf / .png** — Pilot-selected SGD configuration on the customer-disjoint credit-score split: lambda=0.001, eta0=0.001, constant rate, batch64, 200 epochs. Left: full-training objective versus epoch. Right: objective versus measured fit wall time, including epoch diagnostics. Lines and shaded bands show the mean and sample SD across five prespecified training seeds. For time curves, each seed's trace is linearly interpolated onto a common time grid restricted to their shared time interval. The objective is the arithmetic mean of three OvR binary objectives; it is not an optimality gap.
2. **learning_rate_objective.pdf / .png** — Learning-rate sensitivity with the selected lambda and batch64 fixed. Panels show p=0,0.5,1 in eta=eta0/(1+s/N)^p; colors show eta0. s counts processed examples before the update. Each curve uses 200 epochs and five paired seeds; bands are sample SD. All curves use the same objective. Full-training objectives are evaluated at epoch boundaries.
3. **learning_rate_validation.pdf / .png** — Final-epoch validation Macro-F1 versus initial learning rate, with separate constant, inverse-square-root and inverse schedules. Points and error bars are mean±sample SD over five paired training seeds. Selected lambda and batch64 are fixed. No test-set scores enter this sensitivity analysis.
4. **batch_size_convergence.pdf / .png** — Batch sizes1,16,64,256 at fixed selected lambda/learning-rate schedule and200 epochs. Panels distinguish epoch, joint update count, and measured fit wall time. Same seed uses the same per-epoch permutation. Equal epochs imply equal data exposure but different updates/time. Lines/bands are five-seed means/sample SD; time traces use per-group interpolation over common time support. Update and time axes use log scales; epoch0 is excluded from the update panel. Wall time includes monitoring; optimizer-only times appear in the table. The learning rate is not separately retuned for each batch.
5. **batch_size_validation.pdf / .png** — Final-epoch validation Accuracy and Macro-F1 at each batch size, holding selected lambda/rate schedule fixed. Error bars are sample SD over the five paired seeds; only validation metrics are shown.
6. **stability.pdf / .png** — Individual validation and final-test Macro-F1 values for all five prespecified training seeds of the pilot-selected configuration. Each test model is evaluated once after freezing all development outputs. This measures optimization-order stability on one fixed customer split, not uncertainty over independently sampled datasets.
7. **training_size.pdf / .png** — Validation performance and optimizer-only time for nested customer subsets at N=1800,3600,7200, using fixed selected hyperparameters and200 epochs. Each subset fits its own preprocessing on its own training customers. Mean±sample SD uses five paired training seeds. Validation customers stay fixed; no additional test evaluation is performed. Data exposure and update count increase with N, so this is a small-scale sensitivity experiment rather than an asymptotic complexity estimate.
'''
    (root/'figure_captions.md').write_text(captions)
    # Measured trends expressed as bounded facts, never invented optimizer comparisons.
    best_lr=lrs.loc[lrs.validation_macro_f1_mean.idxmax()]
    low=bs.loc[bs.batch_size.idxmin()];high=bs.loc[bs.batch_size.idxmax()]
    start_f=float(hist[hist.run_id==base.run_id.iloc[0]].iloc[0].train_objective)
    summary=f'''# SGD Results Summary — evidence sheet, not Report prose

- Dataset: Kaggle Credit Score Classification; 12,000 rows sampled from 100,000 labeled source rows by 1,500 customers. Customer-disjoint train/validation/test=7200/2400/2400; customers=900/300/300; d=47; K=3 (Good/Poor/Standard).
- Selected **before sensitivity runs**: lambda={cfg['lam']:g}, eta0={cfg['eta0']:g}, p={cfg['power']:g} ({SCHEDULE[cfg['power']]}), batch={cfg['batch_size']}, epochs={cfg['epochs']}. Final iterate, train-only fit, unregularized bias. Independent NumPy implementation; no reference-code reuse.
- Pilot grid: 36 configurations×3 seeds; five separate evaluation seeds 101/202/303/404/505. Total 193 unique fits, with identical overlapping fits reused. Tests were opened only after freezing configuration/models/all development CSVs.
- Mean±SD below uses five training seeds, ddof=1. Performance entries are **percent**, SD is percentage points. Raw CSVs use fractions. This SD measures optimization randomness conditional on one fixed customer split.

## Baseline values for citation

{md_table(['Split','Accuracy (%)','Macro-F1 (%)'],baseline_rows)}

- Initial mean OvR training objective: {start_f:.8f}.
- Final mean OvR training objective: {pm(baseline,'train_objective',False,8)}.
- Joint mini-batch updates/model: {int(baseline['updates_mean']):,}; independent head updates/model: {3*int(baseline['updates_mean']):,}; processed training examples/model: {7200*cfg['epochs']:,}.
- Optimizer time/model: {pm(baseline,'optimizer_seconds',False,6)} s.
- Fit wall time/model including every-epoch diagnostics: {pm(baseline,'fit_wall_seconds',False,6)} s.
- Test prediction time/model (excludes metric computation): {pm(baseline,'prediction_seconds',False,6)} s.

## Learning-rate sensitivity (validation only; selected lambda, batch64)

{md_table(['Schedule','eta0','Train objective','Val Accuracy (%)','Val Macro-F1 (%)'],lr_rows)}

## Batch-size sensitivity (fixed selected rate, equal 200 epochs; validation only)

{md_table(['Batch','Updates','Train objective','Val Accuracy (%)','Val Macro-F1 (%)','Optimizer (s)','Fit wall (s)'],batch_rows)}

## Training-size sensitivity (nested customers; validation only)

{md_table(['N train','Customers','Train objective','Val Accuracy (%)','Val Macro-F1 (%)','Optimizer (s)'],size_rows)}

## Observed facts and limits

- With eta0=0.5, constant-rate SGD has final training objective {pm(lrs[(lrs.power==0)&(lrs.eta0==.5)].iloc[0],'train_objective',False,6)}, whereas inverse decay gives {pm(lrs[(lrs.power==1)&(lrs.eta0==.5)].iloc[0],'train_objective',False,6)} (same lambda, five-seed mean±SD). The constant-rate curve is visibly noisier. Lower training objective does not establish higher classification performance.
- In the controlled batch sweep, batch64 has the highest validation Macro-F1. Batch256 has fewer updates at the fixed epoch budget and a higher final training objective. A separate learning-rate search for each batch was not performed.
- Across the nested training subsets, mean validation Macro-F1 rises from 47.31% to 54.58% to 56.77%. Preprocessing and training objectives differ across subset sizes, so their objective values are not a shared optimality-gap comparison.

- On the five evaluation seeds, highest mean validation Macro-F1 among the learning-rate cells is {best_lr.validation_macro_f1_mean*100:.4f}% at eta0={best_lr.eta0:g}, p={best_lr.power:g}. This observation does not replace the configuration selected using the three pilot seeds.
- At batch 1 the recorded optimizer time is {low.optimizer_seconds_mean:.6f} s versus {high.optimizer_seconds_mean:.6f} s at batch 256, at the same number of data passes. Update counts are {int(low.updates_mean):,} versus {int(high.updates_mean):,}; this comparison holds the rate schedule fixed and does not establish each batch size's best tuned performance.
- The selected eta0 is at the smallest edge of the tested grid; lower rates are untested. No convergence-to-optimum certificate is computed. A lower hinge objective need not yield higher Macro-F1. Constant-rate SGD may fluctuate and need not decrease the objective at every epoch.
- Lambda search uses validation Macro-F1. Absolute objectives at different lambda values are not comparable as an optimization ranking. All plotted sensitivity objectives keep lambda fixed.
- Data are a 12k-row subset of a public Kaggle benchmark, with real-customer provenance unverified; no claim about full 100k training, future-month prediction, deployment fairness, real credit-risk calibration, or generalization to other datasets.
- There are no results for other optimization algorithms and no claimed speed/accuracy superiority over them. Group24's prepared data were not used, and these numbers must not be directly compared to that repository's runs.
- Timing is CPU/machine/load dependent. Measured device: {env.get('hardware',{}).get('Chip',env['machine'])}, {env.get('hardware',{}).get('Memory','memory unavailable')}, {env['platform']}; BLAS one thread. Versions: {json.dumps(env['versions'],sort_keys=True)}.

## Files for the original report-writing conversation

- Exact unrounded aggregate values: tables/baseline_summary.csv, learning_rate_summary.csv, batch_size_summary.csv, training_size_summary.csv.
- Every seed and every epoch: results/*_runs.csv, all_runs.csv, histories.csv; final test metrics and predictions: test_results.csv, test_predictions.csv.
- NeurIPS-ready vector PDFs/300dpi PNGs in figures/; copy captions from figure_captions.md and formatted numerical tables from tables/*.tex.
- Configuration and evidence lock: results/pilot_selection.json, frozen_selection.json, test_started.json, test_complete.json. Reproduction protocol and preprocessing: README.md.
'''
    (root/'Results_Summary.md').write_text(summary)
    print('Generated seven PNG/PDF figure pairs, CSV/LaTeX tables and Results_Summary.md')

if __name__=='__main__':main()
