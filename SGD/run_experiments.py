#!/usr/bin/env python3
"""Staged, reproducible experiments. Test is sealed until development finishes."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.model_selection import train_test_split
import sgd

HERE=Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def dump_json(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    rows=list(rows)
    if not rows:raise ValueError('No CSV rows')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def select_config(rows):
    return min(rows,key=lambda r:(-r['validation_macro_f1'],-r['validation_accuracy'],r['config_id']))['config_id']


def write_lock(out, content):
    if (out/'frozen_selection.json').exists():raise FileExistsError('Selection already frozen')
    content=dict(content)
    content['frozen_at_utc']=now()
    protected=sorted(set(content.get('protected_files',[])) | set(content['models']))
    content['file_hashes']={name:sha(out/name) for name in protected}
    dump_json(out/'frozen_selection.json',content)
    (out/'frozen_selection.sha256').write_text(sha(out/'frozen_selection.json')+'\n')


def verify_lock(out):
    lock=out/'frozen_selection.json'
    if sha(lock)!=(out/'frozen_selection.sha256').read_text().strip():
        raise ValueError('Frozen selection changed')
    content=json.loads(lock.read_text())
    for name,digest in content['file_hashes'].items():
        if sha(out/name)!=digest:raise ValueError(f'Protected artifact changed: {name}')
    return content


def environment():
    hardware={}
    try:
        desc=subprocess.check_output(['system_profiler','SPHardwareDataType','-detailLevel','mini'],text=True)
        for line in desc.splitlines():
            if ':' in line:
                key,value=line.strip().split(':',1)
                if key in ('Model Name','Model Identifier','Chip','Total Number of Cores','Memory'):
                    hardware[key]=value.strip()
    except (OSError,subprocess.CalledProcessError):pass
    import matplotlib
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'machine':platform.machine(),'logical_cpus':os.cpu_count(),'hardware':hardware,
            'versions':{'numpy':np.__version__,'scipy':scipy.__version__,
                        'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,
                        'threadpoolctl':threadpoolctl.__version__},
            'threadpools':threadpoolctl.threadpool_info(),
            'numeric_dtype':'float64','timing_clock':'time.perf_counter',
            'thread_limit':1,'device':'CPU; sequential runs; no GPU',
            'started_at_utc':now()}


def run_development(out, cfg, raw_data_path=None, snapshot_path=None):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory; existing results will not be overwritten')
    out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir()
    start=perf_counter()
    dump_json(out/'config_used.json',cfg)
    dump_json(out/'environment.json',environment())
    if snapshot_path is not None:
        data=sgd.prepare_snapshot(snapshot_path)
    else:
        if raw_data_path is None:raise ValueError('Supply --data train.csv or --snapshot existing/data_snapshot.npz')
        if sha(raw_data_path)!=cfg['raw_train_sha256']:raise ValueError('Raw train.csv checksum mismatch')
        data=sgd.prepare_credit(raw_data_path,cfg['split_seed'],cfg['max_customers'],cfg['sample_seed'])
    X,y=data['X_train'],data['y_train']
    Xv,yv=data['X_validation'],data['y_validation'];classes=data['classes']
    # Deidentified modeling snapshot enables offline reruns without names/SSNs.
    np.savez_compressed(out/'data_snapshot.npz',numeric=data['raw_X'],categorical=data['raw_cat'],
                        y=data['raw_y'],groups=data['groups'],
                        original_row_indices=data['original_row_indices'],
                        **{k+'_indices':v for k,v in data['indices'].items()})
    dump_json(out/'preprocessing.json',data['preprocessing'])
    csv_write(out/'split_indices.csv',(
        {'row_index':int(data['original_row_indices'][i]),'snapshot_index':int(i),
         'customer_code':int(data['groups'][i]),'split':part,'class':int(data['raw_y'][i])}
        for part,indices in data['indices'].items() for i in indices))
    metadata={'name':'Kaggle Credit Score Classification, original labeled train.csv',
              'N_original':100000,'customers_original':12500,'N_total':len(data['raw_y']),
              'customers_sampled':len(np.unique(data['groups'])),
              'N_train':len(X),'N_validation':len(Xv),'N_test':len(data['indices']['test']),
              'd':X.shape[1],'K':len(classes),'class_names':data['class_names'],
              'numeric_features':17,'categorical_features':4,
              'feature_names':data['preprocessing']['feature_names'],
              'raw_train_sha256':cfg['raw_train_sha256'],
              'archive_sha256_observed':'ab5cb9413ad1819700dc006a73b0b03f98b75d9fa0e12b3030b674660497aba6',
              'class_counts':{part:np.bincount(data['raw_y'][ids],minlength=len(classes)).tolist()
                              for part,ids in data['indices'].items()},
              'customers_by_split':{part:len(np.unique(data['groups'][ids])) for part,ids in data['indices'].items()},
              'split_method':'StratifiedGroupKFold 5 folds, shuffle=True, seed2026: fold0 test, fold1 validation, folds2-4 train; groups=customers',
              'sampling':'1500 distinct customers uniformly without replacement, seed3407; all 8 labeled monthly rows retained per selected customer',
              'preprocessing':'fixed per-row parsing, train-median imputation, train 1st/99th percentile clipping, train z-scores; categorical missing token and train-fit one-hot (unknown ignored)',
              'excluded':['ID','Customer_ID (grouping only)','Month','Name','SSN','Type_of_Loan'],
              'source':'https://www.kaggle.com/datasets/parisrohan/credit-score-classification',
              'download_url':'https://www.kaggle.com/api/v1/datasets/download/parisrohan/credit-score-classification',
              'reference_repo_usage':'README and file listing for background only; no source code or prepared data reused',
              'provenance_limit':'Public Kaggle benchmark; no independently verified evidence that records represent real customers; not a validated lending study'}
    dump_json(out/'dataset_metadata.json',metadata)
    histories=[];all_runs=[];cache={};execution=0
    def execute(config,seed,role,subset=None):
        nonlocal execution
        ids=np.arange(len(X)) if subset is None else subset
        key=(tuple(sorted(config.items())),int(seed),hashlib.sha256(ids.tobytes()).hexdigest())
        if key in cache:return cache[key]
        execution+=1
        run_id=f'run_{execution:04d}'
        result=sgd.fit(X[ids],y[ids],Xv,yv,classes,config,seed)
        row={'run_id':run_id,'primary_role':role,'seed':seed,'N_train':len(ids),**config,
             **{k:v for k,v in result['history'][-1].items() if k not in ('epoch','last_eta')}}
        all_runs.append(row)
        histories.extend({'run_id':run_id,**h} for h in result['history'])
        path=f'models/{run_id}.npz'
        np.savez_compressed(out/path,W=result['W'],b=result['b'],classes=classes,train_indices=data['indices']['train'][ids])
        row['model_path']=path
        cache[key]=row
        return row

    base={'batch_size':cfg['pilot_batch_size'],'epochs':cfg['epochs']}
    configs=[dict(base,lam=lam,eta0=eta,power=power)
             for lam,eta,power in itertools.product(cfg['lambda_grid'],cfg['eta0_grid'],cfg['powers'])]
    jobs=list(itertools.product(range(len(configs)),cfg['pilot_seeds']))
    order=np.random.default_rng(cfg['job_order_seed']);order.shuffle(jobs)
    pilot=[]
    for i,(cid,seed) in enumerate(jobs):
        row=execute(configs[cid],seed,'pilot')
        pilot.append({'config_id':cid,**row})
        if (i+1)%12==0:print(f'pilot {i+1}/{len(jobs)} fits',flush=True)
    pilot_summary=[]
    for cid,config in enumerate(configs):
        group=[r for r in pilot if r['config_id']==cid]
        pilot_summary.append({'config_id':cid,**config,
                              'validation_macro_f1':float(np.mean([r['validation_macro_f1'] for r in group])),
                              'validation_accuracy':float(np.mean([r['validation_accuracy'] for r in group])),
                              'train_objective':float(np.mean([r['train_objective'] for r in group]))})
    cid=select_config(pilot_summary);selected=configs[cid]
    csv_write(out/'pilot_runs.csv',pilot);csv_write(out/'pilot_summary.csv',pilot_summary)
    # Selection is written before any five-seed sensitivity or size experiments.
    dump_json(out/'pilot_selection.json',{'selected_config_id':cid,'config':selected,
                                         'selected_at_utc':now(),'rule':cfg['selection_rule']})
    print('PILOT SELECTED '+json.dumps(selected),flush=True)

    jobs=[('learning_rate',dict(selected,eta0=eta,power=power),seed)
          for eta,power,seed in itertools.product(cfg['eta0_grid'],cfg['powers'],cfg['evaluation_seeds'])]
    jobs += [('batch_size',dict(selected,batch_size=batch),seed)
             for batch,seed in itertools.product(cfg['batch_sizes'],cfg['evaluation_seeds'])]
    order.shuffle(jobs);members=[]
    for i,(role,config,seed) in enumerate(jobs):
        members.append({'experiment':role,**execute(config,seed,role)})
        if (i+1)%10==0:print(f'sensitivity {i+1}/{len(jobs)} configurations (cached duplicates reused)',flush=True)
    for role in ('learning_rate','batch_size'):
        csv_write(out/f'{role}_runs.csv',[r for r in members if r['experiment']==role])
    # Nested customer subsets, generated without validation/test labels.
    full=np.arange(len(X));subsets={1.:full}
    train_groups=data['groups'][data['indices']['train']]
    subset_rng=np.random.default_rng(cfg['size_subset_seed'])
    group_order=subset_rng.permutation(np.unique(train_groups))
    for fraction in cfg['training_fractions']:
        kept=group_order[:int(len(group_order)*fraction)]
        subsets[fraction]=np.flatnonzero(np.isin(train_groups,kept))
    size_rows=[]
    size_jobs=list(itertools.product(cfg['training_fractions'],cfg['evaluation_seeds']));order.shuffle(size_jobs)
    for fraction,seed in size_jobs:
        ids=subsets[fraction]
        if fraction==1.:
            row=execute(selected,seed,'training_size')
        else:
            # Every training subset fits all preprocessing from its own customers.
            global_ids=data['indices']['train'][ids]
            X_subset,params=sgd.credit_preprocess(data['raw_X'],data['raw_cat'],global_ids)
            Xs=X_subset[global_ids];Xvs=X_subset[data['indices']['validation']]
            dump_json(out/f'size_preprocessing_{fraction}.json',params)
            execution+=1;run_id=f'run_{execution:04d}'
            result=sgd.fit(Xs,y[ids],Xvs,yv,classes,selected,seed)
            row={'run_id':run_id,'primary_role':'training_size','seed':seed,'N_train':len(ids),**selected,
                 **{k:v for k,v in result['history'][-1].items() if k not in ('epoch','last_eta')}}
            path=f'models/{run_id}.npz';row['model_path']=path
            np.savez_compressed(out/path,W=result['W'],b=result['b'],classes=classes,
                                train_indices=data['indices']['train'][ids])
            all_runs.append(row);histories.extend({'run_id':run_id,**h} for h in result['history'])
        size_rows.append({'fraction':fraction,**row})
        print(f'size fraction={fraction} seed={seed} done',flush=True)
    csv_write(out/'training_size_runs.csv',size_rows)
    csv_write(out/'training_size_indices.csv',(
        {'fraction':fraction,'row_index':int(data['indices']['train'][i])}
        for fraction in cfg['training_fractions'] for i in subsets[fraction]))
    baseline=[execute(selected,seed,'baseline') for seed in cfg['evaluation_seeds']]
    csv_write(out/'baseline_runs.csv',baseline)
    csv_write(out/'all_runs.csv',all_runs);csv_write(out/'histories.csv',histories)
    dump_json(out/'development_complete.json',{'completed_at_utc':now(),
              'unique_fits':execution,'development_wall_seconds':perf_counter()-start,
              'test_metrics_evaluated':False})
    protected=['config_used.json','pilot_selection.json','preprocessing.json','data_snapshot.npz',
               'dataset_metadata.json','baseline_runs.csv','all_runs.csv','histories.csv',
               'pilot_runs.csv','pilot_summary.csv','learning_rate_runs.csv','batch_size_runs.csv',
               'training_size_runs.csv','training_size_indices.csv','development_complete.json']
    protected += [f'size_preprocessing_{f}.json' for f in cfg['training_fractions'] if f!=1.]
    models=[r['model_path'] for r in baseline]
    # Also protect source code: these artifacts document exactly what was run.
    (out/'source').mkdir()
    for name in ('sgd.py','run_experiments.py'):
        (out/'source'/name).write_bytes((HERE/name).read_bytes());protected.append(f'source/{name}')
    write_lock(out,{'config':selected,'seeds':cfg['evaluation_seeds'],'models':models,
                    'protected_files':protected+models,'test_policy':cfg['test_policy'],
                    'selected_from':'pilot only; batch64; no sensitivity retuning'})
    print(f'DEVELOPMENT COMPLETE: {execution} unique fits; test remains unopened',flush=True)


def evaluate_test(out):
    if (out/'test_started.json').exists():
        raise FileExistsError('Test phase has already started; do not re-evaluate or tune after test')
    frozen=verify_lock(out)
    if not (out/'development_complete.json').exists():raise ValueError('Development incomplete')
    # Fail closed: a crashed test phase must be investigated, not silently rerun.
    with (out/'test_started.json').open('x') as f:
        json.dump({'started_at_utc':now(),'selection_sha256':sha(out/'frozen_selection.json')},f,indent=2)
    data=sgd.prepare_snapshot(out/'data_snapshot.npz')
    ids=data['indices']['test'];y=data['raw_y'][ids]
    X=data['X_model'][ids]
    results=[];predictions=[]
    for seed,path in zip(frozen['seeds'],frozen['models']):
        model=np.load(out/path);start=perf_counter()
        prediction=sgd.predict(X,model['W'],model['b'],model['classes'])
        prediction_seconds=perf_counter()-start
        metrics=sgd.classification_metrics(y,prediction,model['classes'])
        results.append({'seed':seed,'model_path':path,'N_test':len(y),
                        'test_accuracy':metrics['accuracy'],'test_macro_f1':metrics['macro_f1'],
                        'prediction_seconds':prediction_seconds})
        predictions.extend({'seed':seed,'row_index':int(i),'y_true':int(a),'y_pred':int(p)}
                           for i,a,p in zip(ids,y,prediction))
    csv_write(out/'test_results.csv',results);csv_write(out/'test_predictions.csv',predictions)
    dump_json(out/'test_complete.json',{'completed_at_utc':now(),'models_evaluated':len(results),
              'selection_sha256':sha(out/'frozen_selection.json'),
              'policy':'all five prespecified seeds evaluated once; no test selection or sensitivity'})
    print('FINAL TEST COMPLETE '+json.dumps(results),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--data',type=Path,help='Original Kaggle train.csv')
    p.add_argument('--snapshot',type=Path,help='Existing deidentified data_snapshot.npz for offline rerun')
    p.add_argument('--stage',choices=['development','test','all'],default='all')
    args=p.parse_args()
    with threadpoolctl.threadpool_limits(limits=1):
        if args.stage in ('development','all'):
            run_development(args.output,json.loads(args.config.read_text()),args.data,args.snapshot)
        if args.stage in ('test','all'):evaluate_test(args.output)

if __name__=='__main__':main()
