#!/usr/bin/env python3
"""Audit saved results; optionally replay training; never predict test again."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import sgd
import run_experiments as runner


def verify(res,development_only=False,replay=True):
    frozen=runner.verify_lock(res)
    cfg=json.loads((res/'config_used.json').read_text())
    data=sgd.prepare_snapshot(res/'data_snapshot.npz')
    prep=json.loads((res/'preprocessing.json').read_text())
    assert prep==data['preprocessing'],'Train-only preprocessing replay differs'
    ids=data['indices'];groups=data['groups']
    split_groups={k:set(groups[v]) for k,v in ids.items()}
    assert not (split_groups['train']&split_groups['validation'] or split_groups['train']&split_groups['test'] or split_groups['validation']&split_groups['test'])
    assert sum(len(v) for v in ids.values())==len(data['raw_y'])
    assert len(set.union(*(set(v) for v in ids.values())))==len(data['raw_y'])
    runs=pd.read_csv(res/'all_runs.csv');hist=pd.read_csv(res/'histories.csv')
    assert len(runs)==193 and runs.run_id.nunique()==193
    assert set(hist.run_id)==set(runs.run_id)
    assert len(hist)==len(runs)*(cfg['epochs']+1)
    for row in runs.itertuples():
        trace=hist[hist.run_id==row.run_id].sort_values('epoch')
        np.testing.assert_array_equal(trace.epoch,np.arange(cfg['epochs']+1))
        np.testing.assert_array_equal(trace.samples_seen,trace.epoch*row.N_train)
        np.testing.assert_array_equal(trace.updates,trace.epoch*int(np.ceil(row.N_train/row.batch_size)))
        np.testing.assert_array_equal(trace.head_updates,trace.updates*3)
        assert np.isfinite(trace.train_objective).all()
        for col in ['fit_wall_seconds','optimizer_seconds','monitor_seconds']:
            assert np.all(np.diff(trace[col])>=0)
        assert np.all(trace.fit_wall_seconds>=trace.optimizer_seconds)
        model=np.load(res/row.model_path)
        assert len(model['train_indices'])==row.N_train
        for field in ['train_objective','validation_accuracy','validation_macro_f1','optimizer_seconds','fit_wall_seconds']:
            np.testing.assert_allclose(getattr(row,field),trace.iloc[-1][field],rtol=0,atol=1e-13)
        last_batch=row.N_train%row.batch_size or row.batch_size
        eta=sgd.learning_rate(row.eta0,row.power,cfg['epochs']*row.N_train-last_batch,row.N_train)
        np.testing.assert_allclose(trace.iloc[-1].last_eta,eta,rtol=0,atol=1e-14)
    base=pd.read_csv(res/'baseline_runs.csv')
    assert base.seed.tolist()==cfg['evaluation_seeds']
    X,y=data['X_train'],data['y_train'];Xv,yv=data['X_validation'],data['y_validation']
    for row in base.itertuples():
        model=np.load(res/row.model_path)
        value=sgd.objective(X,sgd.targets(y,data['classes']),model['W'],model['b'],frozen['config']['lam'])
        np.testing.assert_allclose(value,row.train_objective,rtol=0,atol=1e-13)
        metrics=sgd.classification_metrics(yv,sgd.predict(Xv,model['W'],model['b'],data['classes']),data['classes'])
        np.testing.assert_allclose([metrics['accuracy'],metrics['macro_f1']],
                                   [row.validation_accuracy,row.validation_macro_f1],rtol=0,atol=1e-13)
    replay_seed=None
    if replay:
        # A validation/training replay only; no held-out inference occurs here.
        replay_seed=cfg['evaluation_seeds'][0]
        old=np.load(res/frozen['models'][0])
        new=sgd.fit(X,y,Xv,yv,data['classes'],frozen['config'],replay_seed)
        np.testing.assert_array_equal(new['W'],old['W'])
        np.testing.assert_array_equal(new['b'],old['b'])
    test_checked=False
    if not development_only:
        assert (res/'test_complete.json').exists()
        test=pd.read_csv(res/'test_results.csv');pred=pd.read_csv(res/'test_predictions.csv')
        assert test.seed.tolist()==cfg['evaluation_seeds']
        for row in test.itertuples():
            records=pred[pred.seed==row.seed]
            assert len(records)==len(ids['test'])
            # Recompute summaries from stored predictions, never run models on test.
            metrics=sgd.classification_metrics(records.y_true,records.y_pred,data['classes'])
            np.testing.assert_allclose([metrics['accuracy'],metrics['macro_f1']],
                                       [row.test_accuracy,row.test_macro_f1],rtol=0,atol=1e-13)
        test_checked=True
    report={'checked_at_utc':runner.now(),'protected_file_hashes':'pass',
            'customer_disjointness':'pass','train_only_preprocessing_reconstruction':'exact match',
            'unique_fits':len(runs),'history_rows':len(hist),'history_counts_and_timing':'pass',
            'baseline_train_objective_and_validation_metrics':'pass',
            'replayed_seed':replay_seed,'training_replay':'bitwise identical W,b' if replay else 'not requested',
            'stored_test_prediction_summary_checked':test_checked,
            'additional_test_predictions_computed':0}
    name='verification_development.json' if development_only else 'verification.json'
    runner.dump_json(res/name,report);print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--development-only',action='store_true')
    p.add_argument('--no-replay',action='store_true')
    args=p.parse_args()
    with threadpool_limits(1):verify(args.results,args.development_only,not args.no_replay)

if __name__=='__main__':main()
