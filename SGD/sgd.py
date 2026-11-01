"""Independent mini-batch stochastic subgradient descent for OvR hinge SVMs.

No sklearn estimator is used for training. Columns of W are independent binary
SVMs, each minimizing lambda/2 ||w||^2 + mean(max(0, 1-y*(Xw+b))).
"""
from time import perf_counter
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def objective(X, Y, W, b, lam):
    """Mean across the independently defined binary objectives (not a gap)."""
    penalty = .5 * lam * np.square(W).sum(axis=0)
    hinge = np.maximum(0., 1. - Y * (X @ W + b)).mean(axis=0)
    return float((penalty + hinge).mean())


def subgradient(X, Y, W, b, lam):
    """Per-head subgradient; b is not regularized; kink contribution is zero."""
    active = Y * (X @ W + b) < 1.
    signed_active = Y * active
    m = len(X)  # Actual size, including the last partial batch.
    return lam * W - (X.T @ signed_active) / m, -signed_active.mean(axis=0)


def learning_rate(eta0, power, samples_seen, n_train):
    """Decay by examples processed, so different batches share the same clock."""
    return eta0 / (1. + samples_seen / n_train) ** power


def epoch_batches(n, batch_size, rng):
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def targets(y, classes):
    return np.where(y[:, None] == classes[None, :], 1., -1.)


def predict(X, W, b, classes):
    # np.argmax resolves ties in favor of the first (smallest) class.
    return classes[np.argmax(X @ W + b, axis=1)]


def classification_metrics(y, prediction, classes):
    return {'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, labels=classes,
                                      average='macro', zero_division=0))}


def prepare_digits(split_seed=2026):
    raw_X, raw_y = load_digits(return_X_y=True)
    raw_X = np.asarray(raw_X, dtype=np.float64)
    indices = np.arange(len(raw_y))
    train_val, test = train_test_split(indices, test_size=.2,
                                       random_state=split_seed, stratify=raw_y)
    train, validation = train_test_split(train_val, test_size=.25,
                                         random_state=split_seed + 1,
                                         stratify=raw_y[train_val])
    scaler = StandardScaler().fit(raw_X[train])
    return {'raw_X': raw_X, 'raw_y': raw_y,
            'indices': {'train': train, 'validation': validation, 'test': test},
            'X_train': np.ascontiguousarray(scaler.transform(raw_X[train])),
            'y_train': raw_y[train],
            'X_validation': np.ascontiguousarray(scaler.transform(raw_X[validation])),
            'y_validation': raw_y[validation], 'scaler': scaler,
            'classes': np.unique(raw_y[train])}


def fit(X_train, y_train, X_validation, y_validation, classes, config, seed):
    """Fixed epochs, final iterate. Interface intentionally has no test inputs.

    optimizer_seconds includes initialization, shuffling, slicing and all updates;
    it excludes epoch diagnostics. fit_wall_seconds includes diagnostics. I/O,
    preprocessing, imports and plotting are outside both timers.
    """
    lam, eta0, power = (config[k] for k in ('lam', 'eta0', 'power'))
    batch_size, epochs = (int(config[k]) for k in ('batch_size', 'epochs'))
    if (lam <= 0 or eta0 <= 0 or power < 0 or batch_size < 1 or epochs < 1
            or len(X_train) == 0):
        raise ValueError('Require lambda/eta0>0, power>=0, batch_size/epochs/N>=1')
    wall_start = perf_counter()
    rng = np.random.default_rng(seed)
    Y = targets(y_train, classes)
    W = np.zeros((X_train.shape[1], len(classes)), dtype=np.float64)
    b = np.zeros(len(classes), dtype=np.float64)
    optimizer_seconds = perf_counter() - wall_start
    monitor_seconds = 0.
    updates = samples_seen = 0
    history = []

    def record(epoch, last_eta):
        nonlocal monitor_seconds
        start = perf_counter()
        value = objective(X_train, Y, W, b, lam)
        val = classification_metrics(y_validation,
                                     predict(X_validation, W, b, classes), classes)
        monitor_seconds += perf_counter() - start
        history.append({'epoch': epoch, 'updates': updates,
                        'head_updates': updates * len(classes),
                        'samples_seen': samples_seen, 'last_eta': last_eta,
                        'train_objective': value,
                        'validation_accuracy': val['accuracy'],
                        'validation_macro_f1': val['macro_f1'],
                        'optimizer_seconds': optimizer_seconds,
                        'monitor_seconds': monitor_seconds,
                        'fit_wall_seconds': perf_counter() - wall_start})

    record(0, eta0)
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        for ids in epoch_batches(len(X_train), batch_size, rng):
            eta = learning_rate(eta0, power, samples_seen, len(X_train))
            g, h = subgradient(X_train[ids], Y[ids], W, b, lam)
            W -= eta * g
            b -= eta * h
            updates += 1
            samples_seen += len(ids)
        optimizer_seconds += perf_counter() - start
        record(epoch, eta)
    if not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise FloatingPointError('Nonfinite model: reduce the learning rate')
    return {'W': W, 'b': b, 'history': history}

# Credit dataset support. Parsing uses fixed rules; fitted quantities use train only.
import csv
import re
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

NUMERIC_COLUMNS = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
                   'Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
                   'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
                   'Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month',
                   'Amount_invested_monthly','Monthly_Balance']
CATEGORY_COLUMNS = ['Occupation','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour']
CREDIT_CLASSES = np.array(['Good','Poor','Standard'])


def parse_number(value):
    try:
        x=float(str(value).strip().strip('_'))
        return x if np.isfinite(x) else np.nan
    except (ValueError,TypeError):return np.nan


def parse_history(value):
    match=re.fullmatch(r'(\d+) Years and (\d+) Months',str(value).strip())
    return float(int(match[1])*12+int(match[2])) if match else np.nan


class NumericPreprocessor:
    """Train medians -> train 1st/99th percentile clipping -> train z-scores."""
    def fit(self, X):
        self.median=np.nanmedian(X,axis=0)
        self.median=np.where(np.isfinite(self.median),self.median,0.)
        filled=np.where(np.isnan(X),self.median,X)
        self.lower,self.upper=np.quantile(filled,[.01,.99],axis=0)
        clipped=np.clip(filled,self.lower,self.upper)
        self.mean=clipped.mean(axis=0)
        self.scale=clipped.std(axis=0)
        self.scale=np.where(self.scale==0,1.,self.scale)
        return self

    def transform(self, X):
        filled=np.where(np.isnan(X),self.median,X)
        return (np.clip(filled,self.lower,self.upper)-self.mean)/self.scale

    def parameters(self):
        return {key:getattr(self,key).tolist() for key in ('median','lower','upper','mean','scale')}


def credit_split(y, groups, split_seed):
    fold=np.empty(len(y),dtype=int)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=split_seed)
    for k,(_,ids) in enumerate(splitter.split(np.zeros((len(y),1)),y,groups)):
        fold[ids]=k
    return {'train':np.flatnonzero(fold>=2),'validation':np.flatnonzero(fold==1),
            'test':np.flatnonzero(fold==0)}


def credit_preprocess(numeric, categorical, train_ids):
    prep=NumericPreprocessor().fit(numeric[train_ids])
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64)
    encoder.fit(categorical[train_ids])
    X=np.ascontiguousarray(np.column_stack([prep.transform(numeric),encoder.transform(categorical)]))
    names=NUMERIC_COLUMNS+['Credit_History_Months']+encoder.get_feature_names_out(CATEGORY_COLUMNS).tolist()
    parameters={'numeric':prep.parameters(),'categories':[v.tolist() for v in encoder.categories_],
                'numeric_names':NUMERIC_COLUMNS+['Credit_History_Months'],
                'categorical_names':CATEGORY_COLUMNS,'feature_names':names,
                'missing_rule':'fixed parsing rules; training medians for numeric; __missing__ category',
                'clipping_quantiles':[.01,.99],'onehot_unknown':'all zeros'}
    return X,parameters


def prepare_credit(path, split_seed=2026, max_customers=1500, sample_seed=3407):
    with open(path,newline='') as f:
        rows=list(csv.DictReader(f))
    customers=np.array(sorted({r['Customer_ID'] for r in rows}))
    rng=np.random.default_rng(sample_seed)
    chosen=set(rng.choice(customers,size=min(max_customers,len(customers)),replace=False))
    retained=[(i,r) for i,r in enumerate(rows) if r['Customer_ID'] in chosen]
    group_codes={v:i for i,v in enumerate(sorted(chosen))}
    numeric=[];categorical=[];labels=[];groups=[];original_ids=[]
    for i,r in retained:
        values=[]
        for name in NUMERIC_COLUMNS:
            value=parse_number(r[name])
            if name=='Age' and not (18<=value<=100):value=np.nan
            if name not in ('Changed_Credit_Limit','Delay_from_due_date') and value<0:value=np.nan
            values.append(value)
        values.append(parse_history(r['Credit_History_Age']))
        cats=[]
        for name in CATEGORY_COLUMNS:
            value=r[name].strip()
            if not value or value in ('_','_______','NA','!@9#%8'):value='__missing__'
            cats.append(value)
        numeric.append(values);categorical.append(cats)
        labels.append(int(np.flatnonzero(CREDIT_CLASSES==r['Credit_Score'])[0]))
        groups.append(group_codes[r['Customer_ID']]);original_ids.append(i)
    numeric=np.array(numeric,dtype=np.float64);categorical=np.array(categorical,dtype=str)
    y=np.array(labels);groups=np.array(groups)
    ids=credit_split(y,groups,split_seed)
    X,prep=credit_preprocess(numeric,categorical,ids['train'])
    return {'raw_X':numeric,'raw_cat':categorical,'raw_y':y,'groups':groups,
            'original_row_indices':np.array(original_ids),'indices':ids,
            'X_model':X,'preprocessing':prep,'classes':np.arange(3),
            'class_names':CREDIT_CLASSES.tolist(),
            'X_train':X[ids['train']],'y_train':y[ids['train']],
            'X_validation':X[ids['validation']],'y_validation':y[ids['validation']]}


def prepare_snapshot(path):
    """Reconstruct every fitted transform from the original saved train indices."""
    saved=np.load(path,allow_pickle=False)
    numeric,categorical,y,groups=(saved[k] for k in ('numeric','categorical','y','groups'))
    ids={part:saved[part+'_indices'] for part in ('train','validation','test')}
    X,params=credit_preprocess(numeric,categorical,ids['train'])
    return {'raw_X':numeric,'raw_cat':categorical,'raw_y':y,'groups':groups,
            'original_row_indices':saved['original_row_indices'],'indices':ids,
            'X_model':X,'preprocessing':params,'classes':np.arange(3),
            'class_names':CREDIT_CLASSES.tolist(),
            'X_train':X[ids['train']],'y_train':y[ids['train']],
            'X_validation':X[ids['validation']],'y_validation':y[ids['validation']]}
