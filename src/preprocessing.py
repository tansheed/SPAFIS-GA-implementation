"""
Loading, cleaning, encoding, scaling and splitting.

Leakage rule: imputers / one-hot encoder / min-max scaler are FITTED ON THE
TRAINING SPLIT ONLY and then applied unchanged to validation and test.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list
    feature_names: list
    binary: bool
    info: dict


def load_dataset(path):
    """Read a .csv or .arff file into a DataFrame ('?' and '' become NaN)."""
    path = str(path)
    if path.lower().endswith(".arff"):
        from scipy.io import arff
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        for c in df.columns:      # ARFF nominal values arrive as bytes
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
        return df.replace("?", np.nan)
    return pd.read_csv(path, na_values=["?", "", "NA", "nan", "NaN"], skipinitialspace=True)


def _stratified(idx, y, **kw):
    """train_test_split with stratification when every class has >= 2 members."""
    try:
        return train_test_split(idx, stratify=y, **kw)
    except ValueError:
        return train_test_split(idx, **kw)


def prepare_data(path, target_column, drop_columns, binary_task, normal_labels,
                 split, seed, max_train=None, max_val=None):
    """Full preprocessing pipeline -> DataBundle."""
    df = load_dataset(path)
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    target = target_column or df.columns[-1]
    y_raw = df[target].astype(str).str.strip().str.rstrip(".")
    X_df = df.drop(columns=[target]).replace([np.inf, -np.inf], np.nan)

    # ---- encode target
    if binary_task:
        normal = {s.lower() for s in normal_labels}
        is_normal = y_raw.str.lower().isin(normal)
        if is_normal.sum() == 0:
            print("[warn] no normal label found; treating the majority class as normal")
            is_normal = y_raw == y_raw.value_counts().idxmax()
        y = (~is_normal).astype(int).to_numpy()
        class_names = ["normal", "attack"]
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        class_names = list(le.classes_)

    # ---- split indices FIRST (stratified, reproducible)
    idx = np.arange(len(y))
    tr_idx, rest = _stratified(idx, y, test_size=1 - split["train"], random_state=seed)
    val_frac = split["val"] / (split["val"] + split["test"])
    val_idx, te_idx = _stratified(rest, y[rest], test_size=1 - val_frac, random_state=seed)

    # ---- fit preprocessing on TRAIN ONLY
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scale", MinMaxScaler(clip=True))])   # clip: val/test stay in [0,1]
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    ct.fit(X_df.iloc[tr_idx])
    f = lambda ii: ct.transform(X_df.iloc[ii]).astype(float)
    X_tr, X_val, X_te = f(tr_idx), f(val_idx), f(te_idx)
    y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]
    names = list(ct.get_feature_names_out())

    # ---- optional stratified subsampling (selection only, values unchanged)
    def sub(X, yy, m):
        if m is None or len(yy) <= m:
            return X, yy
        keep, _ = _stratified(np.arange(len(yy)), yy, train_size=m, random_state=seed)
        return X[keep], yy[keep]
    X_tr, y_tr = sub(X_tr, y_tr, max_train)
    X_val, y_val = sub(X_val, y_val, max_val)

    info = {"n_total": len(y), "n_features_raw": X_df.shape[1], "n_features_encoded": X_tr.shape[1],
            "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
            "class_counts_train": dict(zip(*np.unique(y_tr, return_counts=True)))}
    return DataBundle(X_tr, y_tr, X_val, y_val, X_te, y_te, class_names, names, binary_task, info)
