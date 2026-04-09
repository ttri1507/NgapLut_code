# pip install optuna
# pip install lightgbm
# pip install catboost
# pip install openpyxl

# chạy all model
# !python3 flood_all_models.py --models all --forecast-days 7 --train-end-date 2021-12-31 --test-end-date 2022-12-21 --save-pkl false 

# chạy một (vài) model
# !python3 flood_all_models.py --models xgb etr catboost --forecast-days 7 --train-end-date 2021-12-31 --test-end-date 2022-12-21 --save-pkl false 
import os
import time
import warnings
import argparse
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================
# OPTIONAL IMPORTS
# =========================
HAS_XGB = False
HAS_LGBM = False
HAS_CATBOOST = False
HAS_TF = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        Dropout,
        Input,
        Conv1D,
        GlobalAveragePooling1D,
        Add,
        Activation,
    )
    from tensorflow.keras.models import Sequential, Model
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except Exception:
    tf = None
    EarlyStopping = None
    LSTM = Dense = Dropout = Input = Conv1D = GlobalAveragePooling1D = Add = Activation = None
    Sequential = Model = None


# =========================
# DEFAULT CONFIG
# =========================
DATA_PATH = "preprocessed_flood_data3.csv"
VALIDATION_PATH = "validation_2023.csv"
OUTPUT_FOLDER = "results/flood_all_models"

TARGETS = ["H Phú An Max (m)", "H Nhà Bè Max (m)"]

DEFAULT_TRAIN_END_DATE = "2021-12-31"
DEFAULT_TEST_END_DATE = "2022-12-31"
DEFAULT_FORECAST_DAYS = 7
DEFAULT_SAVE_PKL = False

DEFAULT_SELECTED_MODELS = ["xgb", "rf", "lstm", "lgbm", "catboost", "etr", "hgbr", "ridge", "stack"]
ALL_MODEL_CHOICES = [
    "xgb", "rf", "lstm", "lgbm", "catboost",
    "etr", "hgbr", "ridge", "stack"
]

# GPU
USE_GPU = False
USE_MIXED_PRECISION = False

# Optuna trials
N_TRIALS = {
    "xgb": 20,
    "rf": 20,
    "lstm": 10,
    "lgbm": 20,
    "catboost": 20,
    "etr": 20,
    "hgbr": 20,
    "ridge": 20,
    # "tcn": 5,
    "stack": 5,
}
N_SPLITS_CV = 6
RANDOM_STATE = 42

# Sequence models
DEFAULT_SEQ_LEN = 28
SEQ_LEN_CHOICES = [7, 14, 21, 30]
SEQ_EPOCHS = 40
SEQ_BATCH_CHOICES = [32, 64]

# Feature selection
RAW_CORR_THRESHOLD = 0.15
MAX_RAW_FEATURES = 15
MIN_HISTORY_DAYS = 30


# =========================
# ARGPARSE HELPERS
# =========================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Giá trị boolean không hợp lệ. Dùng true/false.")


def parse_date_arg(date_str: str, arg_name: str) -> str:
    dt = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"{arg_name} không hợp lệ: {date_str}. Dùng định dạng YYYY-MM-DD.")
    return dt.strftime("%Y-%m-%d")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train flood forecasting models and forecast future water levels."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_SELECTED_MODELS,
        help="Danh sách model muốn chạy. Ví dụ: --models xgb rf lstm hoặc --models all",
    )

    parser.add_argument(
        "--forecast-days",
        type=int,
        default=DEFAULT_FORECAST_DAYS,
        help="Số ngày dự đoán tương lai, bắt đầu từ ngày sau TEST_END_DATE",
    )

    parser.add_argument(
        "--train-end-date",
        type=str,
        default=DEFAULT_TRAIN_END_DATE,
        help="Ngày kết thúc train. Ví dụ: 2021-12-31",
    )

    parser.add_argument(
        "--test-end-date",
        type=str,
        default=DEFAULT_TEST_END_DATE,
        help="Ngày kết thúc test. Ví dụ: 2022-12-21",
    )

    parser.add_argument(
        "--save-pkl",
        type=str2bool,
        default=DEFAULT_SAVE_PKL,
        help="Có lưu file .pkl hay không. Dùng true/false. Mặc định false.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Đường dẫn file dữ liệu train/test",
    )

    parser.add_argument(
        "--validation-path",
        type=str,
        default=VALIDATION_PATH,
        help="Đường dẫn file validation (nếu có)",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=OUTPUT_FOLDER,
        help="Thư mục lưu kết quả",
    )

    args = parser.parse_args()

    raw_models = [m.lower().strip() for m in args.models]
    if "all" in raw_models:
        args.models = ALL_MODEL_CHOICES
    else:
        invalid = [m for m in raw_models if m not in ALL_MODEL_CHOICES]
        if invalid:
            raise ValueError(
                f"Model không hợp lệ: {invalid}. "
                f"Model hợp lệ gồm: {ALL_MODEL_CHOICES} hoặc 'all'."
            )
        args.models = raw_models

    args.train_end_date = parse_date_arg(args.train_end_date, "--train-end-date")
    args.test_end_date = parse_date_arg(args.test_end_date, "--test-end-date")

    if pd.to_datetime(args.test_end_date) <= pd.to_datetime(args.train_end_date):
        raise ValueError("--test-end-date phải lớn hơn --train-end-date.")

    if args.forecast_days <= 0:
        raise ValueError("--forecast-days phải > 0")

    return args


# =========================
# GPU / ENV SETUP
# =========================
def setup_environment() -> Tuple[bool, int]:
    gpu_count = 0
    if HAS_TF:
        try:
            gpus = tf.config.list_physical_devices("GPU")
            gpu_count = len(gpus)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if USE_GPU and USE_MIXED_PRECISION and gpu_count > 0:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
        except Exception as e:
            print(f"GPU setup warning: {e}")

    print(f"TensorFlow available: {HAS_TF}")
    print(f"Num GPUs Available: {gpu_count}")
    if USE_GPU and gpu_count > 0:
        print("GPU is enabled. LSTM/TCN will use TensorFlow GPU automatically if TensorFlow GPU is installed.")
    else:
        print("GPU is not active for TensorFlow. LSTM/TCN will run on CPU.")

    return gpu_count > 0, gpu_count


# =========================
# BASIC UTILITIES
# =========================
def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def make_safe_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "_")
    )


def parse_mixed_date(x: Any) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT


def collapse_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Ngày" not in df.columns:
        return df

    dup_count = int(df.duplicated(subset=["Ngày"]).sum())
    if dup_count == 0:
        return df.sort_values("Ngày").reset_index(drop=True)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Ngày"]
    other_cols = [c for c in df.columns if c not in numeric_cols + ["Ngày"]]

    agg_map = {c: "mean" for c in numeric_cols}
    for c in other_cols:
        agg_map[c] = lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan

    out = (
        df.groupby("Ngày", as_index=False)
        .agg(agg_map)
        .sort_values("Ngày")
        .reset_index(drop=True)
    )
    print(f"Found {dup_count} duplicate date rows. Collapsed to one row per day.")
    return out


def safe_hist_lookup(hist_daily_avg: Dict[int, float], day_of_year: int, fallback: float) -> float:
    val = hist_daily_avg.get(day_of_year, np.nan)
    return float(fallback if pd.isna(val) else val)


def choose_hour_col(df: pd.DataFrame, target_col: str) -> Optional[str]:
    target_lower = target_col.lower()
    station_tokens = []
    if "phú an" in target_lower or "phu an" in target_lower:
        station_tokens = ["phú an", "phu an"]
    elif "nhà bè" in target_lower or "nha be" in target_lower:
        station_tokens = ["nhà bè", "nha be"]

    for col in df.columns:
        lower = col.lower()
        if ("giờ" in lower or "hour" in lower or "xh" in lower) and any(tok in lower for tok in station_tokens):
            return col

    generic = [c for c in df.columns if ("giờ" in c.lower() or "hour" in c.lower() or "xh" in c.lower())]
    return generic[0] if generic else None


def make_station_hour_output_col(target_col: str) -> str:
    target_lower = target_col.lower()
    if "phú an" in target_lower or "phu an" in target_lower:
        return "H Phú An Giờ xh"
    if "nhà bè" in target_lower or "nha be" in target_lower:
        return "H Nhà Bè Giờ xh"
    return f"{target_col} Giờ xh"


def get_row_for_date(date_value: pd.Timestamp, *dfs: Optional[pd.DataFrame]) -> pd.Series:
    merged = {}
    for df in dfs:
        if df is None or df.empty or "Ngày" not in df.columns:
            continue
        rows = df[df["Ngày"] == date_value]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for col in row.index:
            if col not in merged or pd.isna(merged[col]):
                merged[col] = row[col]
    return pd.Series(merged) if merged else pd.Series(dtype=object)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    df = pd.read_csv(path)
    if "Ngày" not in df.columns:
        raise ValueError("File dữ liệu phải có cột 'Ngày'.")
    df["Ngày"] = df["Ngày"].apply(parse_mixed_date)
    df = df.dropna(subset=["Ngày"]).sort_values("Ngày").reset_index(drop=True)
    df = collapse_duplicate_dates(df)
    return df


def maybe_load_validation(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Ngày" not in df.columns:
        return None
    df["Ngày"] = df["Ngày"].apply(parse_mixed_date)
    df = df.dropna(subset=["Ngày"]).sort_values("Ngày").reset_index(drop=True)
    df = collapse_duplicate_dates(df)
    return df


# =========================
# METRICS
# =========================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "MSE (m²)": np.nan,
            "RMSE (m)": np.nan,
            "R²": np.nan,
            "AAE (m)": np.nan,
            "TAE (m)": np.nan,
            "TER (%)": np.nan,
            "NSE": np.nan,
            "RSR": np.nan,
        }

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    abs_err = np.abs(y_true - y_pred)
    aae = float(np.mean(abs_err))
    tae = float(np.sum(abs_err))
    total_actual = float(np.sum(np.abs(y_true)))
    ter = float(tae / total_actual * 100) if total_actual != 0 else np.nan

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0:
        nse = np.nan
        rsr = np.nan
    else:
        nse = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
        rsr = float(np.sqrt(np.sum((y_true - y_pred) ** 2)) / np.sqrt(denom))

    return {
        "MSE (m²)": float(mse),
        "RMSE (m)": rmse,
        "R²": r2,
        "AAE (m)": aae,
        "TAE (m)": tae,
        "TER (%)": ter,
        "NSE": nse,
        "RSR": rsr,
    }


# =========================
# FEATURE ENGINEERING
# =========================
def build_hist_daily_avg(df: pd.DataFrame, target_col: str) -> Dict[int, float]:
    temp = df[["Ngày", target_col]].dropna().copy()
    temp["doy"] = temp["Ngày"].dt.dayofyear
    grp = temp.groupby("doy")[target_col].mean()
    return {int(k): float(v) for k, v in grp.items()}


def select_raw_features(
    df: pd.DataFrame,
    target_col: str,
    train_end_date: str,
    corr_threshold: float = RAW_CORR_THRESHOLD,
    max_features: int = MAX_RAW_FEATURES,
) -> List[str]:
    train_df = df[df["Ngày"] <= pd.to_datetime(train_end_date)].copy()
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    banned_keywords = ["giờ", "hour", "xh"]
    exclude = {target_col, "year", "month", "day", "day_of_year", "sin_day", "cos_day", "sin_month", "cos_month"}

    candidate_cols = []
    for col in numeric_cols:
        if col in exclude:
            continue
        lower_col = col.lower()
        if any(k in lower_col for k in banned_keywords):
            continue
        if col.startswith("H ") and col != target_col:
            continue
        candidate_cols.append(col)

    if not candidate_cols:
        return []

    corr = train_df[candidate_cols + [target_col]].corr(numeric_only=True)[target_col].drop(target_col)
    corr = corr.abs().sort_values(ascending=False)
    selected = corr[corr >= corr_threshold].index.tolist()
    if not selected:
        selected = corr.head(min(max_features, len(corr))).index.tolist()

    return selected[:max_features]


def make_feature_vector(
    history_df: pd.DataFrame,
    current_date: pd.Timestamp,
    target_col: str,
    raw_feature_values: Dict[str, float],
    hist_daily_avg: Dict[int, float],
    raw_feature_cols: List[str],
) -> Optional[Dict[str, float]]:
    history_df = history_df.sort_values("Ngày").copy()
    history_series = history_df[target_col].dropna().astype(float)

    if len(history_series) < MIN_HISTORY_DAYS:
        return None

    fallback_hist = float(history_series.mean())
    day_of_year = int(current_date.dayofyear)
    hist_avg = safe_hist_lookup(hist_daily_avg, day_of_year, fallback_hist)

    safe_target = target_col.replace(" (m)", "")
    feat: Dict[str, float] = {}

    for col in raw_feature_cols:
        value = raw_feature_values.get(col, np.nan)
        if pd.isna(value):
            if col in history_df.columns and pd.api.types.is_numeric_dtype(history_df[col]):
                hist_col = history_df[col].dropna()
                value = float(hist_col.iloc[-1]) if not hist_col.empty else np.nan
        feat[col] = value

    for lag in [1, 2, 3, 7, 14, 30]:
        feat[f"{safe_target}_lag_{lag}"] = float(history_series.iloc[-lag]) if len(history_series) >= lag else hist_avg

    for window in [3, 7, 14, 30]:
        window_vals = history_series.iloc[-window:] if len(history_series) >= window else history_series
        feat[f"{safe_target}_rolling_mean_{window}"] = float(window_vals.mean())
        feat[f"{safe_target}_rolling_std_{window}"] = float(window_vals.std(ddof=1)) if len(window_vals) > 1 else 0.0
        feat[f"{safe_target}_rolling_max_{window}"] = float(window_vals.max())
        feat[f"{safe_target}_rolling_min_{window}"] = float(window_vals.min())

    feat[f"{safe_target}_diff_1"] = float(history_series.iloc[-1] - history_series.iloc[-2]) if len(history_series) >= 2 else 0.0
    feat[f"{safe_target}_diff_7"] = float(history_series.iloc[-1] - history_series.iloc[-7]) if len(history_series) >= 7 else 0.0

    recent_diff = history_series.diff().dropna().tail(30)
    feat[f"{safe_target}_trend_30d"] = float(recent_diff.mean()) if not recent_diff.empty else 0.0

    feat["year"] = int(current_date.year)
    feat["month"] = int(current_date.month)
    feat["day"] = int(current_date.day)
    feat["day_of_year"] = day_of_year
    feat["sin_day"] = float(np.sin(2 * np.pi * day_of_year / 365.25))
    feat["cos_day"] = float(np.cos(2 * np.pi * day_of_year / 365.25))
    feat["sin_month"] = float(np.sin(2 * np.pi * current_date.month / 12.0))
    feat["cos_month"] = float(np.cos(2 * np.pi * current_date.month / 12.0))

    return feat


def build_supervised_frame(df: pd.DataFrame, target_col: str, raw_feature_cols: List[str]) -> pd.DataFrame:
    df = df.sort_values("Ngày").reset_index(drop=True).copy()
    hist_daily_avg = build_hist_daily_avg(df, target_col)
    hour_col = choose_hour_col(df, target_col)

    rows = []
    for i in range(len(df)):
        history_df = df.iloc[:i].copy()
        current = df.iloc[i]
        raw_feature_values = {col: current[col] if col in current.index else np.nan for col in raw_feature_cols}
        feat = make_feature_vector(history_df, current["Ngày"], target_col, raw_feature_values, hist_daily_avg, raw_feature_cols)
        if feat is None:
            continue
        feat["Ngày"] = current["Ngày"]
        feat[target_col] = current[target_col]
        if hour_col and hour_col in current.index:
            feat["Giờ xuất hiện"] = current[hour_col]
        rows.append(feat)

    out = pd.DataFrame(rows)
    out = out.sort_values("Ngày").reset_index(drop=True)
    return out


# =========================
# SEQUENCE UTILITIES
# =========================
def make_sequences_from_arrays(X_values: np.ndarray, y_values: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_seq, y_seq, idx = [], [], []
    for i in range(seq_len, len(X_values)):
        X_seq.append(X_values[i - seq_len:i])
        y_seq.append(y_values[i])
        idx.append(i)

    if not X_seq:
        return np.empty((0, seq_len, X_values.shape[1]), dtype=np.float32), np.array([]), np.array([])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32), np.asarray(idx)


# =========================
# MODEL AVAILABILITY
# =========================
def validate_selected_models(models: List[str]) -> List[str]:
    valid = []
    for m in models:
        if m == "xgb" and not HAS_XGB:
            print("Skip xgb: chưa cài xgboost.")
            continue
        if m == "lgbm" and not HAS_LGBM:
            print("Skip lgbm: chưa cài lightgbm.")
            continue
        if m == "catboost" and not HAS_CATBOOST:
            print("Skip catboost: chưa cài catboost.")
            continue
        if m in {"lstm", "tcn"} and not HAS_TF:
            print(f"Skip {m}: chưa có TensorFlow.")
            continue
        valid.append(m)

    if not valid:
        raise ValueError("Không có model hợp lệ để chạy. Hãy cài thư viện còn thiếu hoặc đổi --models.")

    return valid


# =========================
# BASE PARAMS
# =========================
def get_xgb_base_params(gpu_available: bool) -> Dict[str, Any]:
    params = {
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if USE_GPU and gpu_available:
        params["device"] = "cuda"
    return params


def get_lgbm_base_params() -> Dict[str, Any]:
    return {
        "objective": "regression",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_catboost_base_params(gpu_available: bool) -> Dict[str, Any]:
    params = {
        "loss_function": "RMSE",
        "random_seed": RANDOM_STATE,
        "verbose": False,
    }
    if USE_GPU and gpu_available:
        params["task_type"] = "GPU"
    return params


def build_ridge_pipeline(alpha: float):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])


class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking cho chuỗi thời gian theo kiểu walk-forward.
    Không dùng sklearn StackingRegressor vì class đó gọi cross_val_predict bên trong,
    trong khi TimeSeriesSplit không tạo ra các partition đầy đủ.
    """

    def __init__(self, estimators, final_estimator=None, passthrough: bool = False, n_splits: int = 3):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.passthrough = passthrough
        self.n_splits = n_splits

    def _to_numpy(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, pd.Series):
            return X.to_frame().values
        return np.asarray(X)

    def _slice(self, X, idx):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.iloc[idx]
        return X[idx]

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        n_samples = len(y_arr)
        if n_samples < 8:
            raise ValueError("Không đủ dữ liệu để train stack cho chuỗi thời gian.")

        max_possible_splits = n_samples - 1
        n_splits = min(self.n_splits, max_possible_splits)
        if n_splits < 2:
            raise ValueError("Stack cần ít nhất 2 fold TimeSeriesSplit.")

        splitter = TimeSeriesSplit(n_splits=n_splits)
        n_estimators = len(self.estimators)
        oof_preds = np.full((n_samples, n_estimators), np.nan, dtype=float)

        for tr_idx, val_idx in splitter.split(np.arange(n_samples)):
            X_tr = self._slice(X, tr_idx)
            X_val = self._slice(X, val_idx)
            y_tr = y_arr[tr_idx]

            for j, (_, est) in enumerate(self.estimators):
                est_fold = clone(est)
                est_fold.fit(X_tr, y_tr)
                oof_preds[val_idx, j] = np.asarray(est_fold.predict(X_val), dtype=float)

        valid_mask = np.all(np.isfinite(oof_preds), axis=1)
        if valid_mask.sum() < max(10, n_estimators * 2):
            raise ValueError("Không đủ out-of-fold predictions hợp lệ để train meta-model cho stack.")

        meta_X = oof_preds[valid_mask]
        if self.passthrough:
            X_valid = self._to_numpy(self._slice(X, np.where(valid_mask)[0]))
            meta_X = np.hstack([meta_X, X_valid])

        self.final_estimator_ = clone(self.final_estimator if self.final_estimator is not None else Ridge(alpha=1.0))
        self.final_estimator_.fit(meta_X, y_arr[valid_mask])

        self.estimators_ = []
        for name, est in self.estimators:
            est_full = clone(est)
            est_full.fit(X, y_arr)
            self.estimators_.append((name, est_full))

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise ValueError("Model stack chưa được fit.")

        base_preds = []
        for _, est in self.estimators_:
            base_preds.append(np.asarray(est.predict(X), dtype=float).reshape(-1, 1))
        meta_X = np.hstack(base_preds)

        if self.passthrough:
            X_np = self._to_numpy(X)
            meta_X = np.hstack([meta_X, X_np])

        return self.final_estimator_.predict(meta_X)


def build_stack_estimators() -> List[Tuple[str, Any]]:
    return [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "etr",
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "hgbr",
            HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
        ),
        ("ridge", build_ridge_pipeline(alpha=1.0)),
    ]


def build_stacking_model(final_alpha: float = 1.0, passthrough: bool = False) -> TimeSeriesStackingRegressor:
    return TimeSeriesStackingRegressor(
        estimators=build_stack_estimators(),
        final_estimator=build_ridge_pipeline(alpha=final_alpha),
        passthrough=passthrough,
        n_splits=3,
    )


# =========================
# MODEL BUILDERS
# =========================
def build_lstm_model(seq_len: int, input_dim: int, units: int, dropout: float, learning_rate: float) -> Sequential:
    model = Sequential([
        Input(shape=(seq_len, input_dim)),
        LSTM(units),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_tcn_model(
    seq_len: int,
    input_dim: int,
    filters: int,
    kernel_size: int,
    dropout: float,
    learning_rate: float,
) -> Model:
    inputs = Input(shape=(seq_len, input_dim))
    x = inputs
    for d in [1, 2, 4]:
        res = x
        x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=d, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=d, activation="relu")(x)
        x = Dropout(dropout)(x)
        if int(res.shape[-1]) != filters:
            res = Conv1D(filters, 1, padding="same")(res)
        x = Add()([x, res])
        x = Activation("relu")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


# =========================
# OPTUNA OBJECTIVES
# =========================
def objective_xgb(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, gpu_available: bool) -> float:
    params = {
        **get_xgb_base_params(gpu_available),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_rf(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_etr(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = ExtraTreesRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_hgbr(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    params = {
        "max_iter": trial.suggest_int("max_iter", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 3.0),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
        "random_state": RANDOM_STATE,
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_ridge(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    params = {
        "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = build_ridge_pipeline(alpha=params["alpha"])
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_stack(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    params = {
        "final_alpha": trial.suggest_float("final_alpha", 1e-3, 100.0, log=True),
        "passthrough": trial.suggest_categorical("passthrough", [False, True]),
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = build_stacking_model(final_alpha=params["final_alpha"], passthrough=params["passthrough"])
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_lgbm(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    params = {
        **get_lgbm_base_params(),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_catboost(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, gpu_available: bool) -> float:
    params = {
        **get_catboost_base_params(gpu_available),
        "iterations": trial.suggest_int("iterations", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
    }
    scores = []
    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    for tr_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
    return float(np.mean(scores))


def objective_sequence(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> float:
    if model_name == "lstm":
        params = {
            "units": trial.suggest_int("units", 16, 64),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
            "seq_len": trial.suggest_categorical("seq_len", SEQ_LEN_CHOICES),
            "batch_size": trial.suggest_categorical("batch_size", SEQ_BATCH_CHOICES),
        }
    else:
        params = {
            "filters": trial.suggest_int("filters", 16, 64),
            "kernel_size": trial.suggest_int("kernel_size", 2, 5),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
            "seq_len": trial.suggest_categorical("seq_len", SEQ_LEN_CHOICES),
            "batch_size": trial.suggest_categorical("batch_size", SEQ_BATCH_CHOICES),
        }

    splitter = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    scores = []

    for tr_idx, val_idx in splitter.split(X_train):
        tf.keras.backend.clear_session()
        X_tr_df, X_val_df = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
        y_val = y_train.iloc[val_idx].reset_index(drop=True)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_df)
        X_val_scaled = scaler.transform(X_val_df)

        X_tr_seq, y_tr_seq, _ = make_sequences_from_arrays(X_tr_scaled, y_tr.values, params["seq_len"])
        X_val_seq, y_val_seq, _ = make_sequences_from_arrays(X_val_scaled, y_val.values, params["seq_len"])
        if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
            continue

        if model_name == "lstm":
            model = build_lstm_model(
                params["seq_len"], X_tr_seq.shape[2],
                params["units"], params["dropout"], params["learning_rate"]
            )
        else:
            model = build_tcn_model(
                params["seq_len"], X_tr_seq.shape[2],
                params["filters"], params["kernel_size"],
                params["dropout"], params["learning_rate"]
            )

        callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        model.fit(
            X_tr_seq,
            y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=SEQ_EPOCHS,
            batch_size=params["batch_size"],
            verbose=0,
            callbacks=callbacks,
        )
        pred = model.predict(X_val_seq, verbose=0).flatten()
        scores.append(mean_squared_error(y_val_seq, pred))

    return float(np.mean(scores)) if scores else float("inf")


# =========================
# FIT / PREDICT HELPERS
# =========================
def optimize_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, gpu_available: bool) -> Dict[str, Any]:
    study = optuna.create_study(direction="minimize")
    n_trials = N_TRIALS.get(model_name, 8)

    if model_name == "xgb":
        study.optimize(lambda t: objective_xgb(t, X_train, y_train, gpu_available), n_trials=n_trials)
    elif model_name == "rf":
        study.optimize(lambda t: objective_rf(t, X_train, y_train), n_trials=n_trials)
    elif model_name == "etr":
        study.optimize(lambda t: objective_etr(t, X_train, y_train), n_trials=n_trials)
    elif model_name == "hgbr":
        study.optimize(lambda t: objective_hgbr(t, X_train, y_train), n_trials=n_trials)
    elif model_name == "ridge":
        study.optimize(lambda t: objective_ridge(t, X_train, y_train), n_trials=n_trials)
    elif model_name == "stack":
        study.optimize(lambda t: objective_stack(t, X_train, y_train), n_trials=n_trials)
    elif model_name == "lgbm":
        study.optimize(lambda t: objective_lgbm(t, X_train, y_train), n_trials=n_trials)
    elif model_name == "catboost":
        study.optimize(lambda t: objective_catboost(t, X_train, y_train, gpu_available), n_trials=n_trials)
    elif model_name in {"lstm", "tcn"}:
        study.optimize(lambda t: objective_sequence(t, X_train, y_train, model_name), n_trials=n_trials)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Best {model_name} params: {study.best_params}")
    return study.best_params


def fit_tabular_model(model_name: str, best_params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, gpu_available: bool):
    if model_name == "xgb":
        params = {**get_xgb_base_params(gpu_available), **best_params}
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
    elif model_name == "rf":
        params = {**best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
    elif model_name == "etr":
        params = {**best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
        model = ExtraTreesRegressor(**params)
        model.fit(X_train, y_train)
    elif model_name == "hgbr":
        params = {**best_params, "random_state": RANDOM_STATE}
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
    elif model_name == "ridge":
        model = build_ridge_pipeline(alpha=float(best_params["alpha"]))
        model.fit(X_train, y_train)
    elif model_name == "stack":
        model = build_stacking_model(
            final_alpha=float(best_params["final_alpha"]),
            passthrough=bool(best_params["passthrough"]),
        )
        model.fit(X_train, y_train)
    elif model_name == "lgbm":
        params = {**get_lgbm_base_params(), **best_params}
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
    elif model_name == "catboost":
        params = {**get_catboost_base_params(gpu_available), **best_params}
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
    else:
        raise ValueError(f"Unsupported tabular model: {model_name}")

    return model


def fit_sequence_model(model_name: str, best_params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    tf.keras.backend.clear_session()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    seq_len = int(best_params.get("seq_len", DEFAULT_SEQ_LEN))
    X_seq, y_seq, idx = make_sequences_from_arrays(X_scaled, y_train.values, seq_len)
    if len(X_seq) == 0:
        raise ValueError(f"Không đủ dữ liệu để train {model_name} với seq_len={seq_len}")

    if model_name == "lstm":
        model = build_lstm_model(
            seq_len,
            X_seq.shape[2],
            int(best_params["units"]),
            float(best_params["dropout"]),
            float(best_params["learning_rate"]),
        )
    else:
        model = build_tcn_model(
            seq_len,
            X_seq.shape[2],
            int(best_params["filters"]),
            int(best_params["kernel_size"]),
            float(best_params["dropout"]),
            float(best_params["learning_rate"]),
        )

    callbacks = [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    model.fit(
        X_seq,
        y_seq,
        validation_split=0.1,
        epochs=SEQ_EPOCHS,
        batch_size=int(best_params.get("batch_size", 32)),
        verbose=0,
        callbacks=callbacks,
    )

    return {
        "model": model,
        "scaler": scaler,
        "seq_len": seq_len,
    }


def predict_on_test_tabular(model, frame: pd.DataFrame, target_col: str, feature_cols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    part = frame[(frame["Ngày"] >= pd.to_datetime(start_date)) & (frame["Ngày"] <= pd.to_datetime(end_date))].copy()
    hour_out_col = make_station_hour_output_col(target_col)
    if part.empty:
        return pd.DataFrame(columns=["Ngày", f"{target_col} Thực tế", f"{target_col} Dự đoán", hour_out_col])

    part[f"{target_col} Dự đoán"] = model.predict(part[feature_cols])
    out = pd.DataFrame({
        "Ngày": part["Ngày"].values,
        f"{target_col} Thực tế": part[target_col].values,
        f"{target_col} Dự đoán": part[f"{target_col} Dự đoán"].values,
        hour_out_col: part["Giờ xuất hiện"].values if "Giờ xuất hiện" in part.columns else np.nan,
    })
    return out


def predict_on_test_sequence(
    bundle: Dict[str, Any],
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    scaler = bundle["scaler"]
    model = bundle["model"]
    seq_len = bundle["seq_len"]
    hour_out_col = make_station_hour_output_col(target_col)

    X_scaled = scaler.transform(frame[feature_cols])
    X_seq, y_seq, idx = make_sequences_from_arrays(X_scaled, frame[target_col].values, seq_len)
    if len(X_seq) == 0:
        return pd.DataFrame(columns=["Ngày", f"{target_col} Thực tế", f"{target_col} Dự đoán", hour_out_col])

    pred = model.predict(X_seq, verbose=0).flatten()
    dates = frame["Ngày"].iloc[idx].reset_index(drop=True)
    actual = frame[target_col].iloc[idx].reset_index(drop=True)
    hours = frame["Giờ xuất hiện"].iloc[idx].reset_index(drop=True) if "Giờ xuất hiện" in frame.columns else pd.Series([np.nan] * len(idx))

    out = pd.DataFrame({
        "Ngày": dates,
        f"{target_col} Thực tế": actual,
        f"{target_col} Dự đoán": pred,
        hour_out_col: hours,
    })
    out = out[(out["Ngày"] >= pd.to_datetime(start_date)) & (out["Ngày"] <= pd.to_datetime(end_date))].reset_index(drop=True)
    return out


def recursive_future_forecast(
    model_name: str,
    fitted_obj: Any,
    history_df: pd.DataFrame,
    feature_history_df: pd.DataFrame,
    raw_data_df: pd.DataFrame,
    target_col: str,
    raw_feature_cols: List[str],
    feature_cols: List[str],
    future_dates: List[pd.Timestamp],
    future_lookup_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    hist_daily_avg = build_hist_daily_avg(history_df, target_col)
    hour_col = choose_hour_col(future_lookup_df, target_col) if future_lookup_df is not None else choose_hour_col(raw_data_df, target_col)
    hour_out_col = make_station_hour_output_col(target_col)
    preds = []

    history_df = history_df.sort_values("Ngày").reset_index(drop=True).copy()
    feature_history_df = feature_history_df.sort_values("Ngày").reset_index(drop=True).copy()

    for d in future_dates:
        base_row = get_row_for_date(d, future_lookup_df, raw_data_df)
        raw_feature_values = {col: base_row[col] if col in base_row.index else np.nan for col in raw_feature_cols}

        feat = make_feature_vector(history_df, d, target_col, raw_feature_values, hist_daily_avg, raw_feature_cols)
        if feat is None:
            continue

        feat_df = pd.DataFrame([feat])
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = np.nan
        feat_df = feat_df[feature_cols]
        feat_df = feat_df.ffill(axis=1).fillna(0.0)

        if model_name in {"xgb", "rf", "etr", "hgbr", "ridge", "stack", "lgbm", "catboost"}:
            pred = float(fitted_obj.predict(feat_df)[0])
        else:
            scaler = fitted_obj["scaler"]
            seq_len = fitted_obj["seq_len"]
            model = fitted_obj["model"]

            feat_row_to_append = pd.DataFrame([{**feat, "Ngày": d}])
            temp_feature_history = pd.concat([feature_history_df, feat_row_to_append], ignore_index=True)
            seq_source = temp_feature_history.tail(seq_len)[feature_cols]
            if len(seq_source) < seq_len:
                continue

            seq_scaled = scaler.transform(seq_source)
            X_input = seq_scaled[np.newaxis, :, :].astype(np.float32)
            pred = float(model.predict(X_input, verbose=0).flatten()[0])
            feature_history_df = temp_feature_history.copy()

        hour_value = base_row[hour_col] if (hour_col and hour_col in base_row.index) else np.nan
        preds.append({
            "Ngày": d,
            f"{target_col} Thực tế": np.nan,
            f"{target_col} Dự đoán": pred,
            hour_out_col: hour_value,
        })

        new_hist_row = {"Ngày": d, target_col: pred}
        for col in raw_feature_cols:
            new_hist_row[col] = raw_feature_values.get(col, np.nan)
        if hour_col:
            new_hist_row[hour_col] = hour_value
        history_df = pd.concat([history_df, pd.DataFrame([new_hist_row])], ignore_index=True)

    return pd.DataFrame(preds)


# =========================
# TRAIN / PREDICT PER TARGET
# =========================
def train_one_target(
    data: pd.DataFrame,
    validation_df: Optional[pd.DataFrame],
    target_col: str,
    selected_models: List[str],
    gpu_available: bool,
    forecast_days: int,
    train_end_date: str,
    test_end_date: str,
    output_folder: str,
    save_pkl: bool,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"Running target: {target_col}")
    print("=" * 70)

    raw_feature_cols = select_raw_features(data, target_col, train_end_date)
    print(f"Selected raw features for {target_col}: {raw_feature_cols}")

    frame = build_supervised_frame(data, target_col, raw_feature_cols)
    frame = frame.dropna(subset=[target_col]).sort_values("Ngày").reset_index(drop=True)

    train_frame = frame[frame["Ngày"] <= pd.to_datetime(train_end_date)].copy()
    test_frame = frame[(frame["Ngày"] > pd.to_datetime(train_end_date)) & (frame["Ngày"] <= pd.to_datetime(test_end_date))].copy()

    if train_frame.empty or test_frame.empty:
        raise ValueError(f"Không đủ train/test data cho {target_col}")

    non_feature_cols = {"Ngày", target_col, "Giờ xuất hiện"}
    feature_cols = [c for c in frame.columns if c not in non_feature_cols]
    feature_cols = unique_preserve_order(feature_cols)

    train_frame[feature_cols] = train_frame[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    test_frame[feature_cols] = test_frame[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    X_train, y_train = train_frame[feature_cols], train_frame[target_col]
    print(f"Train rows: {len(train_frame)} | Test rows: {len(test_frame)}")

    all_results = {
        "test_dfs": {},
        "predict_dfs": {},
        "metrics": {},
        "validation_metrics": {},
        "validation_compare_dfs": {},
        "best_params": {},
        "feature_cols": feature_cols,
        "raw_feature_cols": raw_feature_cols,
    }

    future_start_date = pd.to_datetime(test_end_date) + pd.Timedelta(days=1)
    test_start_date = pd.to_datetime(train_end_date) + pd.Timedelta(days=1)
    plot_context_start = test_start_date - pd.Timedelta(days=30)
    future_dates = list(pd.date_range(start=future_start_date, periods=forecast_days, freq="D"))
    history_until_test = data[data["Ngày"] <= pd.to_datetime(test_end_date)].copy()
    future_lookup_df = validation_df.copy() if validation_df is not None else None

    for model_name in selected_models:
        print(f"\n--- Model: {model_name} ---")
        start_t = time.time()

        best_params = optimize_model(model_name, X_train, y_train, gpu_available)
        all_results["best_params"][model_name] = best_params

        if model_name in {"xgb", "rf", "etr", "hgbr", "ridge", "stack", "lgbm", "catboost"}:
            fitted = fit_tabular_model(model_name, best_params, X_train, y_train, gpu_available)
            test_df = predict_on_test_tabular(
                fitted,
                frame,
                target_col,
                feature_cols,
                str(pd.to_datetime(train_end_date) + pd.Timedelta(days=1)),
                test_end_date,
            )
        else:
            fitted = fit_sequence_model(model_name, best_params, X_train, y_train)
            test_df = predict_on_test_sequence(
                fitted,
                frame,
                target_col,
                feature_cols,
                str(pd.to_datetime(train_end_date) + pd.Timedelta(days=1)),
                test_end_date,
            )

        metrics = calculate_metrics(test_df[f"{target_col} Thực tế"].values, test_df[f"{target_col} Dự đoán"].values)
        print(f"{model_name} test RMSE: {metrics['RMSE (m)']:.4f} | NSE: {metrics['NSE']:.4f} | RSR: {metrics['RSR']:.4f}")

        predict_df = recursive_future_forecast(
            model_name=model_name,
            fitted_obj=fitted,
            history_df=history_until_test,
            feature_history_df=frame[frame["Ngày"] <= pd.to_datetime(test_end_date)].copy(),
            raw_data_df=data,
            target_col=target_col,
            raw_feature_cols=raw_feature_cols,
            feature_cols=feature_cols,
            future_dates=future_dates,
            future_lookup_df=future_lookup_df,
        )

        if validation_df is not None and target_col in validation_df.columns and not predict_df.empty:
            actual_col = f"{target_col} Validation thực tế"
            val_temp = validation_df[["Ngày", target_col]].copy().rename(columns={target_col: actual_col})

            val_eval = predict_df.merge(val_temp, on="Ngày", how="left")
            val_eval["Sai số"] = val_eval[f"{target_col} Dự đoán"] - val_eval[actual_col]
            val_eval["Sai số tuyệt đối"] = np.abs(val_eval["Sai số"])

            y_true_val = val_eval[actual_col].values
            y_pred_val = val_eval[f"{target_col} Dự đoán"].values
            mask = np.isfinite(y_true_val) & np.isfinite(y_pred_val)

            if mask.sum() > 0:
                all_results["validation_metrics"][model_name] = calculate_metrics(y_true_val[mask], y_pred_val[mask])
                print(f"{model_name} validation matched rows: {int(mask.sum())}")

            all_results["validation_compare_dfs"][model_name] = val_eval.copy()

        all_results["test_dfs"][model_name] = test_df
        all_results["predict_dfs"][model_name] = predict_df
        all_results["metrics"][model_name] = metrics

        safe_target = make_safe_name(target_col)
        model_path = os.path.join(output_folder, f"model_{safe_target}_{model_name}")

        if model_name in {"xgb", "rf", "etr", "hgbr", "ridge", "stack", "lgbm", "catboost"}:
            if save_pkl:
                joblib.dump(fitted, model_path + ".pkl")
        else:
            # sequence model vẫn lưu .keras
            fitted["model"].save(model_path + ".keras")
            if save_pkl:
                joblib.dump(
                    {
                        "scaler": fitted["scaler"],
                        "seq_len": fitted["seq_len"],
                        "feature_cols": feature_cols,
                    },
                    model_path + "_bundle.pkl",
                )

        plt.figure(figsize=(14, 6))
        actual_plot_df = data[
            (data["Ngày"] >= plot_context_start) & (data["Ngày"] <= pd.to_datetime(test_end_date))
        ][["Ngày", target_col]].dropna().copy()
        if not actual_plot_df.empty:
            plt.plot(actual_plot_df["Ngày"], actual_plot_df[target_col], label="Actual (1 month + test)")
        if not test_df.empty:
            plt.plot(test_df["Ngày"], test_df[f"{target_col} Dự đoán"], label=f"{model_name} test pred")
        if not predict_df.empty:
            plt.plot(predict_df["Ngày"], predict_df[f"{target_col} Dự đoán"], linestyle="--", label=f"{model_name} future pred")
        plt.axvline(test_start_date, linestyle="--")
        plt.axvline(future_start_date, linestyle=":")
        plt.title(f"{target_col} - {model_name}")
        plt.xlabel("Ngày")
        plt.ylabel("Water level (m)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"plot_{safe_target}_{model_name}.png"))
        plt.close()

        print(f"Finished {model_name} in {time.time() - start_t:.2f}s")

    return all_results


# =========================
# SAVE OUTPUT HELPERS
# =========================
def build_model_prediction_export(results: Dict[str, Dict], model_name: str) -> pd.DataFrame:
    frames = []
    for target in TARGETS:
        if target not in results or model_name not in results[target]["predict_dfs"]:
            continue
        df_pred = results[target]["predict_dfs"][model_name].copy()
        if df_pred is None or df_pred.empty:
            continue

        hour_col = make_station_hour_output_col(target)
        keep_cols = ["Ngày", f"{target} Dự đoán"]
        if hour_col in df_pred.columns:
            keep_cols.append(hour_col)
        frames.append(df_pred[keep_cols])

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for nxt in frames[1:]:
        merged = merged.merge(nxt, on="Ngày", how="outer")

    return merged.sort_values("Ngày").reset_index(drop=True)


def summarize_and_select_best_model(results: Dict[str, Dict], selected_models: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    rows = []
    summary_rows = []

    for model_name in selected_models:
        per_target_nse = []
        nse_sources = []

        for target in TARGETS:
            if target not in results:
                continue

            metric_source = None
            nse_value = np.nan

            if model_name in results[target].get("validation_metrics", {}):
                nse_value = results[target]["validation_metrics"][model_name].get("NSE", np.nan)
                metric_source = "validation"
            elif model_name in results[target].get("metrics", {}):
                nse_value = results[target]["metrics"][model_name].get("NSE", np.nan)
                metric_source = "test"

            rows.append({
                "Mô hình": model_name,
                "Mục tiêu": target,
                "NSE": nse_value,
                "Nguồn NSE": metric_source,
            })

            if pd.notna(nse_value):
                per_target_nse.append(float(nse_value))
                nse_sources.append(metric_source)

        mean_nse = float(np.mean(per_target_nse)) if per_target_nse else np.nan
        summary_rows.append({
            "Mô hình": model_name,
            "NSE trung bình": mean_nse,
            "Số target có NSE": len(per_target_nse),
            "Nguồn dùng xếp hạng": ", ".join(sorted(set([s for s in nse_sources if s]))) if nse_sources else np.nan,
        })

    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["NSE trung bình", "Mô hình"], ascending=[False, True]).reset_index(drop=True)

    best_model = None
    if not summary_df.empty and pd.notna(summary_df.loc[0, "NSE trung bình"]):
        best_model = str(summary_df.loc[0, "Mô hình"])

    if detail_df.empty:
        return summary_df, best_model

    out_df = pd.concat([
        summary_df.assign(**{"Mục tiêu": "__SUMMARY__", "NSE": summary_df["NSE trung bình"], "Nguồn NSE": summary_df["Nguồn dùng xếp hạng"]})[
            ["Mô hình", "Mục tiêu", "NSE", "Nguồn NSE", "Số target có NSE"]
        ],
        detail_df.assign(**{"Số target có NSE": np.nan})[["Mô hình", "Mục tiêu", "NSE", "Nguồn NSE", "Số target có NSE"]],
    ], ignore_index=True)
    return out_df, best_model


# =========================
# SAVE OUTPUTS
# =========================
def save_all_outputs(
    data: pd.DataFrame,
    results: Dict[str, Dict],
    selected_models: List[str],
    output_folder: str,
) -> None:
    metrics_file = os.path.join(output_folder, "model_metrics.csv")
    validation_metrics_file = os.path.join(output_folder, "validation_metrics.csv")
    combined_file = os.path.join(output_folder, "combined_flood_data.csv")
    best_predict_file = os.path.join(output_folder, "best_predict.csv")
    best_model_summary_file = os.path.join(output_folder, "best_model_by_nse.csv")

    metrics_rows = []
    validation_rows = []
    saved_validation_compare_files = []

    for target, res in results.items():
        for model_name, m in res["metrics"].items():
            metrics_rows.append({"Mục tiêu": target, "Mô hình": model_name, **m})

        for model_name, m in res["validation_metrics"].items():
            validation_rows.append({"Mục tiêu": target, "Mô hình": model_name, **m})

    pd.DataFrame(metrics_rows).to_csv(metrics_file, index=False, encoding="utf-8-sig")

    if validation_rows:
        pd.DataFrame(validation_rows).to_csv(validation_metrics_file, index=False, encoding="utf-8-sig")

    for model_name in selected_models:
        out_path = os.path.join(output_folder, f"{model_name}_predictions.csv")
        merged = build_model_prediction_export(results, model_name)
        if not merged.empty:
            merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    combined_df = data.copy()
    combined_df["Ngày"] = pd.to_datetime(combined_df["Ngày"], errors="coerce")
    combined_df = combined_df.dropna(subset=["Ngày"]).sort_values("Ngày").reset_index(drop=True)

    for target in TARGETS:
        if target not in results:
            continue

        for model_name in selected_models:
            if model_name not in results[target]["test_dfs"]:
                continue

            pred_col = f"{target}_Dự đoán_{model_name.upper()}"
            combined_df[pred_col] = np.nan

            test_df = results[target]["test_dfs"][model_name]
            fut_df = results[target]["predict_dfs"][model_name]

            for _, row in pd.concat([test_df, fut_df], ignore_index=True).iterrows():
                mask = combined_df["Ngày"] == row["Ngày"]
                if mask.any():
                    combined_df.loc[mask, pred_col] = row[f"{target} Dự đoán"]
                else:
                    combined_df = pd.concat(
                        [combined_df, pd.DataFrame([{"Ngày": row["Ngày"], pred_col: row[f"{target} Dự đoán"]}])],
                        ignore_index=True,
                    )

    combined_df.sort_values("Ngày").to_csv(combined_file, index=False, encoding="utf-8-sig")

    for target, res in results.items():
        safe_target = make_safe_name(target)
        for model_name, df_cmp in res.get("validation_compare_dfs", {}).items():
            if df_cmp is not None and not df_cmp.empty:
                cmp_path = os.path.join(output_folder, f"validation_compare_{safe_target}_{model_name}.csv")
                df_cmp.sort_values("Ngày").to_csv(cmp_path, index=False, encoding="utf-8-sig")
                saved_validation_compare_files.append(cmp_path)

    ranking_df, best_model = summarize_and_select_best_model(results, selected_models)
    if not ranking_df.empty:
        ranking_df.to_csv(best_model_summary_file, index=False, encoding="utf-8-sig")

    if best_model is not None:
        best_df = build_model_prediction_export(results, best_model)
        if not best_df.empty:
            best_df.to_csv(best_predict_file, index=False, encoding="utf-8-sig")
        print(f"Best model by NSE: {best_model}")

    print("\nSaved files:")
    print(f"- {metrics_file}")
    if validation_rows:
        print(f"- {validation_metrics_file}")

    for model_name in selected_models:
        out_path = os.path.join(output_folder, f"{model_name}_predictions.csv")
        if os.path.exists(out_path):
            print(f"- {out_path}")

    for cmp_path in saved_validation_compare_files:
        print(f"- {cmp_path}")

    if os.path.exists(best_model_summary_file):
        print(f"- {best_model_summary_file}")
    if os.path.exists(best_predict_file):
        print(f"- {best_predict_file}")

    print(f"- {combined_file}")


# =========================
# MAIN
# =========================
def main() -> None:
    args = parse_args()

    output_folder = args.output_folder
    ensure_output_dir(output_folder)

    selected_models = validate_selected_models(args.models)
    gpu_available, _ = setup_environment()

    print("Loading data...")
    data = load_data(args.data_path)
    validation_df = maybe_load_validation(args.validation_path)

    future_start_date = pd.to_datetime(args.test_end_date) + pd.Timedelta(days=1)
    future_end_date = future_start_date + pd.Timedelta(days=args.forecast_days - 1)

    print(f"Rows in data: {len(data)}")
    print(f"Date range: {data['Ngày'].min().date()} -> {data['Ngày'].max().date()}")
    print(f"Train end: {args.train_end_date}")
    print(f"Test: {(pd.to_datetime(args.train_end_date) + pd.Timedelta(days=1)).date()} -> {pd.to_datetime(args.test_end_date).date()}")
    print(f"Future: {future_start_date.date()} -> {future_end_date.date()} ({args.forecast_days} days)")
    print(f"Selected models: {selected_models}")
    print(f"Output folder: {output_folder}")
    print(f"Validation file loaded: {validation_df is not None}")
    print(f"Save PKL: {args.save_pkl}")

    if future_start_date <= pd.to_datetime(args.test_end_date):
        raise ValueError("Khoảng forecast bị lỗi: future_start_date phải sau test_end_date.")

    results: Dict[str, Dict] = {}
    for target in TARGETS:
        if target not in data.columns:
            print(f"Skip {target}: không có trong dữ liệu.")
            continue

        try:
            results[target] = train_one_target(
                data=data,
                validation_df=validation_df,
                target_col=target,
                selected_models=selected_models,
                gpu_available=gpu_available,
                forecast_days=args.forecast_days,
                train_end_date=args.train_end_date,
                test_end_date=args.test_end_date,
                output_folder=output_folder,
                save_pkl=args.save_pkl,
            )
        except Exception as e:
            print(f"Error processing {target}: {e}")

    if results:
        save_all_outputs(
            data=data,
            results=results,
            selected_models=selected_models,
            output_folder=output_folder,
        )
    else:
        print("Không có kết quả nào để lưu.")


if __name__ == "__main__":
    main()
