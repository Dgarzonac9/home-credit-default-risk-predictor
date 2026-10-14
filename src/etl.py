"""ETL para Home Credit Default Risk.

Produce una tabla con una fila por solicitud (SK_ID_CURR), lista para modelar.
Uso:
    from src.etl import run_etl
    train, test = run_etl()
"""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "processed"

DAYS_SENTINEL = 365243  # valor centinela en DAYS_* (significa "sin dato")


# ----------------------------------------------------------------- utilidades
def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memoria: float64 -> float32, int64 -> int32 cuando cabe."""
    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64"):
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def read(name: str, **kw) -> pd.DataFrame:
    return downcast(pd.read_csv(DATA / f"{name}.csv", **kw))


def one_hot(df: pd.DataFrame, prefix: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One-hot de las columnas object; devuelve (df, nuevas_columnas)."""
    cats = df.select_dtypes(["object", "category"]).columns.tolist()
    before = set(df.columns)
    df = pd.get_dummies(df, columns=cats, dummy_na=True, dtype="int8", prefix=prefix and [prefix] * len(cats) or None)
    return df, [c for c in df.columns if c not in before]


def aggregate(df: pd.DataFrame, key: str, num_aggs: dict, prefix: str,
              cat_cols: list[str] | None = None) -> pd.DataFrame:
    """groupby(key) con agregaciones numericas + media de las dummies."""
    aggs = dict(num_aggs)
    for c in cat_cols or []:
        aggs[c] = ["mean"]
    out = df.groupby(key).agg(aggs)
    out.columns = [f"{prefix}_{c}_{f}".upper() for c, f in out.columns]
    out[f"{prefix}_COUNT".upper()] = df.groupby(key).size()
    return out


# ------------------------------------------------------------------ application
def build_application() -> pd.DataFrame:
    train = read("application_train")
    test = read("application_test")
    train["IS_TRAIN"] = 1
    test["IS_TRAIN"] = 0
    df = pd.concat([train, test], ignore_index=True)
    del train, test

    # limpieza de centinelas / categorias invalidas
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_SENTINEL).astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_SENTINEL, np.nan)
    for c in ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE"]:
        df[c] = df[c].replace("XNA", np.nan)
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("Unknown", np.nan)

    # binarias Y/N -> 0/1
    for c in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[c] = (df[c] == "Y").astype("int8")

    # features derivadas
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_MEAN"] = ext.mean(axis=1)
    df["EXT_MIN"] = ext.min(axis=1)
    df["EXT_PROD"] = ext.prod(axis=1, min_count=3)
    df["EXT_WEIGHTED"] = (df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 3 + df["EXT_SOURCE_3"] * 4)
    docs = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCS_PROVIDED"] = df[docs].sum(axis=1).astype("int8")

    df = df.replace([np.inf, -np.inf], np.nan)
    df, _ = one_hot(df)
    return downcast(df.set_index("SK_ID_CURR"))


# ----------------------------------------------------------------------- bureau
def build_bureau() -> pd.DataFrame:
    bb = read("bureau_balance")
    bb, bb_cats = one_hot(bb)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        {"MONTHS_BALANCE": ["min", "max", "size"], **{c: ["mean"] for c in bb_cats}}
    )
    bb_agg.columns = [f"BB_{c}_{f}".upper() for c, f in bb_agg.columns]
    del bb
    gc.collect()

    b = read("bureau")
    b, b_cats = one_hot(b)
    b = b.join(bb_agg, on="SK_ID_BUREAU")
    del bb_agg
    b["CREDIT_DEBT_RATIO"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"].replace(0, np.nan)

    num = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max"],
        "AMT_CREDIT_MAX_OVERDUE": ["max"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],
        "CREDIT_DEBT_RATIO": ["mean", "max"],
        "AMT_ANNUITY": ["max", "mean"],
        "BB_MONTHS_BALANCE_MIN": ["min"],
        "BB_MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    bb_cols = [c for c in b.columns if c.startswith("BB_") and c.endswith("_MEAN")]
    out = aggregate(b, "SK_ID_CURR", num, "BUREAU", b_cats + bb_cols)

    # solo creditos activos
    active = b[b["CREDIT_ACTIVE_Active"] == 1]
    act = active.groupby("SK_ID_CURR").agg(
        {"AMT_CREDIT_SUM": ["sum", "mean"], "AMT_CREDIT_SUM_DEBT": ["sum"], "DAYS_CREDIT": ["max"]}
    )
    act.columns = [f"BUREAU_ACTIVE_{c}_{f}".upper() for c, f in act.columns]
    return downcast(out.join(act))


# ---------------------------------------------------------- previous application
def build_previous() -> pd.DataFrame:
    p = read("previous_application")
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        p[c] = p[c].replace(DAYS_SENTINEL, np.nan)
    p["APP_CREDIT_PERC"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"].replace(0, np.nan)
    p, p_cats = one_hot(p)

    num = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean"],
        "AMT_DOWN_PAYMENT": ["max", "mean"],
        "AMT_GOODS_PRICE": ["max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["max", "mean"],
    }
    out = aggregate(p, "SK_ID_CURR", num, "PREV", p_cats)

    for tag, col in [("APPROVED", "NAME_CONTRACT_STATUS_Approved"),
                     ("REFUSED", "NAME_CONTRACT_STATUS_Refused")]:
        sub = p[p[col] == 1].groupby("SK_ID_CURR").agg(
            {"AMT_ANNUITY": ["mean", "max"], "AMT_CREDIT": ["mean", "sum"], "DAYS_DECISION": ["max"]}
        )
        sub.columns = [f"PREV_{tag}_{c}_{f}".upper() for c, f in sub.columns]
        out = out.join(sub)
    return downcast(out)


# ------------------------------------------------------ POS / tarjeta / cuotas
def build_pos() -> pd.DataFrame:
    pos = read("POS_CASH_balance")
    pos, cats = one_hot(pos)
    pos["LATE"] = (pos["SK_DPD"] > 0).astype("int8")
    num = {
        "MONTHS_BALANCE": ["min", "max", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "LATE": ["mean", "sum"],
        "CNT_INSTALMENT_FUTURE": ["mean", "min"],
    }
    out = aggregate(pos, "SK_ID_CURR", num, "POS", cats)
    out["POS_LOANS"] = pos.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
    return downcast(out)


def build_credit_card() -> pd.DataFrame:
    cc = read("credit_card_balance").drop(columns="SK_ID_PREV")
    cc, cats = one_hot(cc)
    cc["LIMIT_USE"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc["PAYMENT_MIN_RATIO"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / cc["AMT_INST_MIN_REGULARITY"].replace(0, np.nan)
    num = {
        "MONTHS_BALANCE": ["min", "size"],
        "AMT_BALANCE": ["max", "mean"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["max", "mean"],
        "AMT_DRAWINGS_ATM_CURRENT": ["sum", "mean"],
        "AMT_DRAWINGS_CURRENT": ["sum", "mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["sum", "mean"],
        "LIMIT_USE": ["mean", "max"],
        "PAYMENT_MIN_RATIO": ["mean", "min"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "CNT_DRAWINGS_CURRENT": ["mean", "sum"],
    }
    cc = cc.replace([np.inf, -np.inf], np.nan)
    return downcast(aggregate(cc, "SK_ID_CURR", num, "CC", cats))


def build_installments() -> pd.DataFrame:
    ins = read("installments_payments")
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ins["LATE"] = (ins["DPD"] > 0).astype("int8")
    ins["UNDERPAID"] = (ins["PAYMENT_DIFF"] > 0).astype("int8")
    ins = ins.replace([np.inf, -np.inf], np.nan)
    num = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "LATE": ["mean", "sum"],
        "UNDERPAID": ["mean"],
        "PAYMENT_PERC": ["mean", "min", "std"],
        "PAYMENT_DIFF": ["mean", "max", "sum"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean"],
    }
    out = aggregate(ins, "SK_ID_CURR", num, "INS")
    # comportamiento reciente (ultimos 12 meses)
    rec = ins[ins["DAYS_INSTALMENT"] > -365].groupby("SK_ID_CURR").agg(
        {"DPD": ["max", "mean"], "LATE": ["mean"], "PAYMENT_PERC": ["mean"]}
    )
    rec.columns = [f"INS_L12M_{c}_{f}".upper() for c, f in rec.columns]
    return downcast(out.join(rec))


# --------------------------------------------------------------------- pipeline
def run_etl(drop_nan_thresh: float = 0.75, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta todo el ETL. Descarta columnas con mas de `drop_nan_thresh` de nulos."""
    df = build_application()
    for name, fn in [("bureau", build_bureau), ("previous", build_previous), ("pos", build_pos),
                     ("credit_card", build_credit_card), ("installments", build_installments)]:
        print(f"[ETL] {name}...")
        df = df.join(fn())
        gc.collect()

    # columnas casi vacias
    nan_rate = df.isna().mean()
    drop = nan_rate[nan_rate > drop_nan_thresh].index.difference(["TARGET"])
    df = df.drop(columns=drop)
    print(f"[ETL] columnas descartadas por nulos > {drop_nan_thresh:.0%}: {len(drop)}")

    train = df[df["IS_TRAIN"] == 1].drop(columns="IS_TRAIN")
    test = df[df["IS_TRAIN"] == 0].drop(columns=["IS_TRAIN", "TARGET"])
    train["TARGET"] = train["TARGET"].astype("int8")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        train.to_parquet(OUT / "train.parquet")
        test.to_parquet(OUT / "test.parquet")
        (OUT / "feature_names.json").write_text(
            json.dumps([c for c in train.columns if c != "TARGET"], indent=1), encoding="utf-8")
        print(f"[ETL] guardado en {OUT}")
    return train, test
