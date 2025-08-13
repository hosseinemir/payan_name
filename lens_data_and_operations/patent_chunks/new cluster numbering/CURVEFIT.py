import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ---- Logistic (as in the paper) ----
# f(t) = k / (1 + exp(-(t - a) / b))
def logistic(t, k, a, b):
    return k / (1 + np.exp(-(t - a) / b))

def year_at_pct(k, a, b, pct):
    """Year when f(t) reaches pct of K. Returns float (calendar year)."""
    if not np.all(np.isfinite([k, a, b])) or k <= 0 or b == 0 or not (0 < pct < 1):
        return np.nan
    try:
        return a + b * (-np.log((1.0/pct) - 1.0))
    except Exception:
        return np.nan

INPUT_FILE  = "sigma_input_cumulative_no2025.csv"
OUT_VALID   = "cluster_lifecycle_fit_VALID.csv"
OUT_REJECT  = "cluster_lifecycle_fit_REJECTED.csv"

# ---- Load & sanitize ----
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

need = {"Cluster ID","Cluster Name","Year","Cumulative Count"}
if not need.issubset(df.columns):
    raise ValueError(f"Missing columns. Have {list(df.columns)}, need {list(need)}")

df = df.rename(columns={
    "Cluster ID":"cluster_id",
    "Cluster Name":"cluster_name",
    "Year":"year",
    "Cumulative Count":"cum"
})

df = df.dropna(subset=["cluster_id","cluster_name","year","cum"]).copy()
df["year"] = df["year"].astype(int)
df["cum"]  = df["cum"].astype(float)

valid_rows, rejected_rows = [], []

for (cid, cname), g in df.groupby(["cluster_id","cluster_name"]):
    g = g.sort_values("year").drop_duplicates("year", keep="last").copy()
    # enforce non-decreasing cumulative (safety)
    g["cum"] = g["cum"].cummax()

    years = g["year"].to_numpy(dtype=float)
    y     = g["cum"].to_numpy(dtype=float)

    out_base = {
        "cluster_id": cid,
        "cluster_name": cname,
        "first_year": int(years.min()) if len(years)>0 else np.nan,
        "last_year":  int(years.max()) if len(years)>0 else np.nan,
        "n_years":    int(len(years))
    }

    # basic eligibility
    if len(years) < 4 or y[-1] <= 0:
        rejected_rows.append({**out_base,
                              "emerging_year": np.nan,
                              "growth_year": np.nan,
                              "maturity_year": np.nan,
                              "saturation_year": np.nan,
                              "k": np.nan, "a": np.nan, "b": np.nan,
                              "notes": "insufficient_data"})
        continue

    # ---- Fit logistic (bounded) ----
    try:
        k0 = max(y) * 1.05
        # rough a0 ~ year of fastest observed increment
        dy = np.diff(y, prepend=y[0])
        a0 = years[np.argmax(dy)]
        b0 = 1.0

        lower = [max(y), years.min()-5, 0.01]          # k >= max(y), a in [min-5, max+5], b>0
        upper = [max(y)*10, years.max()+5, 10.0]

        (k_fit, a_fit, b_fit), _ = curve_fit(
            logistic, years, y,
            p0=[k0, a0, b0],
            bounds=(lower, upper),
            maxfev=20000
        )

        # stage thresholds per paper
        emerg = year_at_pct(k_fit, a_fit, b_fit, 0.10)
        grow  = year_at_pct(k_fit, a_fit, b_fit, 0.50)
        matur = year_at_pct(k_fit, a_fit, b_fit, 0.90)
        satur = (matur + 5) if np.isfinite(matur) else np.nan

        # round to integer calendar years
        def rnd(x): return int(round(x)) if np.isfinite(x) else np.nan
        emerg_r, grow_r, matur_r, satur_r = map(rnd, [emerg, grow, matur, satur])

        # --- Monotonicity & sanity checks ---
        notes = []
        # All must be finite integers
        if any([pd.isna(v) for v in [emerg_r, grow_r, matur_r, satur_r]]):
            notes.append("nan_stage_year")
        # strict increasing
        if not notes and not (emerg_r < grow_r < matur_r < satur_r):
            notes.append("non_monotonic_stage_years")
        # saturation defined as maturity+5 already ensures last > prev, but keep check above.

        if notes:
            rejected_rows.append({**out_base,
                                  "emerging_year": emerg_r,
                                  "growth_year": grow_r,
                                  "maturity_year": matur_r,
                                  "saturation_year": satur_r,
                                  "k": float(k_fit), "a": float(a_fit), "b": float(b_fit),
                                  "notes": ";".join(notes)})
            continue

        # Optional fit quality (R^2)
        yhat = logistic(years, k_fit, a_fit, b_fit)
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean((y - yhat)**2)))

        valid_rows.append({**out_base,
                           "emerging_year": emerg_r,
                           "growth_year": grow_r,
                           "maturity_year": matur_r,
                           "saturation_year": satur_r,
                           "k": float(k_fit), "a": float(a_fit), "b": float(b_fit),
                           "R2": float(r2) if np.isfinite(r2) else np.nan,
                           "RMSE": float(rmse) if np.isfinite(rmse) else np.nan,
                           "notes": ""})

    except Exception as e:
        rejected_rows.append({**out_base,
                              "emerging_year": np.nan,
                              "growth_year": np.nan,
                              "maturity_year": np.nan,
                              "saturation_year": np.nan,
                              "k": np.nan, "a": np.nan, "b": np.nan,
                              "notes": f"fit_failed:{type(e).__name__}"})

# ---- Save outputs ----
pd.DataFrame(valid_rows).sort_values(["cluster_id","cluster_name"]).to_csv(OUT_VALID, index=False, encoding="utf-8-sig")
pd.DataFrame(rejected_rows).sort_values(["cluster_id","cluster_name"]).to_csv(OUT_REJECT, index=False, encoding="utf-8-sig")
print(f"Saved:\n - {OUT_VALID}\n - {OUT_REJECT}")
