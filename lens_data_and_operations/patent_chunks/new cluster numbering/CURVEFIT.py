# CURVEFIT.py (fixed bounds to avoid Growth=2029 issue)
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

def logistic_rel(t_rel, k, a_rel, b):
    return k / (1.0 + np.exp(-(t_rel - a_rel) / b))

def year_at_pct_rel(k, a_rel, b, pct):
    if not np.all(np.isfinite([k, a_rel, b])) or k <= 0 or b == 0 or not (0 < pct < 1):
        return np.nan
    try:
        return a_rel + b * (-np.log((1.0 / pct) - 1.0))
    except Exception:
        return np.nan

INPUT_FILE  = "sigma_input_cumulative_no2025.csv"
OUT_VALID   = "cluster_lifecycle_fit_VALID.csv"
OUT_REJECT  = "cluster_lifecycle_fit_REJECTED.csv"
OUT_SIMPLE_COMMA = "cluster_stage_years_only.csv"
OUT_SIMPLE_SEMI  = "cluster_stage_years_only_semicolon.csv"
OUT_SIMPLE_TSV   = "cluster_stage_years_only.tsv"

df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]
req = {"Cluster ID","Cluster Name","Year","Cumulative Count"}
if not req.issubset(df.columns):
    raise ValueError("Missing required columns")

df = df.rename(columns={
    "Cluster ID":"cluster_id",
    "Cluster Name":"cluster_name",
    "Year":"year",
    "Cumulative Count":"cum"
}).dropna(subset=["cluster_id","cluster_name","year","cum"])

df["year"] = df["year"].astype(int)
df["cum"]  = df["cum"].astype(float)

valid_rows, rejected_rows = [], []

for (cid, cname), g in df.groupby(["cluster_id","cluster_name"], sort=False):
    g = g.sort_values("year").drop_duplicates("year", keep="last").copy()
    g["cum"] = g["cum"].cummax()

    years = g["year"].to_numpy(float)
    y     = g["cum"].to_numpy(float)

    base = {
        "cluster_id": cid,
        "cluster_name": cname,
        "first_year": int(years.min()) if len(years) else np.nan,
        "last_year":  int(years.max()) if len(years) else np.nan,
        "n_years":    int(len(years))
    }

    if len(years) < 4 or y[-1] <= 0:
        rejected_rows.append({**base,
            "emerging_year":np.nan,"growth_year":np.nan,"maturity_year":np.nan,"saturation_year":np.nan,
            "k":np.nan,"a":np.nan,"b":np.nan,"R2":np.nan,"RMSE":np.nan,"notes":"insufficient_data"})
        continue

    # --- reparameterize time to [0, tmax] ---
    t0 = years.min()
    t_rel = years - t0
    t_range = t_rel.max()
    if t_range <= 0:
        rejected_rows.append({**base,
            "emerging_year":np.nan,"growth_year":np.nan,"maturity_year":np.nan,"saturation_year":np.nan,
            "k":np.nan,"a":np.nan,"b":np.nan,"R2":np.nan,"RMSE":np.nan,"notes":"zero_time_range"})
        continue

    # --- init guesses ---
    k_min = y.max()
    k_max = 3.0 * y.max()
    k0 = min(max(1.15*y[-1], k_min), k_max)  # between [max(y), 3*max(y)]

    # a0: closest to halfway between min and last observed cum (robust to early noise)
    half_obs = (y[0] + y[-1]) / 2.0
    a0_rel = float(t_rel[np.argmin(np.abs(y - half_obs))])

    b0 = max(0.2, 0.25 * t_range)

    # bounds
    lb = np.array([k_min, 0.0,               0.05])
    ub = np.array([k_max, t_range, max(2.0, 0.5*t_range)])

    # residuals
    def resid(params):
        k,a_rel,b = params
        return logistic_rel(t_rel, k, a_rel, b) - y

    try:
        res = least_squares(resid, x0=np.array([k0, a0_rel, b0]),
                            bounds=(lb, ub), loss='soft_l1', max_nfev=20000)
        k_fit, a_rel_fit, b_fit = res.x
        yhat = logistic_rel(t_rel, k_fit, a_rel_fit, b_fit)

        # mark if any param at/near bound
        eps = 1e-6
        at_bound = (abs(k_fit-lb[0])<eps) or (abs(k_fit-ub[0])<eps) or \
                   (abs(a_rel_fit-lb[1])<eps) or (abs(a_rel_fit-ub[1])<eps) or \
                   (abs(b_fit-lb[2])<eps) or (abs(b_fit-ub[2])<eps)

        # stage years (calendar)
        emerg_rel = year_at_pct_rel(k_fit, a_rel_fit, b_fit, 0.10)
        grow_rel  = year_at_pct_rel(k_fit, a_rel_fit, b_fit, 0.50)  # = a_rel_fit
        matur_rel = year_at_pct_rel(k_fit, a_rel_fit, b_fit, 0.90)
        satur_rel = matur_rel + 5.0 if np.isfinite(matur_rel) else np.nan

        def to_year(x_rel):
            return int(round(t0 + x_rel)) if np.isfinite(x_rel) else np.nan

        emerg_y, grow_y, matur_y, satur_y = map(to_year, [emerg_rel, grow_rel, matur_rel, satur_rel])

        notes = []
        if any(pd.isna(v) for v in [emerg_y, grow_y, matur_y, satur_y]):
            notes.append("nan_stage_year")
        if not notes and not (emerg_y < grow_y < matur_y < satur_y):
            notes.append("non_monotonic_stage_years")
        if at_bound:
            notes.append("at_bound")

        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        R2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
        RMSE = float(np.sqrt(np.mean((y - yhat)**2)))

        if notes:
            rejected_rows.append({**base,
                "emerging_year":emerg_y,"growth_year":grow_y,"maturity_year":matur_y,"saturation_year":satur_y,
                "k":float(k_fit),"a":float(t0 + a_rel_fit),"b":float(b_fit),
                "R2":float(R2) if np.isfinite(R2) else np.nan,
                "RMSE":float(RMSE) if np.isfinite(RMSE) else np.nan,
                "notes":";".join(notes)})
            continue

        valid_rows.append({**base,
            "emerging_year":emerg_y,"growth_year":grow_y,"maturity_year":matur_y,"saturation_year":satur_y,
            "k":float(k_fit),"a":float(t0 + a_rel_fit),"b":float(b_fit),
            "R2":float(R2) if np.isfinite(R2) else np.nan,
            "RMSE":float(RMSE) if np.isfinite(RMSE) else np.nan,
            "notes":""})

    except Exception as e:
        rejected_rows.append({**base,
            "emerging_year":np.nan,"growth_year":np.nan,"maturity_year":np.nan,"saturation_year":np.nan,
            "k":np.nan,"a":np.nan,"b":np.nan,"R2":np.nan,"RMSE":np.nan,
            "notes":f"fit_failed:{type(e).__name__}"})

# ----- save outputs safely -----
if valid_rows:
    pd.DataFrame(valid_rows).sort_values(["cluster_id","cluster_name"]).to_csv(OUT_VALID, index=False, encoding="utf-8-sig")
    print(f"✅ {OUT_VALID}")
else:
    print("⚠️ هیچ خوشه‌ای معتبر نبود؛ VALID ساخته نشد.")

if rejected_rows:
    pd.DataFrame(rejected_rows).sort_values(["cluster_id","cluster_name"]).to_csv(OUT_REJECT, index=False, encoding="utf-8-sig")
    print(f"📄 {OUT_REJECT}")
else:
    print("ℹ️ هیچ خوشه‌ای رد نشد؛ REJECTED ساخته نشد.")

# simple outputs
if valid_rows:
    simple = pd.DataFrame(valid_rows, columns=["cluster_id","emerging_year","growth_year","maturity_year","saturation_year"]).copy()
    for c in ["emerging_year","growth_year","maturity_year","saturation_year"]:
        simple[c] = pd.to_numeric(simple[c], errors="coerce").astype("Int64")
    simple = simple.sort_values("cluster_id")
    simple.to_csv(OUT_SIMPLE_COMMA, index=False, encoding="utf-8-sig")
    simple.to_csv(OUT_SIMPLE_SEMI,  index=False, encoding="utf-8-sig", sep=";")
    simple.to_csv(OUT_SIMPLE_TSV,   index=False, encoding="utf-8-sig", sep="\t")
    print(f"🧾 {OUT_SIMPLE_COMMA} | {OUT_SIMPLE_SEMI} | {OUT_SIMPLE_TSV}")
else:
    print("⚠️ چون VALID خالی بود، فایل ساده ساخته نشد.")
