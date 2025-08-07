import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv("sigma_input_cumulative_no2025.csv")

def logistic(x, a, b, x0):
    return a / (1 + np.exp(-(x - x0) / b))

def inverse_logistic(p_ratio, a, b, x0):
    if p_ratio <= 0 or p_ratio >= 1:
        return np.nan
    return x0 + b * np.log((1 / p_ratio) - 1)

results = []

for cluster_id in df["Cluster ID"].unique():
    cluster_df = df[df["Cluster ID"] == cluster_id]
    cluster_name = cluster_df["Cluster Name"].iloc[0]
    x = cluster_df["Year"].values
    y = cluster_df["Cumulative Count"].values

    if len(x) < 4 or max(y) == 0:
        continue

    try:
        a_init = max(y)
        b_init = (max(x) - min(x)) / 8
        x0_init = x[np.searchsorted(y, a_init / 2)] if any(y >= a_init / 2) else np.median(x)

        popt, _ = curve_fit(logistic, x, y, p0=[a_init, b_init, x0_init], maxfev=10000)
        a, b, x0 = popt

        if b <= 0:
            continue  # حذف مدل غیرمنطقی

        t_emerging = round(inverse_logistic(0.10, a, b, x0))
        t_growth   = round(inverse_logistic(0.50, a, b, x0))
        t_maturity = round(inverse_logistic(0.90, a, b, x0))
        t_saturation = round(t_maturity + 5) if not np.isnan(t_maturity) else np.nan

        # بررسی ترتیب مراحل
        if not (t_emerging < t_growth < t_maturity < t_saturation):
            continue  # رد کردن چرخه‌های غیرمنطقی

        results.append({
            "Technology_Theme": cluster_id,
            "Cluster_Name": cluster_name,
            "Emerging": t_emerging,
            "Growth": t_growth,
            "Maturity": t_maturity,
            "Saturation": t_saturation
        })

    except Exception:
        continue

out_df = pd.DataFrame(results)
out_df.to_csv("technology_lifecycle_phases_refined_v2.csv", index=False)
print("✅ فایل خروجی اصلاح‌شده ذخیره شد: technology_lifecycle_phases_refined_v2.csv")
