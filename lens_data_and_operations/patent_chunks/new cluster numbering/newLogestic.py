import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 📥 خواندن داده‌ها
df = pd.read_csv("sigma_input_cumulative_no2025.csv")

# 📈 تابع لجستیک
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# 🔁 معکوس تابع لجستیک برای محاسبه سال رسیدن به نسبت خاصی از K
def inverse_logistic(p_ratio, K, r, t0):
    if p_ratio <= 0 or p_ratio >= 1:
        return np.nan
    return - (np.log((K / (p_ratio * K)) - 1)) / r + t0

# 📤 خروجی‌ها
results = []

# 🧪 اجرای تحلیل برای هر کلاستر
for cluster_id in df["Cluster ID"].unique():
    cluster_df = df[df["Cluster ID"] == cluster_id]
    cluster_name = cluster_df["Cluster Name"].iloc[0]
    years = cluster_df["Year"].values
    counts = cluster_df["Cumulative Count"].values

    if len(years) < 4 or max(counts) == 0:
        continue  # حذف کلاستر ناقص

    try:
        # تخمین اولیه
        initial_guess = [max(counts), 0.3, np.median(years)]
        popt, _ = curve_fit(logistic, years, counts, p0=initial_guess, maxfev=10000)
        K, r, t0 = popt

        # محاسبه سال ورود به هر مرحله
        t_emerging = round(inverse_logistic(0.10, K, r, t0))
        t_growth   = round(inverse_logistic(0.50, K, r, t0))
        t_maturity = round(inverse_logistic(0.90, K, r, t0))

        # بررسی منطقی بودن Saturation
        t_saturation = round(t_maturity + 5) if not np.isnan(t_maturity) else np.nan

        results.append({
            "Technology_Theme": cluster_id,
            "Cluster_Name": cluster_name,
            "Emerging": t_emerging,
            "Growth": t_growth,
            "Maturity": t_maturity,
            "Saturation": t_saturation
        })

    except Exception as e:
        continue  # اگر فیت نشد، رد کن

# 💾 ذخیره فایل خروجی
out_df = pd.DataFrame(results)
out_df.to_csv("technology_lifecycle_phases_corrected.csv", index=False)

print("✅ فایل خروجی درست شده: technology_lifecycle_phases_corrected.csv")
