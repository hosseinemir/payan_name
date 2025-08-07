import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# تابع لجستیک مطابق مقاله استاد
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# مسیر فایل ورودی
input_file = "sigma_input_cumulative_no2025.csv"
df = pd.read_csv(input_file)

# خروجی نهایی
results = []

# پردازش برای هر کلاستر
for cluster_id in df["Cluster ID"].unique():
    cluster_df = df[df["Cluster ID"] == cluster_id].copy()
    years = cluster_df["Year"].values
    counts = cluster_df["Cumulative Count"].values

    if len(years) < 4 or max(counts) == 0:
        continue  # اگر داده کافی نباشه، رد می‌کنیم

    try:
        # مقدار اولیه برای تخمین
        initial_guess = [max(counts), 0.3, np.median(years)]
        
        # فیت تابع لجستیک
        popt, _ = curve_fit(logistic, years, counts, p0=initial_guess, maxfev=10000)
        K, r, t0 = popt
        
        # محاسبه P(t) برای آخرین سال
        final_year = max(years)
        P_t = logistic(final_year, K, r, t0)
        ratio = P_t / K

        results.append({
            "Cluster ID": cluster_id,
            "K": K,
            "P(t)": P_t,
            "P(t)/K": ratio
        })

    except:
        continue  # اگر فیت موفق نشد، ادامه می‌دهیم

# ذخیره فایل خروجی
output_df = pd.DataFrame(results)
output_df.to_csv("logistic_fit_results.csv", index=False)

print("✅ تحلیل انجام شد. فایل خروجی ذخیره شد به نام logistic_fit_results.csv")
