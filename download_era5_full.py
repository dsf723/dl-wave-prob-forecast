import cdsapi
import os
import time

c = cdsapi.Client(verify=0)

# 3 个点位
sites = {
    "zhoushan": {"area": [31.5, 122.0, 29.5, 124.0]},
    "oregon":   {"area": [41.8, -131.0, 39.8, -129.0]},
    "hawaii":   {"area": [21.7, -159.0, 19.7, -157.0]},
}

variables = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period",
    "mean_wave_direction",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# 按季度分：Q1=1-3月, Q2=4-6月, Q3=7-9月, Q4=10-12月
quarters = {
    "Q1": ["01", "02", "03"],
    "Q2": ["04", "05", "06"],
    "Q3": ["07", "08", "09"],
    "Q4": ["10", "11", "12"],
}

out_dir = r"F:\Offshore Power Forecasting\era5_data"
os.makedirs(out_dir, exist_ok=True)

years = [str(y) for y in range(1995, 2025)]

for site_name, site_info in sites.items():
    for year in years:
        for q_name, months in quarters.items():
            out_file = os.path.join(out_dir, f"{site_name}_{year}_{q_name}.nc")
            
            if os.path.exists(out_file) and os.path.getsize(out_file) > 10000:
                print(f"[跳过] {out_file}")
                continue
            
            print(f"[下载] {site_name} {year} {q_name} ...")
            
            try:
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": variables,
                        "year": year,
                        "month": months,
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                                 "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                                 "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                                 "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"],
                        "area": site_info["area"],
                        "format": "netcdf",
                    },
                    out_file
                )
                size_mb = os.path.getsize(out_file) / 1024 / 1024
                print(f"[完成] {out_file} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[失败] {site_name} {year} {q_name}: {e}")
                if os.path.exists(out_file):
                    os.remove(out_file)
            
            time.sleep(1)

print("\n=== 全部下载完成 ===")
