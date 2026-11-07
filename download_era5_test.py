import cdsapi

c = cdsapi.Client(verify=0)

# 测试下载：舟山点位 2023-01-01 一天的波浪数据
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'significant_height_of_combined_wind_waves_and_swell',
            'mean_wave_period',
            'mean_wave_direction',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
        ],
        'year': '2023',
        'month': '01',
        'day': ['01'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': [31.5, 122.0, 29.5, 124.0],
        'format': 'netcdf',
    },
    'test_zhoushan.nc'
)

print("下载完成：test_zhoushan.nc")
