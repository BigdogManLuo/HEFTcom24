import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

# 数据准备（保持原数据不变）
lats_dwd = [53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
lons_dwd = [1.702, 1.767, 1.832, 1.897, 1.962, 2.027]
lats_gfs = [53.59, 53.84, 54.09]
lons_gfs = [1.522, 1.772, 2.022]
lat_solar = [52.4872562,52.8776682,52.1354277,52.4880497,51.9563696,52.2499177,
            52.6416477,52.2700912,52.1960768,52.7082618,52.4043468,52.0679429,
            52.024023,52.7681276,51.8750506,52.5582373,52.4478922,52.5214863,
            52.8776682,52.0780721]
lon_solar = [0.4012455,0.7906532,-0.2640343,-0.1267052,0.6588173,1.3894081,
            1.3509559,0.7082557,0.1534462,0.7302284,1.0762977,1.1751747,
            0.2962684,0.1699257,0.9115028,0.7137489,0.1204872,1.5706825,
            1.1916542,-0.0113488]

# 生成坐标点
def generate_coordinates(lats, lons):
    return [f"{lat:.3f}°N, {lon:.3f}°E" for lat, lon in zip(lats, lons)]

coordinates_dwd = generate_coordinates(
    [lat for lat in lats_dwd for _ in lons_dwd],
    [lon for _ in lats_dwd for lon in lons_dwd]
)

coordinates_gfs = generate_coordinates(
    [lat for lat in lats_gfs for _ in lons_gfs],
    [lon for _ in lats_gfs for lon in lons_gfs]
)

# 创建可视化对象
dwd_trace = go.Scattergeo(
    lon=[lon for _ in lats_dwd for lon in lons_dwd],
    lat=[lat for lat in lats_dwd for _ in lons_dwd],
    mode='markers',
    name='DWD (Hornsea1)',
    marker=dict(
        size=8,
        color='#74b9ff',  # 标准蓝色
        symbol='circle',
        line=dict(width=0.5, color='black'),
        opacity=0.8
    ),
    hoverinfo='text',
    text=coordinates_dwd
)

gfs_trace = go.Scattergeo(
    lon=[lon for _ in lats_gfs for lon in lons_gfs],
    lat=[lat for lat in lats_gfs for _ in lons_gfs],
    mode='markers',
    name='GFS (Hornsea1)',
    marker=dict(
        size=8,
        color='#ff7f0e',  # 标准橙色
        symbol='square',
        line=dict(width=0.5, color='black'),
        opacity=0.8
    ),
    hoverinfo='text',
    text=coordinates_gfs
)

solar_trace = go.Scattergeo(
    lon=lon_solar,
    lat=lat_solar,
    mode='markers',
    name='DWD & GFS (Photovoltaic)',
    marker=dict(
        size=8,
        color='#ff7675',  # 标准绿色
        symbol='triangle-up',
        line=dict(width=0.5, color='black'),
        opacity=0.8
    ),
    hoverinfo='text',
    text=[f"{lat:.5f}°N, {lon:.5f}°E" for lat, lon in zip(lat_solar, lon_solar)]
)

# 专业布局设置
layout = go.Layout(
    title=dict(
        text='Geospatial Distribution of Meteorological Grid Points',
        x=0.5,
        xanchor='center',
        font=dict(size=20, family='Times New Roman, sans-serif')
    ),
    showlegend=True,
    legend=dict(
        x=0.5,
        y=0.1,
        bgcolor='rgba(255,255,255,0.7)',
        font=dict(size=25, family='Times New Roman, sans-serif')
    ),
    geo=dict(
        resolution=50,
        showland=True,
        showlakes=True,
        landcolor='rgb(240, 240, 240)',
        countrycolor='rgb(200, 200, 200)',
        lakecolor='rgb(255, 255, 255)',
        coastlinecolor='rgb(150, 150, 150)',
        coastlinewidth=1,
        showcountries=True,
        projection=dict(type="mercator"),
        lataxis=dict(
            range=[51, 54.5],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)',
            dtick=1
        ),
        lonaxis=dict(
            range=[-3, 3],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)',
            dtick=1
        ),
        projection_scale=1,
        center=dict(lat=53.2, lon=1.65)
    ),
    font=dict(
        family='Arial, sans-serif',
        size=12,
        color='black'
    ),
    margin=dict(l=50, r=50, b=50, t=80, pad=4),
    width=1000,
    height=800,
    annotations=[
        dict(
            x=0.7,
            y=0.02,
            xref='paper',
            yref='paper',
            text='Map data © Natural Earth, CARTO',
            showarrow=False,
            font=dict(size=20, color='#666666'),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='#cccccc',
            borderwidth=1,
            borderpad=4
        ),
        dict(
            x=0.05,
            y=0.05,
            xref='paper',
            yref='paper',
            text='',
            showarrow=False,
            font=dict(size=12),
            xanchor='left'
        ),
        dict(
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            text='',
            showarrow=False,
            font=dict(size=12),
            xanchor='left'
        )
    ]
)

# 组合图形
fig = go.Figure(data=[dwd_trace, gfs_trace, solar_trace], layout=layout)

# 导出图形
plot(fig, validate=False, filename='academic_geo_plot.html')


# 在生成fig对象后添加：
fig.update_layout(
    # 确保所有文本元素可见
    font=dict(size=12), 
    # 调整输出尺寸（保持长宽比）
    width=1200,  # 增大宽度
    height=960   # 增大高度
)


# 专业级保存参数
# fig.write_image(
#     "output.png",
#     engine="kaleido",
#     scale=2,  # 平衡质量与速度
#     width=1200,  # 原始尺寸略增
#     height=960,
#     validate=False,  # 关闭校验加速
# )

'''
# 创建坐标点
coordinates_dwd = []
coordinates_gfs = []
idx = 0
for lat in lats_dwd:
    for lon in lons_dwd:
        coordinates_dwd.append({"lat": lat, "lon": lon, "name": f"dwd_{idx}"})
        idx += 1

idx = 0
for lat in lats_gfs:
    for lon in lons_gfs:
        coordinates_gfs.append({"lat": lat, "lon": lon, "name": f"gfs_{idx}"})
        idx += 1

# 转换为 DataFrame
df_dwd = pd.DataFrame(coordinates_dwd)
df_gfs = pd.DataFrame(coordinates_gfs)

# 绘制地图
zoom=8.5
fig = px.scatter_mapbox(
    df_dwd,
    lat="lat",
    lon="lon",
    hover_name="name",
    mapbox_style="open-street-map",
    zoom=zoom,
    color_discrete_sequence=["blue"],
)

fig.add_trace(
    px.scatter_mapbox(
        df_gfs,
        lat="lat",
        lon="lon",
        hover_name="name",
        color_discrete_sequence=["red"],
    ).data[0]
)
'''


'''
# 添加dwd覆盖面的多边形
polygon = go.Scattermapbox(
    fill="toself",
    lon=[lons_dwd[0], lons_dwd[5], lons_dwd[5], lons_dwd[0], lons_dwd[0]],
    lat=[lats_dwd[0], lats_dwd[0], lats_dwd[5], lats_dwd[5], lats_dwd[0]],
    marker={'size': 0, 'color': "#74b9ff"},
    name="覆盖面"
)

fig.add_trace(polygon)

# 添加gfs覆盖面的多边形
polygon = go.Scattermapbox(
    fill="toself",
    lon=[lons_gfs[0], lons_gfs[2], lons_gfs[2], lons_gfs[0], lons_gfs[0]],
    lat=[lats_gfs[0], lats_gfs[0], lats_gfs[2], lats_gfs[2], lats_gfs[0]],
    marker={'size': 0, 'color': "#ff7675"},
    name="覆盖面"
)

fig.add_trace(polygon)
'''


#fig.show()



'''
#==========================================绘制光伏场地图==========================================
#经纬度点
lat_solar=[52.4872562,52.8776682,52.1354277,52.4880497,51.9563696,52.2499177,52.6416477,52.2700912,52.1960768,52.7082618,52.4043468,52.0679429,52.024023,52.7681276,51.8750506,52.5582373,52.4478922,52.5214863,52.8776682,52.0780721]
lon_solar=[0.4012455,0.7906532,-0.2640343,-0.1267052,0.6588173,1.3894081,1.3509559,0.7082557,0.1534462,0.7302284,1.0762977,1.1751747,0.2962684,0.1699257,0.9115028,0.7137489,0.1204872,1.5706825,1.1916542,-0.0113488]

#创建坐标点
coordinates_solar=[]
for i in range(len(lat_solar)):
    coordinates_solar.append({"lat":lat_solar[i],"lon":lon_solar[i],"name":f"solar_{i}","size":0.001})



#转换为DataFrame
df_solar=pd.DataFrame(coordinates_solar)

#绘制地图
zoom=8
fig=px.scatter_mapbox(
    df_solar,
    lat="lat",
    lon="lon",
    hover_name="name",
    mapbox_style="open-street-map",
    zoom=zoom,
    color_discrete_sequence=["red"],
    size="size",
    size_max=8  # 调整点的最大大小
)
fig.show()
'''



'''
* latitude           (latitude) float64 53.77 53.84 53.9 53.97 54.03 54.1
* longitude          (longitude) float64 1.702 1.767 1.832 1.897 1.962 2.027

ref_datetime	valid_datetime	point	latitude	longitude
2020-09-20	0	0	52.4872562	0.4012455
2020-09-20	0	1	52.8776682	0.7906532
2020-09-20	0	2	52.1354277	-0.2640343
2020-09-20	0	3	52.4880497	-0.1267052
2020-09-20	0	4	51.9563696	0.6588173
2020-09-20	0	5	52.2499177	1.3894081
2020-09-20	0	6	52.6416477	1.3509559
2020-09-20	0	7	52.2700912	0.7082557
2020-09-20	0	8	52.1960768	0.1534462
2020-09-20	0	9	52.7082618	0.7302284
2020-09-20	0	10	52.4043468	1.0762977
2020-09-20	0	11	52.0679429	1.1751747
2020-09-20	0	12	52.024023	0.2962684
2020-09-20	0	13	52.7681276	0.1699257
2020-09-20	0	14	51.8750506	0.9115028
2020-09-20	0	15	52.5582373	0.7137489
2020-09-20	0	16	52.4478922	0.1204872
2020-09-20	0	17	52.5214863	1.5706825
2020-09-20	0	18	52.8776682	1.1916542
2020-09-20	0	19	52.0780721	-0.0113488



  * latitude           (latitude) float64 53.59 53.84 54.09
  * longitude          (longitude) float64 1.522 1.772 2.022


'''
