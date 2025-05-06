import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def format_func(value, tick_number):
    if value >= 1e6:
        return f'{value/1e6:.1f}M'  
    elif value >= 1e3:
        return f'{value/1e3:.1f}k' 
    else:
        return f'{value:.0f}'  

x = np.arange(-200, 210, 10)
y = np.arange(-200, 210, 10)
x, y = np.meshgrid(x, y)
z = 0.07 * x**2 + 3.57 * y**2 + x * y


plt.figure(figsize=(8,6))
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'font.size': 20})
heatmap = plt.contourf(x, y, z, cmap='coolwarm', levels=50)
cbar = plt.colorbar(heatmap)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Trading Loss (£)', fontname='Times New Roman', fontsize=16)
plt.xlabel('Power Forecasting Error (MWh)', fontname='Times New Roman')
plt.ylabel('Price Spread Forecasting Error (£/MWh)', fontname='Times New Roman')
plt.savefig('../figs/trading_loss.png',dpi=660)

