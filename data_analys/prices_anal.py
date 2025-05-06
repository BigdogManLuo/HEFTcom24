import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Load the data
energy_data = pd.read_csv('../data/raw/Energy_Data_20200920_20240118.csv')

# Convert the 'Date' column to datetime
energy_data['dtm'] = pd.to_datetime(energy_data['dtm'])

#Hour of day
energy_data['hour'] = energy_data['dtm'].dt.hour

#Price Spread
energy_data['Price_Spread'] = energy_data['DA_Price'] - energy_data['SS_Price']

#Select half year of data, decreasing by year months from 2024-01-01
energy_data1 = energy_data[(energy_data['dtm'] <= '2024-01-01') & (energy_data['dtm'] >= '2023-07-01')]
energy_data2 = energy_data[(energy_data['dtm'] <= '2023-07-01') & (energy_data['dtm'] >= '2023-01-01')]
energy_data3 = energy_data[(energy_data['dtm'] <= '2023-01-01') & (energy_data['dtm'] >= '2022-07-01')]
energy_data4 = energy_data[(energy_data['dtm'] <= '2022-07-01') & (energy_data['dtm'] >= '2022-01-01')]

#Plot the average price and volatility range for each hour, sorted by hour of day.

#Mean value
mean_value1=energy_data1.groupby('hour')['Price_Spread'].agg(['mean'])
mean_value2=energy_data2.groupby('hour')['Price_Spread'].agg(['mean'])
mean_value3=energy_data3.groupby('hour')['Price_Spread'].agg(['mean'])
mean_value4=energy_data4.groupby('hour')['Price_Spread'].agg(['mean'])


#Standard deviation
std_value1=energy_data1.groupby('hour')['Price_Spread'].agg(['std'])
std_value2=energy_data2.groupby('hour')['Price_Spread'].agg(['std'])
std_value3=energy_data3.groupby('hour')['Price_Spread'].agg(['std'])
std_value4=energy_data4.groupby('hour')['Price_Spread'].agg(['std'])

#Plot
plt.style.use(['science','ieee'])
plt.figure(figsize=(12,12))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 21})
Mean_value={'mean1':mean_value1['mean'],'mean2':mean_value2['mean'],'mean3':mean_value3['mean'],'mean4':mean_value4['mean']}
Std_value={'std1':std_value1['std'],'std2':std_value2['std'],'std3':std_value3['std'],'std4':std_value4['std']}
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.errorbar(Mean_value['mean'+str(i+1)].index,Mean_value['mean'+str(i+1)],yerr=Std_value['std'+str(i+1)],fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.xlabel('Hour of Day')
    plt.ylabel('Price Spread (£/MWh)')
    plt.grid()

    if i==0:
        plt.title('2024-01-01 to 2023-07-01')
    elif i==1:
        plt.title('2023-07-01 to 2023-01-01')
    elif i==2:
        plt.title('2023-01-01 to 2022-07-01')
    else:
        plt.title('2022-07-01 to 2022-01-01')

    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    plt.legend(['Mean'],loc='upper right')

    plt.gca().tick_params(width=2)
    plt.gca().tick_params(axis='x',direction='in',length=6,width=2)
    plt.gca().tick_params(axis='y',direction='in',length=6,width=2)
    plt.gca().yaxis.set_tick_params(width=2)
    plt.gca().xaxis.set_tick_params(width=2)

    plt.ylim(-150,150)

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
    
plt.tight_layout()

plt.savefig('../figs/price_spread.png',dpi=660)
