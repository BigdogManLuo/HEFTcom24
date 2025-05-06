import numpy as np
import matplotlib.pyplot as plt
import scienceplots

#Load Data
Revenue_case1={
    "q50":np.load("../data/revenues/case1/Revenue_q50.npy"),
    "ST":np.load("../data/revenues/case1/Revenue_ST.npy"),
    "acc_oriented":np.load("../data/revenues/case1/Revenue_acc_oriented.npy"),
    "e2e":np.load("../data/revenues/case1/Revenue_e2e.npy")
}

Revenue_case2={
    "q50":np.load("../data/revenues/case2/Revenue_q50.npy"),
    "ST":np.load("../data/revenues/case2/Revenue_ST.npy"),
    "acc_oriented":np.load("../data/revenues/case2/Revenue_acc_oriented.npy"),
    "e2e":np.load("../data/revenues/case2/Revenue_e2e.npy")
}



OptimalRevenue_case1=np.load("../data/revenues/case1/OptimalRevenue.npy")
OptimalRevenue_case2=np.load("../data/revenues/case2/OptimalRevenue.npy")

# Calculate Regret
Regret_case1={}
for key in Revenue_case1.keys():
    Regret_case1[key]=OptimalRevenue_case1-Revenue_case1[key]
    
Regret_case2={}
for key in Revenue_case2.keys():
    Regret_case2[key]=OptimalRevenue_case2-Revenue_case2[key]


power_pred_case1=np.load("../data/revenues/case1/power_pred.npy")
power_pred_case2=np.load("../data/revenues/case2/power_pred.npy")

power_true_case1=np.load("../data/revenues/case1/power_true.npy")
power_true_case2=np.load("../data/revenues/case2/power_true.npy")

pd_pred_case1=np.load("../data/revenues/case1/pd_pred.npy")
pd_pred_case2=np.load("../data/revenues/case2/pd_pred.npy")
pd_pred_e2e_case1=np.load("../data/revenues/case1/pd_pred_e2e.npy")
pd_pred_e2e_case2=np.load("../data/revenues/case2/pd_pred_e2e.npy")
pd_pred_acc_oriented_case1=np.load("../data/revenues/case1/pd_pred_acc_oriented.npy")
pd_pred_acc_oriented_case2=np.load("../data/revenues/case2/pd_pred_acc_oriented.npy")

pd_true_case1=np.load("../data/revenues/case1/pd_true.npy")
pd_true_case2=np.load("../data/revenues/case2/pd_true.npy")

#%% 
power_error_case1=power_pred_case1-power_true_case1
power_error_case2=power_pred_case2-power_true_case2

pd_error_q50_case1=0-pd_true_case1
pd_error_q50_case2=0-pd_true_case2
pd_error_ST_case1=pd_pred_case1-pd_true_case1
pd_error_ST_case2=pd_pred_case2-pd_true_case2
pd_error_e2e_case1=pd_pred_e2e_case1-pd_true_case1
pd_error_e2e_case2=pd_pred_e2e_case2-pd_true_case2
pd_error_acc_oriented_case1=pd_pred_acc_oriented_case1-pd_true_case1
pd_error_acc_oriented_case2=pd_pred_acc_oriented_case2-pd_true_case2

# Calculate E2E Loss
Loss_e2e_case1 = 0.07 * power_error_case1**2 + 3.57 * pd_error_e2e_case1**2 + power_error_case1 * pd_error_e2e_case1
Loss_acc_oriented_case1 = 0.07 * power_error_case1**2 + 3.57 * pd_error_acc_oriented_case1**2 + power_error_case1 * pd_error_acc_oriented_case1
Loss_ST_case1 = 0.07 * power_error_case1**2 + 3.57 * pd_error_ST_case1**2 + power_error_case1 * pd_error_ST_case1
Loss_q50_case1 = 0.07 * power_error_case1**2 + 3.57 * pd_error_q50_case1**2 + power_error_case1 * pd_error_q50_case1

Loss_e2e_case2 = 0.07 * power_error_case2**2 + 3.57 * pd_error_e2e_case2**2 + power_error_case2 * pd_error_e2e_case2
Loss_acc_oriented_case2 = 0.07 * power_error_case2**2 + 3.57 * pd_error_acc_oriented_case2**2 + power_error_case2 * pd_error_acc_oriented_case2
Loss_ST_case2 = 0.07 * power_error_case2**2 + 3.57 * pd_error_ST_case2**2 + power_error_case2 * pd_error_ST_case2
Loss_q50_case2 = 0.07 * power_error_case2**2 + 3.57 * pd_error_q50_case2**2 + power_error_case2 * pd_error_q50_case2

plt.style.use(['science', 'ieee'])
plt.figure(figsize=(24, 12))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})

# Subplot 1
plt.subplot(2, 4, 1)
plt.scatter(power_error_case1, pd_error_q50_case1, s=1, alpha=0.5, color='b')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-500, 600)
plt.title('q50 (Case 1)')
plt.grid()

# Subplot 2
plt.subplot(2, 4, 2)
plt.scatter(power_error_case1, pd_error_ST_case1, s=1, alpha=0.5, color='b')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-500, 600)
plt.title('ST (Case 1)')
plt.grid()

# Subplot 3
plt.subplot(2, 4, 3)
plt.scatter(power_error_case1, pd_error_acc_oriented_case1, s=1, alpha=0.5, color='b')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-500, 600)
plt.title('Acc-Oriented (Case 1)')
plt.grid()

# Subplot 4 with red points for Loss_e2e_case1 < Loss_ST_case1
plt.subplot(2, 4, 4)
mask = (Loss_e2e_case1 < Loss_ST_case1) & (abs(pd_error_ST_case1)<abs(pd_error_e2e_case1))
plt.scatter(power_error_case1[~mask], pd_error_e2e_case1[~mask], s=1, alpha=0.5, color='b', label='Normal')
plt.scatter(power_error_case1[mask], pd_error_e2e_case1[mask], s=2, alpha=1, color='#e84393', label='SoDR')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-500, 600)
plt.title('E2E (Case 1)')
plt.legend()
plt.grid()

# Subplot 5
plt.subplot(2, 4, 5)
plt.scatter(power_error_case2, pd_error_q50_case2, s=1, alpha=0.5, color='b')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-300, 300)
plt.title('q50 (Case 2)')
plt.grid()

# Subplot 6
plt.subplot(2, 4, 6)
plt.scatter(power_error_case2, pd_error_ST_case2, s=1, alpha=0.5, color='b')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-300, 300)
plt.title('ST (Case 2)')
plt.grid()

# Subplot 7
plt.subplot(2, 4, 7)
plt.scatter(power_error_case2, pd_error_acc_oriented_case2, s=1, alpha=0.5, color='b')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-300, 300)
plt.title('Acc-Oriented (Case 2)')
plt.grid()

# Subplot 8 with red points for Loss_e2e_case2 < Loss_ST_case2
plt.subplot(2, 4, 8)
mask = (Loss_e2e_case2 < Loss_ST_case2) & (abs(pd_error_ST_case2)<abs(pd_error_e2e_case2))
plt.scatter(power_error_case2[~mask], pd_error_e2e_case2[~mask], s=1, alpha=0.5, color='b', label='Normal')
plt.scatter(power_error_case2[mask], pd_error_e2e_case2[mask], s=2, alpha=1, color='#e84393', label='SoDR')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-300, 300)
plt.title('E2E (Case 2)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('../figs/power_price_error.png', dpi=660)

print("=================================== Case 1 Revenue ===================================")
print("q50:", Revenue_case1["q50"].sum())
print("ST:", Revenue_case1["ST"].sum())
print("acc_oriented:", Revenue_case1["acc_oriented"].sum())
print("e2e:", Revenue_case1["e2e"].sum())
print("=================================== Case 2 Revenue ===================================")
print("q50:", Revenue_case2["q50"].sum())
print("ST:", Revenue_case2["ST"].sum())
print("acc_oriented:", Revenue_case2["acc_oriented"].sum())
print("e2e:", Revenue_case2["e2e"].sum())

print("=================================== Case 1 Regret ===================================")
print("q50:", Regret_case1["q50"].sum())
print("ST:", Regret_case1["ST"].sum())
print("acc_oriented:", Regret_case1["acc_oriented"].sum())
print("e2e:", Regret_case1["e2e"].sum())
print("=================================== Case 2 Regret ===================================")
print("q50:", Regret_case2["q50"].sum())
print("ST:", Regret_case2["ST"].sum())
print("acc_oriented:", Regret_case2["acc_oriented"].sum())
print("e2e:", Regret_case2["e2e"].sum())

'''
plt.figure(figsize=(8, 6))
mask = (Loss_e2e_case1 < Loss_ST_case1) & (abs(pd_error_ST_case1)<abs(pd_error_e2e_case1))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 18})
plt.scatter(power_error_case1[~mask], -pd_error_e2e_case1[~mask], s=1, alpha=0.5, color='b', label='Normal')
plt.scatter(power_error_case1[mask], pd_error_e2e_case1[mask], s=2, alpha=1, color='#e84393', label='SoDR')
plt.xlabel('Power Error (MWh)')
plt.ylabel('Price Error (£/MWh)')
plt.ylim(-500, 600)
plt.title('E2E (Case 1)')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/pd_error_e2e_case1.png', dpi=660)
plt.show()
'''