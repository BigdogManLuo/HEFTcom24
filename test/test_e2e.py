import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from utils_e2e import generateModellingTable, trainTestSplit, NNDataset, MLPModel, trading_loss,train_model,test_model

def run_case(caseNumber, acc_oriented=True, e2e=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    IntegratedDataset, modelling_table = generateModellingTable(caseNumber)
    results = {}
    # acc_oriented
    if acc_oriented:
        features_train, features_test, labels_train, labels_test, train_data, test_data, scaler_labels = trainTestSplit(modelling_table, caseNumber, is_E2E=False)
        acc_model_path = f"../models/Prices/{'history' if caseNumber==1 else 'latest'}/acc_focus.pth"
        model_params = dict(input_size=features_train.shape[1], hidden_layer_size=16, output_size=1, num_layers=3)
        train_params = dict(batch_size=1024, lr=1e-4, num_epochs=50 if caseNumber==1 else 40, device=device)
        model = train_model(features_train, labels_train, train_data, model_params, train_params, acc_model_path, is_E2E=False)
        Revenue, pd_pred = test_model(model, features_test, scaler_labels, test_data, device, is_E2E=False)
        results['acc_oriented'] = dict(Revenue=Revenue, pd_pred=pd_pred)
        np.save(f"../data/revenues/case{caseNumber}/Revenue_acc_oriented.npy", Revenue)
        np.save(f"../data/revenues/case{caseNumber}/pd_pred_acc_oriented.npy", pd_pred)
    # e2e
    if e2e:
        features_train, features_test, labels_train, labels_test, train_data, test_data, scaler_labels = trainTestSplit(modelling_table, caseNumber, is_E2E=True)
        e2e_model_path = f"../models/Prices/{'history' if caseNumber==1 else 'latest'}/e2e.pth"
        model_params = dict(input_size=features_train.shape[1], hidden_layer_size=32 if caseNumber==1 else 8, output_size=1, num_layers=2)
        train_params = dict(batch_size=512, lr=1e-2, num_epochs=40 if caseNumber==1 else 60, device=device)
        model = train_model(features_train, labels_train, train_data, model_params, train_params, e2e_model_path, is_E2E=True)
        Revenue, pd_pred = test_model(model, features_test, scaler_labels, test_data, device, is_E2E=True)
        results['e2e'] = dict(Revenue=Revenue, pd_pred=pd_pred)
        np.save(f"../data/revenues/case{caseNumber}/Revenue_e2e.npy", Revenue)
        np.save(f"../data/revenues/case{caseNumber}/pd_pred_e2e.npy", pd_pred)

def print_summary():
    
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

if __name__ == "__main__":
    all_results = {}
    for case in [1,2]:
        run_case(case, acc_oriented=True, e2e=True)
    print_summary()

