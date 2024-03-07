from utils import Trainer,NNRegressor
import pandas as pd
import numpy as np

if __name__=="__main__":
    
        #加载数据集
        type="wind"
        dataset=pd.read_csv(f"data/dataset/dwd/{type.capitalize()}Dataset.csv")

        #提取features和labels
        features_table=dataset.iloc[:,:-1]
        labels_table=dataset.iloc[:,-1]

        #划分训练集和测试集
        train_features_table=features_table[int(0.35*len(features_table)):int(0.8*len(features_table))]
        train_labels_table=labels_table[int(0.35*len(labels_table)):int(0.8*len(labels_table))]
        test_features_table=features_table[int(0.8*len(features_table)):]
        test_labels_table=labels_table[int(0.8*len(labels_table)):]

        #转换为numpy
        train_features=np.array(train_features_table)
        train_labels=np.array(train_labels_table)
        test_features=np.array(test_features_table)
        test_labels=np.array(test_labels_table)
    
        #训练器配置
        params={
            "lr": 0.001,
            "batch_size": 1024,
            "num_epochs": 40,
            "hidden_size": 64,
            "num_layers": 2,
            "loss_fn": "quantile",
            "alpha": 0.5,
            "verbose":True
        }

        model=NNRegressor(**params)
        model.fit(train_features,train_labels)
        forecast=model.predict(test_features)

        #评估
        print("MAE:",np.mean(np.abs(forecast-test_labels)))

