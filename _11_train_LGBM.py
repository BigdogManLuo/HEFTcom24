from lightgbm import LGBMRegressor
from utils import Trainer


if __name__=="__main__":
    
    configs={
        "source":"dwd", #数据来源
        "type":"solar", #预测对象类型
        "Regressor":LGBMRegressor, #模型类型
        "full":False, #是否使用全量数据
        "model_name":"LGBM" #模型名称
        }
    trainer=Trainer(**configs)
    
    Params={}
    Num_leaves=[350,350,350,350,600,475,475,400,150]
    Num_estimators=[550,550,550,400,1000,1000,385,300,650]
    Min_data_in_leaf=[300,300,200,300,200,200,400,200,200]
    l2=[5,5,5,5,0,30,30,20,50]
    
    for idx, quantile in enumerate(range(10,100,10)):
        
        params={
            'objective': 'quantile',
            'num_leaves': Num_leaves[idx],
            'n_estimators': Num_estimators[idx],
            'min_data_in_leaf': Min_data_in_leaf[idx],
            'lambda_l2':l2[idx],
            'verbose':-1,
            'alpha':quantile/100
            }
        
        Params[f"q{quantile}"]=params
    
    trainer.train(Params)
    trainer.test()
    
    
    #-------------wind  dwd----------------------
    configs={
        "source":"dwd", #数据来源
        "type":"wind", #预测对象类型
        "Regressor":LGBMRegressor, #模型类型
        "full":False, #是否使用全量数据
        "model_name":"LGBM" #模型名称
        }
    trainer=Trainer(**configs)

    Params={}
    Num_leaves=[100,100,250,425,600,425,100,275,250]
    Num_estimators=[100,100,100,100,100,100,200,100,100]
    Min_data_in_leaf=[700,300,1000,1000,800,1000,300,900,900]
    l2=[30,0,50,50,20,35,0,30,40]
    
    for idx, quantile in enumerate(range(10,100,10)):
        
        params={
            'objective': 'quantile',
            'alpha':quantile/100,
            'num_leaves': Num_leaves[idx],
            'n_estimators': Num_estimators[idx],
            'min_data_in_leaf': Min_data_in_leaf[idx],
            'lambda_l2':l2[idx],
            'verbose':-1
            }
        
        Params[f"q{quantile}"]=params

    trainer.train(Params)
    trainer.test()
    
