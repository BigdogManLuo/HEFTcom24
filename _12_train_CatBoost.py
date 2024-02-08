from catboost import CatBoostRegressor
from utils import Trainer

if __name__=="__main__":
    
    configs={
        "source":"dwd", #数据来源
        "type":"solar", #预测对象类型
        "Regressor":CatBoostRegressor, #模型类型
        "full":False, #是否使用全量数据
        "model_name":"CatBoost" #模型名称
        }
    trainer=Trainer(**configs)
    
    Params={}
    for idx, quantile in enumerate(range(10,100,10)):
        params={
            'iterations':200, 
            'learning_rate':1e-1,
            'silent':False,
            'loss_function':f'Quantile:alpha={quantile/100}'
        }
        Params[f"q{quantile}"]=params

    trainer.train(Params)
    trainer.test()

