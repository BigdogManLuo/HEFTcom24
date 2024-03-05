from requests import Session
import pandas as pd
from tqdm import tqdm

def get_particpants():
    url = "https://www.rebase.energy/api/challenges/heftcom2024/leaderboard?type=trading&students=false"
    session = Session()
    resp = session.get(url)
    print(resp)
    data=resp.json()
    
    #创建一个空的序列
    particpants = pd.DataFrame(columns=['participant_id','name'  ])
    for i in data['rows']:
        particpant={
        'participant_id':i['id'],
        'name':i['name']
        }
        particpants = particpants._append(particpant,ignore_index=True)

    return particpants

def get_leaderboard(type="trading",particpant_id=None):

    url=f"https://www.rebase.energy/api/challenges/heftcom2024/score?participant_id={particpant_id}&type={type}"
    session = Session()
    resp = session.get(url)
    data=resp.json()
    return data

if __name__ == "__main__":

    particpants=get_particpants() #参与者信息
    score_data_trading=pd.DataFrame(columns=['market_day','score'])
    score_data_forecasting=pd.DataFrame(columns=['market_day','score'])
    for ids in tqdm(particpants['participant_id']):
    
        particpant_data_trading=get_leaderboard(particpant_id=ids,type="trading")
        particpant_data_forecasting=get_leaderboard(particpant_id=ids,type="forecasting")
        data_trading={
            'market_day':particpant_data_trading['market_days'][-1]['day'],
            'score':particpant_data_trading['market_days'][-1]['score']
        }
        data_forecasting={
            'market_day':particpant_data_forecasting['market_days'][-1]['day'],
            'score':particpant_data_forecasting['market_days'][-1]['score']
        }
        score_data_trading=score_data_trading._append(data_trading,ignore_index=True)
        score_data_forecasting=score_data_forecasting._append(data_forecasting,ignore_index=True)

    score_data_trading["name"]=particpants['name']
    score_data_forecasting["name"]=particpants['name']

    #读取Leaderboard
    Leaderboard_trading=pd.read_csv("leaderboard/Leaderboard_trading.csv",index_col=0,encoding='ISO-8859-1')
    Leaderboard_forecasting=pd.read_csv("leaderboard/Leaderboard_forecasting.csv",index_col=0,encoding='ISO-8859-1')

    #创建Leaderboard,其中market_day为索引
    lateest_market_day=score_data_trading['market_day'][0]
    latest_leaderboard_trading=pd.DataFrame(columns=[lateest_market_day],index=particpants['name'])
    latest_leaderboard_forecasting=pd.DataFrame(columns=[lateest_market_day],index=particpants['name'])
    for i in range(len(particpants)):
        latest_leaderboard_trading[lateest_market_day][i]=score_data_trading['score'][i]
        latest_leaderboard_forecasting[lateest_market_day][i]=score_data_forecasting['score'][i]

    #合并Leaderboard
    Leaderboard_trading=pd.concat([Leaderboard_trading,latest_leaderboard_trading],axis=1)
    Leaderboard_forecasting=pd.concat([Leaderboard_forecasting,latest_leaderboard_forecasting],axis=1)

    #保存Leaderboard
    Leaderboard_trading.to_csv("leaderboard/Leaderboard_trading.csv")
    Leaderboard_forecasting.to_csv("leaderboard/Leaderboard_forecasting.csv")
