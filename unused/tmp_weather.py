#%% 查询天气
import requests
model='DWD_ICON-EU'
# As a 6x6 grid
lats = [53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
lons = [1.702, 1.767, 1.832, 1.897, 1.962, 2.027]

variables = 'WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity'

base_url = 'https://api.rebase.energy'
url = f"{base_url}/weather/v2/query"
query_type='grid'

body = {
    'model': model,
    'latitude': lats,
    'longitude': lons,
    'variables': variables,
    'type': query_type,
    'output-format': 'json',
    'forecast-horizon': 'latest'
}

resp = requests.post(url, json=body, headers={'Authorization': rebase_api_client.api_key})
print(resp.status_code)  #Response:200


#%%
import requests
model='DWD_ICON-EU'
# As a 6x6 grid
lats = [53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
lons = [1.702, 1.767, 1.832, 1.897, 1.962, 2.027]

variables = 'WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity'

base_url = 'https://api.rebase.energy'
url = f"{base_url}/weather/v2/query"
query_type='grid'

body = {
    'model': model,
    'latitude': lats,
    'longitude': lons,
    'variables': variables,
    'type': query_type,
    'output-format': 'json',
    'start-date': '2023-08-01',
    'end-date': '2023-08-03',
}

resp = requests.post(url, json=body, headers={'Authorization': rebase_api_client.api_key})
print(resp.status_code)  #Response:401
