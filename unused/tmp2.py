import comp_utils


rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())

market_index=rebase_api_client.get_variable(day="2024-01-04",variable="market_index")
day_ahead_price=rebase_api_client.get_variable(day="2023-12-01",variable="day_ahead_price")
imbalance_price=rebase_api_client.get_variable(day="2024-01-04",variable="imbalance_price")
wind_total_production=rebase_api_client.get_variable(day="2024-01-04",variable="wind_total_production")
solar_total_production=rebase_api_client.get_variable(day="2024-01-04",variable="solar_total_production")
solar_and_wind_forecast=rebase_api_client.get_variable(day="2023-12-01",variable="solar_and_wind_forecast")


day_ahead_demand_forecast=rebase_api_client.get_day_ahead_demand_forecast()
hornsea_dwd=rebase_api_client.get_hornsea_dwd()