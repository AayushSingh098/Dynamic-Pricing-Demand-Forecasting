# Test automatic demand forecasting and the pricing recommendation engine

from pricing_engine import (
    load_pricing_data,
    load_demand_model,
    forecast_demand,
    recommend_price
)


elasticity_df, pricing_df = load_pricing_data()
demand_model = load_demand_model()

print("Demand model loaded successfully.")
print("Model type:", type(demand_model).__name__)
print("Number of model features:", demand_model.n_features_in_)


item_id = "HOUSEHOLD_2_505"
store_id = "TX_2"

forecasted_demand = forecast_demand(
    item_id=item_id,
    store_id=store_id,
    demand_model=demand_model,
    pricing_df=pricing_df
)

print("\nFORECASTED DEMAND")
print("-----------------")
print("Product:", item_id)
print("Store:", store_id)
print("Predicted demand:", round(forecasted_demand, 2))


result = recommend_price(
    item_id=item_id,
    store_id=store_id,
    current_demand=forecasted_demand,
    elasticity_df=elasticity_df,
    pricing_df=pricing_df
)

print("\nPRICING RECOMMENDATION")
print("----------------------")

for key, value in result.items():
    print(f"{key}: {value}")