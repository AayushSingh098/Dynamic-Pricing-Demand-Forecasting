# Core pricing engine for ML-based demand forecasting and price optimization

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

ELASTICITY_PATH = BASE_DIR / "data" / "processed" / "elasticity_lookup.parquet"
PRICING_DATA_PATH = BASE_DIR / "data" / "processed" / "pricing_features_deployment.parquet"
MODEL_PATH = BASE_DIR / "models" / "demand_forecast_lgbm.pkl"


MODEL_FEATURES = [
    "item_id",
    "store_id",
    "cat_id",
    "sell_price",
    "wday",
    "month",
    "year",
    "week_of_year",
    "is_weekend",
    "snap_active",
    "lag_1_demand",
    "lag_7_demand",
    "rolling_7d_demand",
    "rolling_28d_demand",
    "previous_price",
    "price_change_pct",
    "price_changed"
]

CATEGORICAL_FEATURES = [
    "item_id",
    "store_id",
    "cat_id"
]

PRICE_CHANGES = [
    -20, -15, -10, -5, 0, 5, 10, 15, 20
]


def load_pricing_data():
    elasticity_df = pd.read_parquet(ELASTICITY_PATH)
    pricing_df = pd.read_parquet(PRICING_DATA_PATH)

    return elasticity_df, pricing_df


def load_demand_model():
    return joblib.load(MODEL_PATH)


def get_latest_product_row(item_id, store_id, pricing_df):
    product_history = pricing_df[
        (pricing_df["item_id"] == item_id)
        & (pricing_df["store_id"] == store_id)
    ].sort_values("date")

    if product_history.empty:
        raise ValueError("No product-store history found.")

    return product_history.iloc[-1].copy()


def prepare_model_input(row, demand_model):
    input_data = pd.DataFrame(
        [[row[feature] for feature in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    stored_categories = getattr(
        demand_model.booster_,
        "pandas_categorical",
        None
    )

    if stored_categories:
        for column, categories in zip(
            CATEGORICAL_FEATURES,
            stored_categories
        ):
            input_data[column] = pd.Categorical(
                input_data[column],
                categories=categories
            )
    else:
        for column in CATEGORICAL_FEATURES:
            input_data[column] = input_data[column].astype("category")

    return input_data


def predict_demand_at_price(
    latest_row,
    candidate_price,
    demand_model
):
    scenario_row = latest_row.copy()

    current_price = float(latest_row["sell_price"])

    if current_price <= 0:
        raise ValueError("Current product price must be greater than zero.")

    price_change_pct = (
        candidate_price - current_price
    ) / current_price

    scenario_row["previous_price"] = current_price
    scenario_row["sell_price"] = candidate_price
    scenario_row["price_change_pct"] = price_change_pct
    scenario_row["price_changed"] = int(
        abs(price_change_pct) > 1e-9
    )

    input_data = prepare_model_input(
        scenario_row,
        demand_model
    )

    prediction = demand_model.predict(input_data)[0]

    return max(0.0, float(prediction))


def get_product_elasticity(
    item_id,
    store_id,
    elasticity_df
):
    product = elasticity_df[
        (elasticity_df["item_id"] == item_id)
        & (elasticity_df["store_id"] == store_id)
    ]

    if product.empty:
        raise ValueError(
            "Product-store combination not found in elasticity data."
        )

    return float(product.iloc[0]["final_elasticity"])


def recommend_price(
    item_id,
    store_id,
    demand_model,
    elasticity_df,
    pricing_df
):
    latest = get_latest_product_row(
        item_id,
        store_id,
        pricing_df
    )

    elasticity = get_product_elasticity(
        item_id,
        store_id,
        elasticity_df
    )

    current_price = float(latest["sell_price"])

    if current_price <= 0:
        raise ValueError("Current product price must be greater than zero.")

    scenarios = []

    for change in PRICE_CHANGES:
        candidate_price = current_price * (
            1 + change / 100
        )

        predicted_demand = predict_demand_at_price(
            latest_row=latest,
            candidate_price=candidate_price,
            demand_model=demand_model
        )

        predicted_revenue = (
            candidate_price * predicted_demand
        )

        scenarios.append({
            "price_change_pct": float(change),
            "new_price": float(candidate_price),
            "predicted_demand": float(predicted_demand),
            "revenue_after": float(predicted_revenue)
        })

    current_scenario = next(
        scenario
        for scenario in scenarios
        if scenario["price_change_pct"] == 0
    )

    best_scenario = max(
        scenarios,
        key=lambda scenario: scenario["revenue_after"]
    )

    current_demand = current_scenario["predicted_demand"]
    current_revenue = current_scenario["revenue_after"]

    predicted_demand = best_scenario["predicted_demand"]
    predicted_revenue = best_scenario["revenue_after"]

    if current_revenue > 0:
        revenue_improvement_pct = (
            (predicted_revenue - current_revenue)
            / current_revenue
            * 100
        )
    else:
        revenue_improvement_pct = 0.0

    return {
        "item_id": item_id,
        "store_id": store_id,
        "elasticity": round(elasticity, 3),
        "current_price": round(current_price, 2),
        "recommended_price": round(
            best_scenario["new_price"],
            2
        ),
        "price_change_pct": round(
            best_scenario["price_change_pct"],
            2
        ),
        "current_demand": round(
            current_demand,
            2
        ),
        "predicted_demand": round(
            predicted_demand,
            2
        ),
        "current_revenue": round(
            current_revenue,
            2
        ),
        "predicted_revenue": round(
            predicted_revenue,
            2
        ),
        "revenue_improvement_pct": round(
            revenue_improvement_pct,
            2
        ),
        "scenarios": scenarios
    }


def get_pricing_recommendation(
    item_id,
    store_id,
    demand_model,
    elasticity_df,
    pricing_df
):
    return recommend_price(
        item_id=item_id,
        store_id=store_id,
        demand_model=demand_model,
        elasticity_df=elasticity_df,
        pricing_df=pricing_df
    )