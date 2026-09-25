# Streamlit dashboard for demand forecasting and dynamic pricing

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from pricing_engine import (
    load_pricing_data,
    load_demand_model,
    get_pricing_recommendation
)


st.set_page_config(
    page_title="Dynamic Pricing & Demand Forecasting",
    page_icon="📈",
    layout="wide"
)


@st.cache_resource
def load_model():
    return load_demand_model()


@st.cache_data
def load_data():
    return load_pricing_data()


demand_model = load_model()
elasticity_df, pricing_df = load_data()


st.title("Dynamic Pricing & Demand Forecasting")

st.write(
    "Forecast product demand and evaluate different price scenarios "
    "to identify the price with the highest estimated revenue."
)


item_ids = sorted(
    elasticity_df["item_id"].dropna().unique()
)

selected_item = st.selectbox(
    "Select Product",
    item_ids
)


available_stores = sorted(
    elasticity_df.loc[
        elasticity_df["item_id"] == selected_item,
        "store_id"
    ].dropna().unique()
)

selected_store = st.selectbox(
    "Select Store",
    available_stores
)


if st.button("Generate Recommendation"):

    try:
        result = get_pricing_recommendation(
            item_id=selected_item,
            store_id=selected_store,
            demand_model=demand_model,
            elasticity_df=elasticity_df,
            pricing_df=pricing_df
        )

        st.subheader("Pricing Recommendation")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Current Price",
            f"${result['current_price']:.2f}"
        )

        col2.metric(
            "Recommended Price",
            f"${result['recommended_price']:.2f}",
            f"{result['price_change_pct']:.1f}%"
        )

        col3.metric(
            "Price Elasticity",
            f"{result['elasticity']:.2f}"
        )


        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Forecasted Demand",
            f"{result['current_demand']:.2f}"
        )

        col5.metric(
            "Expected Demand",
            f"{result['predicted_demand']:.2f}"
        )

        col6.metric(
            "Expected Revenue",
            f"${result['predicted_revenue']:.2f}",
            f"{result['revenue_improvement_pct']:.2f}%"
        )


        st.divider()

        st.subheader("Price Scenario Analysis")

        scenario_df = pd.DataFrame(
            result["scenarios"]
        )

        scenario_df = scenario_df[
            [
                "price_change_pct",
                "new_price",
                "predicted_demand",
                "revenue_after"
            ]
        ].copy()

        scenario_df.columns = [
            "Price Change (%)",
            "Price",
            "Expected Demand",
            "Expected Revenue"
        ]

        scenario_df = scenario_df.round(2)


        chart_data = scenario_df[
            [
                "Price",
                "Expected Revenue"
            ]
        ].set_index("Price")

        st.line_chart(chart_data)


        st.subheader("Tested Price Scenarios")

        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True
        )


        if result["price_change_pct"] == 0:
            st.success(
                "The current price produced the highest estimated "
                "revenue among the tested price scenarios."
            )
        else:
            st.success(
                f"Recommended price: "
                f"${result['recommended_price']:.2f} "
                f"({result['price_change_pct']:+.1f}%). "
                f"Estimated revenue improvement: "
                f"{result['revenue_improvement_pct']:.2f}%."
            )


    except ValueError as error:
        st.error(str(error))

    except Exception as error:
        st.error(
            "Unable to generate the pricing recommendation. "
            f"Error: {error}"
        )