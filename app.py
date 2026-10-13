import numpy as np
import pandas as pd
import streamlit as st
from safargo_sim import *

st.set_page_config(page_title="SafarGo RL Dispatch MVP", layout="wide")
st.title("SafarGo - RL Vehicle Repositioning (MVP)")
st.write("A small simulation of 10 zones and 30 vehicles. The Q-Learning agent "
         "decides every 5 minutes where each idle vehicle should go.")

episodes = st.sidebar.slider("Training episodes", 200, 2000, 1000, 100)
n_eval = st.sidebar.slider("Test days", 10, 60, 30, 10)


@st.cache_resource
def get_agent(n):
    return train(n)


if st.button("Train agent and compare with baselines"):
    with st.spinner("Training... (about 1 minute)"):
        agent, curve = get_agent(episodes)
    rng = np.random.default_rng(5)
    res = {
        "Random": evaluate(lambda e, v, s, x: random_policy(e, v, rng), n_eval),
        "Send to busiest zone": evaluate(lambda e, v, s, x: busiest_policy(e, v, rng), n_eval),
        "Stay where you are": evaluate(lambda e, v, s, x: stay_policy(e, v, rng), n_eval),
        "Q-Learning agent": evaluate(lambda e, v, s, x: agent.act(s, 0.0), n_eval),
    }
    st.subheader("Reward curve")
    st.line_chart(pd.Series(curve, name="Episode reward"))

    st.subheader("Results (average per simulated day)")
    df = pd.DataFrame({k: {"Avg wait (min)": v["wait"],
                           "Rides completed": v["rides"],
                           "Fuel use (km)": v["km"],
                           "Cancelled rides": v["cancelled"],
                           "Worst zone service rate": v["min_zone"]}
                       for k, v in res.items()}).T
    st.dataframe(df.round(2))

    st.subheader("Service rate per zone (fairness check)")
    st.bar_chart(pd.DataFrame({k: v["zr"] for k, v in res.items()}, index=ZONES))
    st.caption("Safety fallback: if the agent's reward drops below the baseline, "
               "dispatch switches to the simple rule.")