import pandas as pd
import numpy as np
import streamlit as st
import os

CLIENTS_PATH = "saved_data/Base_client.parquet"
APPLICATION_TRAIN_PATH = "saved_data/application_train.csv"

@st.cache_data
def load_all_clients():
    df = pd.read_parquet(CLIENTS_PATH)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    return df

def get_client_list():
    df = load_all_clients()
    return df["SK_ID_CURR"].unique().tolist()

def get_client_data(SK_ID_CURR):
    df = load_all_clients()
    df = df[df["SK_ID_CURR"] == int(SK_ID_CURR)]
    if df.empty:
        return pd.DataFrame()
    return df.drop(columns=["SK_ID_CURR"])

@st.cache_data
def load_raw_applications():
    df = pd.read_csv(APPLICATION_TRAIN_PATH)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    clients_df = load_all_clients()
    ids = clients_df["SK_ID_CURR"].unique()
    return df[df["SK_ID_CURR"].isin(ids)]

def get_raw_client_info(sk_id_curr):
    df = load_raw_applications()
    return df[df["SK_ID_CURR"] == int(sk_id_curr)]

@st.cache_data
def load_application_train():
    df = pd.read_csv(APPLICATION_TRAIN_PATH)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    return df

