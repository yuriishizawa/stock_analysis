import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pandas_datareader as pdr
import yfinance as yf

sns.set()

st.write("""
# App para análise de ações
"""
)

tickers = ['VALE3', 'PETR4', 'BPAC3', 'ABEV3', 'ITUB4', 'SANB4', 'AZUL4', 'BBDC4', 'KLBN4', 'BBAS3',
           'ITSA4', 'JBSS3', 'GOLL4', 'SUZB3', 'VIVT3', 'CSNA3', 'ELET3', 'B3SA3', 'WEGE3', 'GGBR4']

for i, j in enumerate(tickers):
    if j.endswith('.SA'):
        pass
    else:
        tickers[i] = j + '.SA'

data = pdr.DataReader(tickers,data_source='yahoo',
                        start='2020-01-01', end='2021-05-13')

data.resample('D').ffill()

df = pd.DataFrame(data['Adj Close'].values, index=data.index, columns = data['Adj Close'].columns.values)

df_melt = df.reset_index().melt(id_vars='Date', var_name='Index', value_name='Preço Ajustado')

def plot_melted(df, yaxis = 'Preço Ajustado',dash = False):
    df_melt = df.reset_index().melt(id_vars='Date', var_name='Symbol', value_name='Preço Ajustado')
    fig = px.line(df_melt, x="Date", y="Preço Ajustado", color="Symbol",labels={'Preço Ajustado': yaxis})
    if dash:
        fig.add_hline(1,line_dash="dash")
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()
    return fig, df_melt

df_return = df.apply(lambda x: x/x[0])
fig1, df_return_melted = plot_melted(df_return, 
                    'Retorno diário acumulado', dash=True)

st.plotly_chart(fig1,use_container_width=True)