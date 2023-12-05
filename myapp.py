import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from datetime import datetime

import yfinance as yf
yf.pdr_override()
import pandas_datareader as pdr
from pandas_datareader import data as pdr

sns.set()

st.sidebar.title("""
Análise de ações brasileiras
"""
)

st.title("Análise de ações brasileiras")

text1 = st.sidebar.text_area('Digite os tickers separados por vírgula e sem espaço',
                            value="BBAS3,ITSA4,VALE3,PETR4")

tickers = text1.split(',')

for i, j in enumerate(tickers):
    if not j.endswith('.SA'):
        tickers[i] = j + '.SA'

start = st.sidebar.date_input('Período inicial',value=datetime(2020,1,1))
end = st.sidebar.date_input('Período final')
data = pdr.get_data_yahoo(tickers,
                        start=start, end=end)

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
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        )
    )

    return fig, df_melt

st.write("""
### **Autor**: Yuri Ishizawa
### [Github](http://github.com/yuriishizawa)
### [LinkedIn](http://linkedin.com/in/yuriishizawa)
""")

df_return = df.apply(lambda x: x/x[0])

st.write("""
## Retornos diários
""")

df_perc_change = df.pct_change().dropna()
fig3, df_perc_change_melted = plot_melted(df_perc_change, 'Retorno diário')

st.plotly_chart(fig3,use_container_width=True)


st.write("""
## Retornos diários acumulados
""")
fig1, df_return_melted = plot_melted(df_return, 
                    'Retorno diário acumulado', dash=True)
st.plotly_chart(fig1,use_container_width=True)

st.write("""
## Média de retornos diários vs Volatilidade (Desvio-padrão)
""")

mean_std = pd.DataFrame(df_perc_change.mean(),index=df_perc_change.mean().index, columns=['Média'])
mean_std['Desvio-padrão'] = df_perc_change.std().values
fig4 = px.scatter(mean_std.reset_index(), x="Desvio-padrão", y="Média",text='index')
fig4.add_hline(0,line_dash='dash')

st.plotly_chart(fig4, use_container_width=True)

st.write("""
## Análise de clusters hierárquicos das ações correlatas
""")

ax = ff.create_dendrogram(df_return.T, labels=df_return.columns)

st.plotly_chart(ax, use_container_width=False)
