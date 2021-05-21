import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pandas_datareader as pdr
import yfinance as yf

sns.set()

st.title("""
App para análise de ações
"""
)

text1 = st.text_input('Digite os tickers separados por vírgula e sem espaço')

tickers = text1.split(',')

for i, j in enumerate(tickers):
    if j.endswith('.SA'):
        pass
    else:
        tickers[i] = j + '.SA'

start = st.date_input('Período inicial')
end = st.date_input('Período final')
data = pdr.DataReader(tickers,data_source='yahoo',
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

st.write("""
# Gráfico de retornos diários
""")
df_perc_change = df.pct_change().dropna()
fig3, df_perc_change_melted = plot_melted(df_perc_change, 'Retorno diário')

st.plotly_chart(fig3,use_container_width=True)

st.write("""
# Gráfico de retornos diários acumulados
""")
fig1, df_return_melted = plot_melted(df_return, 
                    'Retorno diário acumulado', dash=True)
st.plotly_chart(fig1,use_container_width=True)

st.write("""
# Gráfico de média de retornos diários vs Volatilidade (Desvio-padrão)
""")

mean_std = pd.DataFrame(df_perc_change.mean(),index=df_perc_change.mean().index, columns=['Média'])
mean_std['Desvio-padrão'] = df_perc_change.std().values
fig4 = px.scatter(mean_std.reset_index(), x="Desvio-padrão", y="Média",text='index')
fig4.add_hline(0,line_dash='dash')

st.plotly_chart(fig4, use_container_width=True)

st.write("""
# Análise de clusters hierárquicos das ações correlatas
""")

ax = sns.clustermap(data = df_return,row_cluster=False, metric='correlation',
               figsize=(16,6),yticklabels=False,standard_scale=1,dendrogram_ratio=(.1, .2),cbar_pos=(0, .2, .03, .4))

st.pyplot(ax, use_container_width=False)