"""
Módulo para análise de dados financeiros
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr

from stock_analysis import logger

yf.pdr_override()


def plot_melted(df, yaxis="Preço Ajustado", dash=False):
    """
    Função para plotar gráfico de linhas com múltiplas séries

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados
    yaxis : str
        Nome do eixo y
    dash : bool
        Adiciona linha tracejada em y=1 se True
    Retorno
    -------
    fig : plotly.graph_objects.Figure
        Objeto com o gráfico
    df_melt : pd.DataFrame
        DataFrame melted

    """
    df_melt = df.reset_index().melt(
        id_vars="Date", var_name="Symbol", value_name="Preço Ajustado"
    )
    fig = px.line(
        df_melt,
        x="Date",
        y="Preço Ajustado",
        color="Symbol",
        labels={"Preço Ajustado": yaxis},
    )
    if dash:
        fig.add_hline(1, line_dash="dash")
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


def get_stock_data(tickers, start, end):
    """
    Função para obter dados de ações

    Parâmetros
    ----------
    tickers : list
        Lista de tickers
    start : datetime
        Data inicial
    end : datetime
        Data final
    Retorno
    -------
    df_stock_data : pd.DataFrame
        DataFrame com os dados
    """
    stock_data = pdr.get_data_yahoo(tickers, start=start, end=end)
    stock_data = stock_data.resample("D").ffill()

    df_stock_data = pd.DataFrame(
        stock_data["Adj Close"].values,
        index=stock_data.index,
        columns=stock_data["Adj Close"].columns.values,
    )

    return df_stock_data


def format_tickers_with_suffix(tickers_str):
    """
    Função para formatar tickers com sufixo .SA

    Parâmetros
    ----------
    text1 : str
        String com os tickers
    Retorno
    -------
    tickers : list
        Lista de tickers formatados
    """
    # Remove espaços em branco e faz o split
    tickers = [
        ticker.upper()
        for ticker in tickers_str.replace(" ", "").split(",")
        if ticker != ""
    ]

    # Adiciona sufixo .SA se não tiver, pois é o padrão para ações brasileiras
    for i, j in enumerate(tickers):
        if not j.endswith(".SA"):
            tickers[i] = j + ".SA"
    logger.info(f"Tickers ajustados: {tickers}")
    return tickers


def generate_portifolio(stock_data, max_share=10):
    portifolio = pd.DataFrame(stock_data.T.iloc[:, 0])
    portifolio.columns = ["Preço Ajustado"]

    sectors = []

    for column in portifolio.index:
        sectors.append(yf.Ticker(column).info["sector"])

    portifolio["Setor"] = sectors
    portifolio["Quantidade"] = np.random.randint(1, 1000, size=portifolio.shape[0])
    portifolio["Total Investido"] = (
        portifolio["Preço Ajustado"]
        .mul(portifolio["Quantidade"])
        .apply(lambda x: round(x, 2))
    )
    portifolio["Porc. da Carteira"] = (
        portifolio["Total Investido"].div(portifolio["Total Investido"].sum()).mul(100)
    )

    portifolio["pct_aux"] = np.where(
        portifolio["Porc. da Carteira"] > max_share,
        max_share,
        portifolio["Porc. da Carteira"],
    )

    portifolio["Quantidade"] = (
        (
            portifolio["Quantidade"].mul(
                portifolio["pct_aux"].div(portifolio["Porc. da Carteira"])
            )
        )
        // 100
        * 100
    )

    portifolio["Quantidade"] = np.where(
        portifolio["Quantidade"] < 100, 100, portifolio["Quantidade"]
    )

    portifolio["Total Investido"] = (
        portifolio["Preço Ajustado"]
        .mul(portifolio["Quantidade"])
        .apply(lambda x: round(x, 2))
    )
    portifolio["Porc. da Carteira"] = (
        portifolio["Total Investido"].div(portifolio["Total Investido"].sum()).mul(100)
    )

    return portifolio.drop(columns=["pct_aux"]).reset_index(names=["Ticker"])


def main():
    """
    Função principal para análise de ações brasileiras
    """

    st.sidebar.title(
        """
    Análise de ações brasileiras
    """
    )

    st.title("Análise de ações brasileiras")

    text1 = st.sidebar.text_area(
        "Digite os tickers separados por vírgula e sem espaço",
        value="BBAS3,ITSA4,VALE3,PETR4",
    )

    tickers = format_tickers_with_suffix(text1)

    start = st.sidebar.date_input("Período inicial", value=datetime(2020, 1, 1))
    end = st.sidebar.date_input("Período final")

    stock_data = get_stock_data(tickers, start, end)

    st.write(
        """
    ### **Autor**: Yuri Ishizawa
    ### [Github](http://github.com/yuriishizawa)
    ### [LinkedIn](http://linkedin.com/in/yuriishizawa)
    """
    )

    returns = stock_data.apply(lambda x: x / x[0])

    st.write(
        """
    ## Retornos diários
    """
    )

    pct_returns = stock_data.pct_change().dropna()
    fig3, _ = plot_melted(pct_returns, "Retorno diário")

    st.plotly_chart(fig3, use_container_width=True)

    st.write(
        """
    ## Retornos diários acumulados
    """
    )
    fig1, _ = plot_melted(returns, "Retorno diário acumulado", dash=True)
    st.plotly_chart(fig1, use_container_width=True)

    st.write(
        """
    ## Média de retornos diários vs Volatilidade (Desvio-padrão)
    """
    )

    mean_std = pd.DataFrame(
        pct_returns.mean(), index=pct_returns.mean().index, columns=["Média"]
    )
    mean_std["Desvio-padrão"] = pct_returns.std().values
    fig4 = px.scatter(
        mean_std.reset_index(), x="Desvio-padrão", y="Média", text="index"
    )
    fig4.add_hline(0, line_dash="dash")

    st.plotly_chart(fig4, use_container_width=True)

    st.write(
        """
    ## Análise de clusters hierárquicos das ações correlatas
    """
    )

    ax = ff.create_dendrogram(returns.T, labels=returns.columns)

    st.plotly_chart(ax, use_container_width=True)
