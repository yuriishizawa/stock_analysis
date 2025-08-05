"""
Comprehensive unit tests for stock analysis application
Testing Framework: pytest with unittest.mock
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
import plotly.graph_objects as go

# Import the functions to test (assuming they're in a module called app)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    plot_melted,
    get_stock_data,
    format_tickers_with_suffix,
    generate_portifolio,
    main
)


class TestPlotMelted:
    """Test cases for the plot_melted function"""
    
    def setup_method(self):
        """Setup test data"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.test_df = pd.DataFrame({
            'AAPL': np.random.rand(10) * 100,
            'GOOGL': np.random.rand(10) * 100,
            'MSFT': np.random.rand(10) * 100
        }, index=dates)
        self.test_df.index.name = 'Date'
    
    def test_plot_melted_basic_functionality(self):
        """Test basic functionality of plot_melted"""
        fig, df_melt = plot_melted(self.test_df)
        
        # Check return types
        assert isinstance(fig, go.Figure)
        assert isinstance(df_melt, pd.DataFrame)
        
        # Check melted dataframe structure
        assert 'Date' in df_melt.columns
        assert 'Symbol' in df_melt.columns
        assert 'Preço Ajustado' in df_melt.columns
        assert len(df_melt) == len(self.test_df) * len(self.test_df.columns)
    
    def test_plot_melted_with_custom_yaxis(self):
        """Test plot_melted with custom y-axis label"""
        custom_yaxis = "Custom Price Label"
        fig, df_melt = plot_melted(self.test_df, yaxis=custom_yaxis)
        
        # Check that custom label is used
        assert fig.layout.yaxis.title.text == custom_yaxis
    
    def test_plot_melted_with_dash_line(self):
        """Test plot_melted with dash line enabled"""
        fig, df_melt = plot_melted(self.test_df, dash=True)
        
        # Check that horizontal line is added
        assert len(fig.layout.shapes) > 0 or any('hline' in str(shape) for shape in fig.data)
    
    def test_plot_melted_without_dash_line(self):
        """Test plot_melted without dash line"""
        fig, df_melt = plot_melted(self.test_df, dash=False)
        
        # Verify structure is maintained without dash
        assert isinstance(fig, go.Figure)
        assert isinstance(df_melt, pd.DataFrame)
    
    def test_plot_melted_empty_dataframe(self):
        """Test plot_melted with empty dataframe"""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            plot_melted(empty_df)
    
    def test_plot_melted_single_column(self):
        """Test plot_melted with single column dataframe"""
        single_col_df = self.test_df[['AAPL']]
        fig, df_melt = plot_melted(single_col_df)
        
        assert isinstance(fig, go.Figure)
        assert len(df_melt['Symbol'].unique()) == 1
    
    def test_plot_melted_range_selector_buttons(self):
        """Test that range selector buttons are configured"""
        fig, _ = plot_melted(self.test_df)
        
        rangeselector = fig.layout.xaxis.rangeselector
        assert rangeselector is not None
        assert len(rangeselector.buttons) == 5
        
        # Check specific button configurations
        button_labels = [btn.label for btn in rangeselector.buttons if hasattr(btn, 'label')]
        expected_labels = ['1m', '6m', 'YTD', '1y']
        for label in expected_labels:
            assert label in button_labels


class TestGetStockData:
    """Test cases for the get_stock_data function"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_tickers = ['AAPL', 'GOOGL']
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 12, 31)
    
    @patch('app.yf.download')
    def test_get_stock_data_multiple_tickers(self, mock_download):
        """Test get_stock_data with multiple tickers"""
        # Mock yfinance data structure for multiple tickers
        mock_data = {}
        for ticker in self.test_tickers:
            mock_data[ticker] = {
                'Adj Close': pd.Series(np.random.rand(10) * 100, 
                                     index=pd.date_range('2023-01-01', periods=10))
            }
        mock_download.return_value = mock_data
        
        result = get_stock_data(self.test_tickers, self.start_date, self.end_date)
        
        # Verify function call
        mock_download.assert_called_once_with(
            self.test_tickers, 
            start=self.start_date, 
            end=self.end_date, 
            group_by="ticker", 
            auto_adjust=False
        )
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == self.test_tickers
    
    @patch('app.yf.download')
    def test_get_stock_data_single_ticker(self, mock_download):
        """Test get_stock_data with single ticker"""
        single_ticker = ['AAPL']
        
        # Mock yfinance data structure for single ticker
        mock_data = {
            'Adj Close': pd.Series(np.random.rand(10) * 100, 
                                 index=pd.date_range('2023-01-01', periods=10))
        }
        mock_download.return_value = mock_data
        
        result = get_stock_data(single_ticker, self.start_date, self.end_date)
        
        # Check result structure for single ticker
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == single_ticker
    
    @patch('app.yf.download')
    def test_get_stock_data_resampling(self, mock_download):
        """Test that data is properly resampled daily"""
        mock_data = {
            'AAPL': {
                'Adj Close': pd.Series([100, 101, 102], 
                                     index=pd.DatetimeIndex(['2023-01-01', '2023-01-03', '2023-01-05']))
            }
        }
        mock_download.return_value = mock_data
        
        result = get_stock_data(['AAPL'], self.start_date, self.end_date)
        
        # Check that forward fill has been applied (daily resampling)
        assert isinstance(result, pd.DataFrame)
        # Should have more rows due to daily resampling with forward fill
        assert len(result) >= 3
    
    @patch('app.yf.download')
    def test_get_stock_data_network_error(self, mock_download):
        """Test get_stock_data handles network errors"""
        mock_download.side_effect = Exception("Network error")
        
        with pytest.raises(Exception, match="Network error"):
            get_stock_data(self.test_tickers, self.start_date, self.end_date)
    
    @patch('app.yf.download')
    def test_get_stock_data_empty_response(self, mock_download):
        """Test get_stock_data handles empty response"""
        mock_download.return_value = pd.DataFrame()
        
        with pytest.raises((KeyError, AttributeError)):
            get_stock_data(self.test_tickers, self.start_date, self.end_date)


class TestFormatTickersWithSuffix:
    """Test cases for the format_tickers_with_suffix function"""
    
    @patch('app.logger')
    def test_format_tickers_basic_functionality(self, mock_logger):
        """Test basic ticker formatting"""
        input_str = "BBAS3,ITSA4,VALE3"
        result = format_tickers_with_suffix(input_str)
        
        expected = ["BBAS3.SA", "ITSA4.SA", "VALE3.SA"]
        assert result == expected
        mock_logger.info.assert_called_once()
    
    @patch('app.logger')
    def test_format_tickers_with_spaces(self, mock_logger):
        """Test ticker formatting with spaces"""
        input_str = " BBAS3 , ITSA4 , VALE3 "
        result = format_tickers_with_suffix(input_str)
        
        expected = ["BBAS3.SA", "ITSA4.SA", "VALE3.SA"]
        assert result == expected
    
    @patch('app.logger')
    def test_format_tickers_already_with_suffix(self, mock_logger):
        """Test tickers that already have .SA suffix"""
        input_str = "BBAS3.SA,ITSA4,VALE3.SA"
        result = format_tickers_with_suffix(input_str)
        
        expected = ["BBAS3.SA", "ITSA4.SA", "VALE3.SA"]
        assert result == expected
    
    @patch('app.logger')
    def test_format_tickers_lowercase_input(self, mock_logger):
        """Test ticker formatting with lowercase input"""
        input_str = "bbas3,itsa4,vale3"
        result = format_tickers_with_suffix(input_str)
        
        expected = ["BBAS3.SA", "ITSA4.SA", "VALE3.SA"]
        assert result == expected
    
    @patch('app.logger')
    def test_format_tickers_empty_string(self, mock_logger):
        """Test ticker formatting with empty string"""
        input_str = ""
        result = format_tickers_with_suffix(input_str)
        
        assert result == []
    
    @patch('app.logger')
    def test_format_tickers_single_ticker(self, mock_logger):
        """Test ticker formatting with single ticker"""
        input_str = "BBAS3"
        result = format_tickers_with_suffix(input_str)
        
        expected = ["BBAS3.SA"]
        assert result == expected
    
    @patch('app.logger')
    def test_format_tickers_with_empty_elements(self, mock_logger):
        """Test ticker formatting with empty elements between commas"""
        input_str = "BBAS3,,ITSA4,,"
        result = format_tickers_with_suffix(input_str)
        
        expected = ["BBAS3.SA", "ITSA4.SA"]
        assert result == expected
    
    @patch('app.logger')
    def test_format_tickers_mixed_suffixes(self, mock_logger):
        """Test ticker formatting with mixed suffixes"""
        input_str = "BBAS3.SA,AAPL,VALE3.BO"
        result = format_tickers_with_suffix(input_str)
        
        expected = ["BBAS3.SA", "AAPL.SA", "VALE3.BO.SA"]
        assert result == expected


class TestGeneratePortifolio:
    """Test cases for the generate_portifolio function"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.test_stock_data = pd.DataFrame({
            'BBAS3.SA': np.random.rand(10) * 100,
            'ITSA4.SA': np.random.rand(10) * 100,
            'VALE3.SA': np.random.rand(10) * 100
        }, index=dates)
    
    @patch('app.yf.Ticker')
    def test_generate_portifolio_basic_functionality(self, mock_ticker):
        """Test basic portfolio generation"""
        # Mock sector information
        mock_info = MagicMock()
        mock_info.info = {"sector": "Financial Services"}
        mock_ticker.return_value = mock_info
        
        result = generate_portifolio(self.test_stock_data)
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['Ticker', 'Preço Ajustado', 'Setor', 'Quantidade', 
                          'Total Investido', 'Porc. da Carteira']
        assert list(result.columns) == expected_columns
        assert len(result) == len(self.test_stock_data.columns)
    
    @patch('app.yf.Ticker')
    def test_generate_portifolio_max_share_constraint(self, mock_ticker):
        """Test portfolio generation with max share constraint"""
        mock_info = MagicMock()
        mock_info.info = {"sector": "Technology"}
        mock_ticker.return_value = mock_info
        
        max_share = 5.0
        result = generate_portifolio(self.test_stock_data, max_share=max_share)
        
        # Check that no position exceeds max share
        assert all(result['Porc. da Carteira'] <= max_share + 0.1)  # Small tolerance for rounding
    
    @patch('app.yf.Ticker')
    def test_generate_portifolio_minimum_quantity(self, mock_ticker):
        """Test that minimum quantity is enforced"""
        mock_info = MagicMock()
        mock_info.info = {"sector": "Energy"}
        mock_ticker.return_value = mock_info
        
        result = generate_portifolio(self.test_stock_data)
        
        # Check that all quantities are at least 100
        assert all(result['Quantidade'] >= 100)
    
    @patch('app.yf.Ticker')
    def test_generate_portifolio_percentage_sum(self, mock_ticker):
        """Test that portfolio percentages sum to approximately 100%"""
        mock_info = MagicMock()
        mock_info.info = {"sector": "Healthcare"}
        mock_ticker.return_value = mock_info
        
        result = generate_portifolio(self.test_stock_data)
        
        # Check that percentages sum to approximately 100%
        total_percentage = result['Porc. da Carteira'].sum()
        assert abs(total_percentage - 100.0) < 0.1  # Allow small rounding error
    
    @patch('app.yf.Ticker')
    def test_generate_portifolio_sector_assignment(self, mock_ticker):
        """Test that sectors are properly assigned"""
        sectors = ["Technology", "Financial Services", "Energy"]
        mock_infos = []
        for sector in sectors:
            mock_info = MagicMock()
            mock_info.info = {"sector": sector}
            mock_infos.append(mock_info)
        
        mock_ticker.side_effect = mock_infos
        
        result = generate_portifolio(self.test_stock_data)
        
        # Check that sectors are assigned
        assert all(sector in sectors for sector in result['Setor'])
        assert len(result['Setor'].unique()) <= len(sectors)
    
    @patch('app.yf.Ticker')
    def test_generate_portifolio_single_stock(self, mock_ticker):
        """Test portfolio generation with single stock"""
        single_stock_data = self.test_stock_data[['BBAS3.SA']]
        
        mock_info = MagicMock()
        mock_info.info = {"sector": "Financial Services"}
        mock_ticker.return_value = mock_info
        
        result = generate_portifolio(single_stock_data)
        
        assert len(result) == 1
        assert result.iloc[0]['Porc. da Carteira'] == 100.0
    
    @patch('app.yf.Ticker')
    def test_generate_portifolio_yfinance_error(self, mock_ticker):
        """Test portfolio generation handles yfinance errors"""
        mock_ticker.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            generate_portifolio(self.test_stock_data)


class TestMainFunction:
    """Test cases for the main function"""
    
    @patch('app.st')
    @patch('app.get_stock_data')
    @patch('app.format_tickers_with_suffix')
    @patch('app.plot_melted')
    def test_main_streamlit_components(self, mock_plot_melted, mock_format_tickers, 
                                     mock_get_stock_data, mock_st):
        """Test that main function sets up Streamlit components correctly"""
        # Setup mocks
        mock_format_tickers.return_value = ['BBAS3.SA', 'ITSA4.SA']
        mock_stock_data = pd.DataFrame({'BBAS3.SA': [100, 101], 'ITSA4.SA': [50, 51]})
        mock_get_stock_data.return_value = mock_stock_data
        mock_plot_melted.return_value = (MagicMock(), pd.DataFrame())
        
        # Mock Streamlit components
        mock_st.sidebar.text_area.return_value = "BBAS3,ITSA4"
        mock_st.sidebar.date_input.side_effect = [datetime(2020, 1, 1), datetime(2023, 12, 31)]
        
        # Call main function
        main()
        
        # Verify Streamlit components are called
        mock_st.sidebar.title.assert_called_once()
        mock_st.title.assert_called_once()
        mock_st.sidebar.text_area.assert_called_once()
        assert mock_st.sidebar.date_input.call_count == 2
        assert mock_st.write.call_count >= 3  # Multiple write calls for different sections
        assert mock_st.plotly_chart.call_count >= 3  # Multiple charts
    
    @patch('app.st')
    @patch('app.get_stock_data')
    @patch('app.format_tickers_with_suffix')
    @patch('app.plot_melted')
    @patch('app.ff.create_dendrogram')
    def test_main_data_processing(self, mock_create_dendrogram, mock_plot_melted, 
                                mock_format_tickers, mock_get_stock_data, mock_st):
        """Test data processing logic in main function"""
        # Setup test data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        mock_stock_data = pd.DataFrame({
            'BBAS3.SA': [100, 101, 102, 101, 103],
            'ITSA4.SA': [50, 51, 52, 51, 53]
        }, index=dates)
        
        # Setup mocks
        mock_format_tickers.return_value = ['BBAS3.SA', 'ITSA4.SA']
        mock_get_stock_data.return_value = mock_stock_data
        mock_plot_melted.return_value = (MagicMock(), pd.DataFrame())
        mock_create_dendrogram.return_value = MagicMock()
        
        # Mock Streamlit inputs
        mock_st.sidebar.text_area.return_value = "BBAS3,ITSA4"
        mock_st.sidebar.date_input.side_effect = [datetime(2020, 1, 1), datetime(2023, 12, 31)]
        
        # Call main function
        main()
        
        # Verify data processing functions are called
        mock_format_tickers.assert_called_once_with("BBAS3,ITSA4")
        mock_get_stock_data.assert_called_once()
        
        # Verify plotting functions are called
        assert mock_plot_melted.call_count >= 2  # Called for returns and pct_returns
        mock_create_dendrogram.assert_called_once()
    
    @patch('app.st')
    def test_main_error_handling(self, mock_st):
        """Test main function error handling"""
        # Mock an error in one of the components
        mock_st.sidebar.text_area.side_effect = Exception("Streamlit error")
        
        # Main function should handle errors gracefully
        with pytest.raises(Exception, match="Streamlit error"):
            main()


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple functions"""
    
    @patch('app.yf.download')
    @patch('app.yf.Ticker')
    def test_full_workflow_integration(self, mock_ticker, mock_download):
        """Test complete workflow from ticker input to portfolio generation"""
        # Setup mocks
        mock_data = {
            'BBAS3.SA': {
                'Adj Close': pd.Series([100, 101, 102], 
                                     index=pd.date_range('2023-01-01', periods=3))
            }
        }
        mock_download.return_value = mock_data
        
        mock_info = MagicMock()
        mock_info.info = {"sector": "Financial Services"}
        mock_ticker.return_value = mock_info
        
        # Test workflow
        input_tickers = "bbas3"
        formatted_tickers = format_tickers_with_suffix(input_tickers)
        stock_data = get_stock_data(formatted_tickers, datetime(2023, 1, 1), datetime(2023, 12, 31))
        portfolio = generate_portifolio(stock_data)
        fig, df_melt = plot_melted(stock_data)
        
        # Verify results
        assert formatted_tickers == ["BBAS3.SA"]
        assert isinstance(stock_data, pd.DataFrame)
        assert isinstance(portfolio, pd.DataFrame)
        assert isinstance(fig, go.Figure)
        assert isinstance(df_melt, pd.DataFrame)
    
    def test_edge_case_scenarios(self):
        """Test various edge cases"""
        # Test with empty ticker string
        result = format_tickers_with_suffix("")
        assert result == []
        
        # Test with whitespace only
        result = format_tickers_with_suffix("   ")
        assert result == []
        
        # Test with special characters
        result = format_tickers_with_suffix("BBAS3@#$")
        assert result == ["BBAS3@#$.SA"]


if __name__ == "__main__":
    pytest.main([__file__])