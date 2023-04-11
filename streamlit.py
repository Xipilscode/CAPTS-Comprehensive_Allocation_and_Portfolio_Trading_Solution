# Import dependencies
import streamlit as st
from PIL import Image
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('utils')



# create the navigation menu
def navigation():
    page = st.sidebar.selectbox("Choose a page to continue", ["Home", "Step 1: Capital Allocation", "Step 2: Portfolio Optimization", "Step 3 : GRID Bot"])
    if page == "Home":
        home()
    elif page == "Step 1: Capital Allocation":
        step_1()
    elif page == "Step 2: Portfolio Optimization":
        step_2()
    elif page == "Step 3 : GRID Bot":
        step_3()

# Define home page container 
def home():
    with st.spinner("Loading Home page..."):
         # Header image 
        img_header = Image.open('Capital_Allocation_Optimization/streamlit_front_end_cap_alloc/data/images/main_page_1.jpeg')
        st.image(img_header, width=None)

        # Header name of the project w/description
        st.markdown(
        """
        <h2 style="text-align: center;">
        CAPTS: Comprehensive Allocation and Portfolio Trading Solution</h2>
        
        <div style="text-align:justify;">
        This project aims to provide a comprehensive and integrated solution for 
        portfolio allocation, financial analysis and algorithmic trading of three 
        asset classes (crypto, commodities, stocks), consisting of the following 
        3 steps:
        <br/>
        <br/>
        """,
        unsafe_allow_html=True,
        )

        # Header name of Step 1  w/description
        st.markdown(
        """
        <h3 style="text-align: left;">
        Step 1: Capital Allocation Optimization</h3>
        
        <div style="text-align:justify;">
        The objective of this step is to gather, clean, and analyze data for 3 assets : cryptocurrencies , commodities, and stocks, leveraging financial API such as Yahoo Finance with Pandas. The data will be structured and saved in JSON format, and analyzed and visualized with Numpy and PyViz. The analyzed data will be stored in SQL for future use.
        <br/>
        <br/>
        """,
        unsafe_allow_html=True,
        )
        
        # Header name of Step 2  w/description
        st.markdown(
        """
        <h3 style="text-align: left;">
        Step 2: Machine Learning for Portfolio Optimization </h3>
        
        <div style="text-align:justify;">
        In this step, machine learning algorithms will be applied to analyze and optimize the portfolio. Techniques such as linear regression, decision trees, and clustering will be used to identify patterns and make predictions about future price movements. Financial metrics, such as Sharpe ratio and Sortino ratio, will also be employed to evaluate and optimize the portfolio.
        <br/>
        <br/>
    
        """,
        unsafe_allow_html=True,
        )

        # Header name of Step 3  w/description
        st.markdown(
        """
        <h3 style="text-align: left;">
        Step 3: GRID Bot for Backtesting and Trading </h3>
        
        <div style="text-align:justify;">
        In this step , a GRID bot will be developed for backtesting and bug fixing using a paper trading platform. The bot will use the data collected in the first project and insights from the second project to make trades based on various financial strategies, including mean reversion and trend following. The bot's performance will be optimized through algorithmic trading strategies.
        <br/>
        <br/>
        """,
        unsafe_allow_html=True,
        )



# Step 1 page
def step_1():
    with st.spinner("Loading capital Allocation page..."):
        # Header image 
        img_header = Image.open('Capital_Allocation_Optimization/streamlit_front_end_cap_alloc/data/images/main_page_2.jpeg')
        st.image(img_header, width=None)

        # Header name of Step 1  w/description
        st.markdown(
        """
        <h3 style="text-align: center;">
        Step 1: Capital Allocation Optimization</h3>
        
        <div style="text-align:justify;">
        The objective of this step is to gather, clean, and analyze data for 3 asset classes : cryptocurrencies , commodities, and stocks, leveraging financial API such as Yahoo Finance with Pandas. The data will be structured and saved in JSON format, analyzed and visualized with Numpy and PyViz. The analyzed data will be stored in SQL for future use.
        <br/>
        <br/>
        """,
        unsafe_allow_html=True,
        )
        # Prompt user to input capital allocation amount
        capital_sum = st.text_input("How much capital would you like to allocate?")


        # Prompt user to choose assets in asset classes 
        crypto_selected = st.multiselect("Choose cryptocurencies:", options=['BTC-USD', 'ETH-USD', 'DOGE-USD', 'MATIC-USD', 'AVAX-USD', 'SOL-USD', 'TRX-USD', 'ATOM-USD', 'UNI7083-USD', 'LINK-USD'])
        stocks_selected = st.multiselect("Choose stocks:", options=['AMZN', 'AAPL', 'TSLA', 'GOOGL', 'NVDA', 'MSFT', 'TSM', 'META','XOM', 'LAC'])
        commodities_selected = st.multiselect("Choose commodities:", options=['GC=F', 'SI=F', 'CL=F', 'HG=F', 'LBS=F', 'ZS=F', 'GF=F', 'KE=F', 'CT=F', 'ZR=F'])

        api_pull = {'crypto':crypto_selected
                    ,'stocks':stocks_selected
                    ,'commodities':commodities_selected
                    }


        # Prompt user to choose time period 
        st.write("Choose the analysis period:\n"
        "Note that you can only choose a period starting from Jan 1st, 2020!") 

        # Get the current date
        today = datetime.datetime.now().date()

        # Set the earliest allowed start date to January 1st, 2019
        earliest_start_date = datetime.date(2020, 1, 1)

        # Get the user selected start and end dates
        selected_start_date = st.date_input("Select the start date", earliest_start_date)

        # If the selected start date is earlier than the earliest allowed start date, set it to the earliest start date
        if selected_start_date < earliest_start_date:
            selected_start_date = earliest_start_date

        selected_end_date = st.date_input("Select the end date", today)

        # If the selected end date is later than the current date, set it to the current date
        if selected_end_date > today:
            selected_end_date = today

        # If the selected end date is earlier than the selected start date, set the end date to the start date
        if selected_end_date < selected_start_date:
            selected_end_date = selected_start_date

        # Display the selected dates
        st.write("Selected start date:", selected_start_date)
        st.write("Selected end date:", selected_end_date)

        # Call fetch_asset_data function to pull data from API
        data = fetch_asset_data(api_pull, selected_start_date, selected_end_date)   

        # Call create_price function to get DataFrames for each asset class
        crypto_price_df, stocks_price_df, commodities_price_df = create_price_df(data, api_pull)

        # Display Data Frames of closing pices por each asset class
        st.write('First 5 rows of closing prices for crypto assets:')
        st.dataframe(crypto_price_df.head(), width=500, height=300)

        st.write('First 5 rows of closing prices for stocks:')
        st.dataframe(stocks_price_df.head(), width=500, height=300)

        st.write('First 5 rows of closing prices for commodities:')
        st.dataframe(commodities_price_df.head(), width=500, height=300)

        # Call function to calculate the log returns for each asset class 
        crypto_log_returns, stocks_log_returns, commodities_log_returns = calculate_log_returns(crypto_price_df, stocks_price_df, commodities_price_df)

        # Display Data Frames for logarithmic returns of the assets for chosen time period.
        st.write('First 5 rows of logarithmic returns for crypto assets:')
        st.dataframe(crypto_log_returns.head(), width=500, height=300)

        st.write('First 5 rows of logarithmic returns for stocks:')
        st.dataframe(stocks_log_returns.head(), width=500, height=300)

        st.write('First 5 rows of logarithmic returns for commodities:')
        st.dataframe(commodities_log_returns.head(), width=500, height=300)
        
        # Define the number of tickers selected for each asset class
        crypto_num_selected_tickers = len(api_pull['crypto'])
        stocks_num_selected_tickers = len(api_pull['stocks'])
        commodities_num_selected_tickers = len(api_pull['commodities'])

        # Call function to generate random weights for each asset class
        crypto_random_wts = generate_random_weights(crypto_num_selected_tickers)
        stocks_random_wts = generate_random_weights(stocks_num_selected_tickers)
        commodities_random_wts = generate_random_weights(commodities_num_selected_tickers)

        # Call function to rebalance weights, so each assect class would wave 100% allocation 
        crypto_rebalanced_wts = rebalance_weights(crypto_random_wts)
        stocks_rebalanced_wts = rebalance_weights(stocks_random_wts)
        commodities_rebalanced_wts = rebalance_weights(commodities_random_wts)

        # Call function to calculate expected returns for each asset class and annualize them.
        # Annualize crypto 365, stock and commodities 252 days
        crypto_expected_returns = calculate_expected_returns(crypto_log_returns, crypto_rebalanced_wts)
        stocks_expected_returns = calculate_expected_returns(stocks_log_returns, stocks_rebalanced_wts)
        commodities_expected_returns = calculate_expected_returns(commodities_log_returns, commodities_rebalanced_wts)

        # Call function to calculate expected volatility for each asset
        crypto_expected_volatility = calculate_expected_volatility(crypto_log_returns, crypto_rebalanced_wts)
        stocks_expected_volatility = calculate_expected_volatility(stocks_log_returns, stocks_rebalanced_wts)
        commodities_expected_volatility = calculate_expected_volatility(commodities_log_returns, commodities_rebalanced_wts)

        # Call function to calculate the Sharpe Ratio for each asset class using the calculate_sharpe_ratio function
        crypto_sharpe_ratio = calculate_sharpe_ratio(crypto_expected_returns, crypto_expected_volatility)
        stocks_sharpe_ratio = calculate_sharpe_ratio(stocks_expected_returns, stocks_expected_volatility)
        commodities_sharpe_ratio = calculate_sharpe_ratio(commodities_expected_returns, commodities_expected_volatility)

        # Prompt user to choose number of simulations
        num_of_portfolios = st.slider("Choose number of portfolios simulated:", min_value=500, max_value=5000, step=500)

        # Call the Monte Carlo simulation function for each asset class.
        crypto_all_weights, crypto_ret_arr, crypto_vol_arr, crypto_sharpe_ratio_arr = mc_sim(num_of_portfolios, crypto_log_returns, crypto_num_selected_tickers)
        stocks_all_weights, stocks_ret_arr, stocks_vol_arr, stocks_sharpe_ratio_arr = mc_sim(num_of_portfolios, stocks_log_returns, stocks_num_selected_tickers)
        commodities_all_weights, commodities_ret_arr, commodities_vol_arr, commodities_sharpe_ratio_arr =  mc_sim(num_of_portfolios, commodities_log_returns, commodities_num_selected_tickers)

        # Create data frame with the weights, the returns, the volatility, and the Sharpe Ratio for each asset class
        crypto_simulations_data = [crypto_ret_arr, crypto_vol_arr, crypto_sharpe_ratio_arr, crypto_all_weights]
        stocks_simulations_data = [stocks_ret_arr,  stocks_vol_arr,  stocks_sharpe_ratio_arr,  stocks_all_weights]
        commodities_simulations_data = [commodities_ret_arr,  commodities_vol_arr,  commodities_sharpe_ratio_arr,  commodities_all_weights]

        # Create a DataFrame from sim data and Transpose, so will look like our original one.
        crypto_simulations_df = pd.DataFrame(data=crypto_simulations_data).T
        stocks_simulations_df = pd.DataFrame(data=stocks_simulations_data).T
        commodities_simulations_df = pd.DataFrame(data=commodities_simulations_data).T

        # Give the columns names for crypto
        crypto_simulations_df.columns = [
            'Returns',
            'Volatility',
            'Sharpe Ratio',
            'Portfolio Weights'
        ]
        # Make sure the data types are correct, we don't want our floats to be strings.
        # Infer data types to convert columns with mixed data types to their appropriate data types.
        crypto_simulations_df = crypto_simulations_df.infer_objects()

        st.write('First 5 rows of simulated portfolios for crypto asset class:')
        st.dataframe(crypto_simulations_df.head(), width=500, height=300)





# Step 2 page
def step_2():
    with st.spinner("Loading portfolio Optimization page..."):
        # Header text
        st.markdown(
        """
        <h3 style="text-align: center;">
        Step 2: Machine Learning for Portfolio Optimization</h3>
        <br/>
        <br/>
        Coming soon before the next bull market!
        """,
        unsafe_allow_html=True,
        )
     
        # Header image
        img_header = Image.open('Capital_Allocation_Optimization/streamlit_front_end_cap_alloc/data/images/page_2.png')
        st.image(img_header, width=None)
    

# Step 3 page
def  step_3():
     with st.spinner("Loading GRID Bot page..."):
        # Header text
        st.markdown(
        """
        <h3 style="text-align: center;">
        Step 3: GRID Bot for Backtesting and Tradingn</h3>
        <br/>
        <br/>
        Coming soon!
        """,
        unsafe_allow_html=True,
        )
  
        # Header image
        img_header = Image.open('Capital_Allocation_Optimization/streamlit_front_end_cap_alloc/data/images/bot_page.jpeg')
        st.image(img_header, use_column_width=True)

        st.markdown(
        """
        <div style="text-align:justify;">
        Ah, the trading Grid bot, a financial whiz that puts the "grid" in action.
        <br/>
        Let us break it down for you:
        <br/>
        At its core, a trading Grid bot is a sophisticated algorithm that buys and sells assets automatically according to a predetermined set of rules. The key feature of a Grid bot is that it uses a series of up and down buy and sell orders to capture gains in a sideways market.
        <br/>
        How does it work, you'll ask? Imagine it like a crossword puzzle. The bot fills in the boxes of the puzzle with buy orders, each one slightly lower than the last. Then it fills in the corresponding boxes with sell orders, each one slightly higher than the last. As the market moves sideways, the bot can capture gains by trading within this grid of orders.
        <br/>
        But like any good crossword puzzler, the Grid bot has a few tricks up its sleeve. It can adjust the size and spacing of its grid as the market moves, ensuring it can still capture profits even if the market starts to move more rapidly. And it can also be programmed to automatically reinvest profits back into the grid, compounding gains over time.
        <br/>
        Of course, like any investment strategy, there are risks involved. If the market breaks out of the grid, the bot may suffer losses. But for those willing to take the risk, a Grid bot can be a powerful tool in their trading arsenal.
        <br/>
        So there you have it - the trading Grid bot, a financial strategy that puts the "smart" in "smart investing".
        
        """,
            unsafe_allow_html=True,
        )

# Main function to run the app
def main():
    navigation()
    
# Run the app
if __name__ == "__main__":
    main()



