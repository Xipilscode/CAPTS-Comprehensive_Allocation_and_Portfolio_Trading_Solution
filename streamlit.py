# Import dependencies
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from PIL import Image
import sys
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import pyfolio as pf
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
import requests

sys.path.append('utils')
from ipywidgets import interactive, interact_manual, RadioButtons, HTML, HBox, VBox, Layout
from PIL import Image
import datetime
import io
import base64

# Set display options for Pandas
 


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
        img_header = Image.open('https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/main_page_1.jpeg')
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
        img_header = Image.open('https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/main_page_2.jpeg')
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
        # capital_sum = st.text_input("How much capital would you like to allocate?")
        

        # Prompt user to choose assets in asset classes 
        crypto_tickers = st.multiselect("Choose cryptocurencies:", options=['AAVE-USD', 'ALGO-USD', 'BAT-USD', 'BCH-USD', 'BTC-USD', 'DAI-USD', 'ETH-USD', 'LINK-USD', 'LTC-USD', 'MATIC-USD', 'MKR-USD', 'NEAR-USD', 'PAXG-USD', 'SOL-USD', 'TRX-USD', 'USDT-USD ', 'WBTC-USD'])
        stocks_tickers = st.multiselect("Choose stocks:", options=['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VO', 'VB', 'VEA', 'VWO', 'XLF ', 'XLV', 'XLE', 'XLY', 'XLC', 'XLK', 'XLI', 'XLP', 'XLB', 'XLU', 'XLRE'])
        commodities_tickers = st.multiselect("Choose commodities:", options=['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC', 'GSG', 'IAU ', 'PPLT', 'SIVR', 'MOO', 'NIB', 'JO', 'JJG', 'WEAT', 'UGA', 'DBE', 'REMX', 'OIL'])
        bonds_tickers = st.multiselect("Choose bonds:", options=['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'JNK', 'MUB', 'TIP', 'BNDX', 'EMB', 'VWOB', 'PFF', 'BKLN', 'FLOT', 'GSY', 'SCHO', 'SCHR', 'SCHZ'])

        # Define the tickers for each asset class.
        api_pull = {'crypto': crypto_tickers,
                    'stocks': stocks_tickers,
                    'commodities': commodities_tickers,
                    'bonds': bonds_tickers}

        # Prompt user to choose time period 
        st.write("Choose the analysis period:\n"
        "Note that you can only choose a period starting from Jan 1st, 2020!") 

        # Get the current date
        today = datetime.datetime.now().date()

        # Set the earliest allowed start date to January 1st, 2019
        earliest_start_date = datetime.date(2020, 10, 14)

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

        # Function to fetch asset data using yfinance
        def fetch_asset_data(api_pull, start_date, end_date):
            data_frames = [] # Initialize an empty list to store data frames

            # Iterate over asset classes and their corresponding tickers
            for asset_class, tickers in api_pull.items():
                for ticker in tickers:
                    # Download historical data for each ticker within the specified date range
                    historical_data = yf.download(ticker, start=start_date, end=end_date)
                    historical_data['Asset Class'] = asset_class # Add a column indicating the asset class
                    historical_data['Ticker'] = ticker # Add a column indicating the ticker
                    data_frames.append(historical_data) # Append the data frame to the list

            # Concatenate all data frames in the list into a single data frame
            combined_data = pd.concat(data_frames)
            return combined_data 


        # Create a run button
        run_button = st.button("Run Analysis")



        if run_button:


            # Call fetch_asset_data function to pull data from API
            data = fetch_asset_data(api_pull, selected_start_date, selected_end_date)           
            
            #Reset the index of a DataFrame and set a new index with multiple columns
            data = data.reset_index().set_index(['Asset Class', 'Ticker','Date'])
            
            #Remane column
            data = data.rename(columns={"Adj Close": "Adj_Close"})

            # Display the loaded data
            st.write("Loaded data:")
            st.write(data.head(10))

            
            # Calculate logarithmic returns for each asset
            def calculate_log_returns(df):
                """
                Calculate the logarithmic returns of an asset.
                If the 'Adj Close' column is present, use it for calculating log. returns.
                Otherwise use the 'Close' column
                """
           
                if 'Adj Close' in df.columns:
                    return np.log(1 + df['Adj Close'].pct_change())    
                else:
                    return np.log(1 + df['Close'].pct_change())
                

            # data.groupby(['Asset Class', 'Ticker'], group_keys=False).apply(o_c_pct_change)
            data["Returns"] = data.groupby(['Asset Class', 'Ticker'], group_keys=False).apply(calculate_log_returns)



                    #Define function 
            # Selects data from the dataframe based on the given asset_class and Returns columns
            def transpose_df(df, asset_class):
                df_data = df.loc[(asset_class),['Returns']].reset_index()

                # Pivot the dataframe with index as 'Ticker', columns as 'Date', and values as 'Returns'
                df_data_pivot = df_data.pivot(index='Ticker',columns='Date', values='Returns')

                # Drop columns containing any NaN values from the pivoted dataframe
                df_data_pivot = df_data_pivot.dropna(axis=1)
                return df_data_pivot


            # Call transpose_df function and ranspose data subset for all asset classes, store in a new dataframe
            df_crypto_data_transposed = transpose_df(data,'crypto')
            df_stocks_data_transposed = transpose_df(data,'stocks')
            df_commodities_data_transposed = transpose_df(data,'commodities')
            df_bonds_data_transposed = transpose_df(data,'bonds')


                    # Define function for PCA
            def perform_pca(df):
                """
                Perform PCA on a given dataframe and return a dataframe of principal components.

                Parameters:
                df (pandas.DataFrame): The dataframe to perform PCA on.

                Returns:
                pandas.DataFrame: A dataframe of principal components.
                """
                # Replace NaN and infinite values with column means
                # df_clean = df.replace([np.inf, -np.inf], np.nan).fillna(df.mean())

                # Choose the number of principal components to explain 90% of the variance
                pca = PCA(n_components=0.9)

                # Fit the PCA model to the cleaned data
                pca.fit(df)

                # Transform the data into principal components
                results = pca.transform(df)

                # Create a dataframe from the principal components
                results_df = pd.DataFrame(results, index=df.index)

                # Return the dataframe of principal components
                return results_df

            # Apply PCA function for all asset classes
            crypto_pca = perform_pca(df_crypto_data_transposed)
            stocks_pca = perform_pca(df_stocks_data_transposed)
            commodities_pca = perform_pca(df_commodities_data_transposed)
            bonds_pca = perform_pca(df_bonds_data_transposed)

            # Add a short explanation of PCA
            st.markdown(
            """
            <h3 style="text-align: center;">
            Principal Component Analysis (PCA)</h3>
            
            <div style="text-align:justify;">
            Our web platform utilizes Principal Component Analysis (PCA) to identify assets with negative correlations,
            ensuring that our portfolio is diversified. By analyzing asset classes and identifying similar behavior patterns among assets,
            we group those that behave similarly in the same category. This process reduces the dimensionality of high-dimensional data while preserving 
            its variance, enabling us to identify principal components that are uncorrelated and orthogonal. By utilizing PCA in our analysis, 
            we provide you with a more accurate and diversified selection of assets.
            """,
            unsafe_allow_html=True,
            )

            st.write('First 5 rows of crypto asset PCA:')
            st.dataframe(crypto_pca.head(), width=500, height=300)
            
            st.write('First 5 rows of stocks asset PCA:')
            st.dataframe(stocks_pca.head(), width=500, height=300)

            st.write('First 5 rows of commodities PCA:')
            st.dataframe(commodities_pca.head(), width=500, height=300)
            
            st.write('First 5 rows of bonds PCA:')
            st.dataframe(bonds_pca.head(), width=500, height=300)
            
            # Define number of clusters
            crypto_num_clusters = len(crypto_tickers)
            stocks_num_clusters = len(stocks_tickers)
            commodities_num_clusters = len(commodities_tickers)
            bonds_num_clusters = len(bonds_tickers)
            

            # Define clusters, pick the best asset from eachj cluster based on sharpe ratio
            def define_clusters(df, df_transpose_returns, num_clusters):  
                # Define an empty list to store the inertias of each KMeans object
                inertias = []
                
                # Iterate over the numbers 1 to num_clusters (inclusive)
                for n in range(1, num_clusters +1):
                    # Instantiate a KMeans object with n clusters
                    kmeans = KMeans(n_clusters=n, random_state=42)
                    # Fit the KMeans object to the DataFrame and calculate the inertia
                    kmeans.fit(df)
                    # Append the inertia of the KMeans object to the list
                    inertias.append(kmeans.inertia_)
                
                # Find the number of clusters that produces the lowest inertia
                best_num_clusters = np.argmin(inertias) + 1
                # Calculate the mean inertia of the KMeans objects
                mean_inertia = np.mean(inertias)
                
                # Iterate over the numbers 1 to num_clusters (inclusive)
                for k in range(1, num_clusters +1 ):
                    # If the inertia of the KMeans object is less than or equal to the mean inertia
                    if inertias[k-1] <= mean_inertia:
                        # Set the number of clusters to k and break the loop
                        best_num_clusters = k
                        break
                
                # Instantiate a KMeans object with the best number of clusters
                kmeans_final = KMeans(n_clusters=best_num_clusters, random_state=42)
                # Fit the KMeans object to the DataFrame
                kmeans_final.fit(df)   
                # Create a DataFrame of cluster labels
                final_df = pd.DataFrame(kmeans_final.predict(df), index=df.index, columns=['Clusters'])
                
                # Calculate the Sharpe ratio of the asset class
                sharpe_ratio = df_transpose_returns.apply(lambda x: x.mean() / x.std(), axis=1)
                
                # Combine the cluster labels with the Sharpe ratios
                sharpe_cluster = pd.concat([final_df, sharpe_ratio], axis=1).rename(columns={0: 'Sharpe_Ratio'})
                
                # Get the best Sharpe ratios for each cluster
                best_cluster_sharpes = sharpe_cluster.groupby('Clusters').max()['Sharpe_Ratio'].to_list()
                
                # Get the tickers for the assets with the best Sharpe ratios
                best_cluster_sharpes_df = sharpe_cluster.loc[sharpe_cluster['Sharpe_Ratio'].isin(best_cluster_sharpes)]
                list_best_tickers = best_cluster_sharpes_df.index.to_list()
                
                # Return the DataFrame of cluster labels, the DataFrame of tickers with the best Sharpe ratios, and the list of tickers
                return sharpe_cluster, best_cluster_sharpes_df, list_best_tickers,     



            # # VAR 1 Call the function
            crypto_sharpe_ratio, best_cluster_crypto, best_crypto_ticekrs = define_clusters(crypto_pca, df_crypto_data_transposed, crypto_num_clusters)
            stocks_sharpe_ratio, best_cluster_stocks, best_stocks_ticekrs = define_clusters(stocks_pca,  df_stocks_data_transposed, stocks_num_clusters)
            commodities_sharpe_ratio, best_cluster_commodities, best_commodities_ticekrs = define_clusters(commodities_pca, df_commodities_data_transposed, commodities_num_clusters)
            bonds_sharpe_ratio, best_cluster_bonds, best_bonds_ticekrs = define_clusters(bonds_pca, df_bonds_data_transposed, bonds_num_clusters)

            st.markdown(
            """
            <h3 style="text-align: center;">
            Assets with highest Sharpe ratio</h3>

            <div style="text-align:justify;">
            Using an automated and comprehensive approach, our web platform identifies assets with the highest potential for returns. 
            This is achieved by selecting assets with the best Sharpe ratios within each cluster through instantiating a KMeans object 
            with the most suitable number of clusters, and utilizing KMeans clustering and PCA-transformed data. Subsequently, individual 
            tickers are chosen based on their highest ratio, enabling us to identify assets with the highest Sharpe ratio in a targeted manner.
            Our algorithm-driven process ensures that you save time and maximize your investment potential.


            """,
            unsafe_allow_html=True,
            )


            st.write('Clusters for crypto asset class:')
            st.dataframe(best_cluster_crypto, width=500, height=300)

            st.write('Clusters for stocks:')
            st.dataframe(best_cluster_stocks, width=500, height=300)

            st.write('Clusters for commodities:')
            st.dataframe(best_cluster_commodities, width=500, height=300)
            
            st.write('Clusters for commodities:')
            st.dataframe(best_cluster_commodities, width=500, height=300)

            #Select the data for the best tickers in each asset class and transpose the resulting dataframes.
            crypto_results = df_crypto_data_transposed.loc[best_crypto_ticekrs].T
            stocks_results = df_stocks_data_transposed.loc[best_stocks_ticekrs].T
            commodities_results = df_commodities_data_transposed.loc[best_commodities_ticekrs].T
            bonds_results = df_bonds_data_transposed.loc[best_bonds_ticekrs].T


            # Define tickers selected for each asset class
            crypto_selected_tickers = crypto_results.columns.tolist()
            stocks_selected_tickers = stocks_results.columns.tolist()
            commodities_selected_tickers = commodities_results.columns.tolist()
            bonds_selected_tickers = bonds_results.columns.tolist()


            # Calculate the number of tickers selected for each asset class
            crypto_num_selected_tickers = len(crypto_selected_tickers)
            stocks_num_selected_tickers = len(stocks_selected_tickers)
            commodities_num_selected_tickers = len(commodities_selected_tickers)
            bonds_num_selected_tickers = len(bonds_selected_tickers)


            st.markdown("""
            <h3 style="text-align: center;">
            Perform Monte Carlo Simulations</h3>

            <div style="text-align:justify;">
            This step generates portfolios using Monte Carlo simulation, calculates their expected returns, volatilities, 
            and Sharpe ratios for each asset class, and returns arrays containing these metrics for all generated portfolios.

            """,
            unsafe_allow_html=True,
            )



            # Define trading days constant
            CRYPTO_TRADING_DAYS = 365
            ETFS_TRADING_DAYS = 252

                
            # # Prompt user to choose number of simulations for Streamlit !!!
            # num_of_portfolios = st.slider("Choose number of portfolios simulated:", min_value=500, max_value=10000, step=500)   

            #Define number of simulations(portfolios generated by Monte Carlo)
            num_of_portfolios = 5000

            def monte_carlo_simulation(num_of_portfolios, num_selected_tickers, results):
                """
                This function performs a Monte Carlo simulation for generating portfolios and calculating their
                expected returns, volatilities, and Sharpe ratios for a single asset class.
                
                Parameters:
                - num_of_portfolios (int): Number of portfolios to be generated by the Monte Carlo simulation.
                - num_selected_tickers (int): Number of selected tickers for the asset class.
                - results (DataFrame): DataFrame containing results for the asset class.
                
                Returns:
                - Tuple of 4 NumPy arrays: weights, returns, volatilities, and Sharpe ratios for the asset class.
                """
                if results is crypto_results or 'Crypto' in results.columns or 'Crypto' in results.columns:
                    trading_days = CRYPTO_TRADING_DAYS
                else:
                    trading_days = ETFS_TRADING_DAYS 
                    
                # Initialize arrays to store weights, returns, volatilities, and Sharpe ratios
                all_weights = np.zeros((num_of_portfolios, num_selected_tickers))
                ret_arr = np.zeros(num_of_portfolios)
                vol_arr = np.zeros(num_of_portfolios)
                sharpe_ratio_arr = np.zeros(num_of_portfolios)

                # Start the simulations
                for ind in range(num_of_portfolios):
                    # Calculate random weights and normalize them
                    weights = np.array(np.random.random(num_selected_tickers))
                    weights /= np.sum(weights)

                    # Store the weights in the corresponding array
                    all_weights[ind, :] = weights

                    # Calculate expected returns and store them in the corresponding array
                    ret_arr[ind] = np.sum((results.mean() * weights) * trading_days)

                    # Calculate the volatility and store it in the corresponding array
                    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(results.cov() * trading_days, weights)))

                    # Calculate Sharpe Ratio and store it in the corresponding array
                    sharpe_ratio_arr[ind] = ret_arr[ind] / vol_arr[ind]

                return all_weights, ret_arr, vol_arr, sharpe_ratio_arr    


            # Call the Monte Carlo simulation function for each asset class
            crypto_all_weights, crypto_ret_arr, crypto_vol_arr, crypto_sharpe_ratio_arr = monte_carlo_simulation(num_of_portfolios, crypto_num_selected_tickers, crypto_results)
            stocks_all_weights, stocks_ret_arr, stocks_vol_arr, stocks_sharpe_ratio_arr = monte_carlo_simulation(num_of_portfolios, stocks_num_selected_tickers, stocks_results)
            commodities_all_weights, commodities_ret_arr, commodities_vol_arr, commodities_sharpe_ratio_arr = monte_carlo_simulation(num_of_portfolios, commodities_num_selected_tickers, commodities_results)
            bonds_all_weights, bonds_ret_arr, bonds_vol_arr, bonds_sharpe_ratio_arr = monte_carlo_simulation(num_of_portfolios, bonds_num_selected_tickers, bonds_results)

            # Create data frame with the weights, the returns, the volatility, and the Sharpe Ratio for each asset class
            crypto_simulations_data = [crypto_ret_arr, crypto_vol_arr, crypto_sharpe_ratio_arr, crypto_all_weights]
            stocks_simulations_data = [stocks_ret_arr, stocks_vol_arr, stocks_sharpe_ratio_arr, stocks_all_weights]
            commodities_simulations_data = [commodities_ret_arr, commodities_vol_arr, commodities_sharpe_ratio_arr, commodities_all_weights]
            bonds_simulations_data = [bonds_ret_arr, bonds_vol_arr, bonds_sharpe_ratio_arr, bonds_all_weights]


            # Create a DataFrame from sim data and Transpose, so it will look like our original one.
            crypto_simulations_df = pd.DataFrame(data=crypto_simulations_data).T
            stocks_simulations_df = pd.DataFrame(data=stocks_simulations_data).T
            commodities_simulations_df = pd.DataFrame(data=commodities_simulations_data).T
            bonds_simulations_df = pd.DataFrame(data=bonds_simulations_data).T        


            def process_simulations_df(simulations_df):
                """
                This function processes the simulations data for a single asset class.
                
                Parameters:
                - simulations_df (DataFrame): DataFrame containing the simulations data for the asset class.
                
                Returns:
                - Tuple of two Series: one for the portfolio with the maximum Sharpe Ratio, and another for the portfolio with the minimum volatility.
                """
                # Give the columns names
                simulations_df.columns = [
                    'Returns',
                    'Volatility',
                    'Sharpe Ratio',
                    'Portfolio Weights'
                ]

                # Make sure the data types are correct
                simulations_df = simulations_df.infer_objects()
                
                # Find the Max Sharpe Ratio to find a better portfolio that provides the largest risk-adjusted returns
                max_sharpe_ratio = simulations_df.loc[simulations_df['Sharpe Ratio'].idxmax()]

                # Find the minimum volatility from the simulations to identify a portfolio that takes on the least amount of risk
                min_volatility = simulations_df.loc[simulations_df['Volatility'].idxmin()]

                # Create Series for the rows with the maximum Sharpe Ratio and the minimum volatility
                max_sharpe_ratio_row = pd.Series({
                    'Returns': max_sharpe_ratio['Returns'],
                    'Volatility': max_sharpe_ratio['Volatility'],
                    'Sharpe Ratio': max_sharpe_ratio['Sharpe Ratio'],
                    'Portfolio Weights': max_sharpe_ratio['Portfolio Weights']
                })

                min_volatility_row = pd.Series({
                    'Returns': min_volatility['Returns'],
                    'Volatility': min_volatility['Volatility'],
                    'Sharpe Ratio': min_volatility['Sharpe Ratio'],
                    'Portfolio Weights': min_volatility['Portfolio Weights']
                })
                
                return max_sharpe_ratio_row, min_volatility_row, simulations_df    

            #Call the process_simulations_df() function for all 4 asset classes:
            crypto_max_sharpe_ratio, crypto_min_volatility, crypto_processed_df = process_simulations_df(crypto_simulations_df)
            stocks_max_sharpe_ratio, stocks_min_volatility, stocks_processed_df = process_simulations_df(stocks_simulations_df)
            commodities_max_sharpe_ratio, commodities_min_volatility, commodities_processed_df = process_simulations_df(commodities_simulations_df)
            bonds_max_sharpe_ratio, bonds_min_volatility, bonds_processed_df = process_simulations_df(bonds_simulations_df)




            # Create Monte Carlo simulation scatter plot
            def create_scatter_plot(df):
                # Create a scatter plot.
                fig = px.scatter(
                    df,
                    x='Volatility',
                    y='Returns',
                    color='Sharpe Ratio',
                    color_continuous_scale=['#F6D55C', '#3CAEA3', '#16425B'],
                    size_max=15,
                    hover_name=df.index
                )

                # Get the name of the DataFrame and update the plot title.
                df_name = str(df.index.name).split()[0]
                fig.update_layout(
                    xaxis_title="Standard Deviation",
                    yaxis_title="Returns",
                    hovermode="closest",
                    coloraxis_colorbar=dict(title="Sharpe Ratio")
                )

                # Plot the Max Sharpe Ratio, using a `Red Star`.
                max_sharpe_ratio = df.loc[df['Sharpe Ratio'].idxmax()]
                fig.add_trace(
                    go.Scatter(
                        x=[max_sharpe_ratio['Volatility']],
                        y=[max_sharpe_ratio['Returns']],
                        mode='markers',
                        marker=dict(symbol='star', color='red', size=12),
                        showlegend=False,
                        hovertext=['Max Sharpe Ratio']
                    )
                )

                # Plot the Min Volatility, using a `Yellow Star`.
                min_volatility = df.loc[df['Volatility'].idxmin()]
                fig.add_trace(
                    go.Scatter(
                        x=[min_volatility['Volatility']],
                        y=[min_volatility['Returns']],
                        mode='markers',
                        marker=dict(symbol='star', color='yellow', size=12),
                        showlegend=False,
                        hovertext=['Min Volatility']
                    )
                )

                # Display the plot in Streamlit app
                st.plotly_chart(fig)

            # Create scatter plot for crypto.
            st.write('Portfolio Returns Vs. Risk for crypto')
            create_scatter_plot(crypto_processed_df)


            # Create scatter plot for stocks.
            st.write('Portfolio Returns Vs. Risk for stocks')
            create_scatter_plot(stocks_processed_df)


            # Create scatter plot for commodities.
            st.write('Portfolio Returns Vs. Risk for commodities')
            create_scatter_plot(commodities_processed_df)


            # Create scatter plot for bonds.
            st.write('Portfolio Returns Vs. Risk for bonds')
            create_scatter_plot(bonds_processed_df)

            
            def create_portfolio_weights_df(selected_tickers, max_sharpe_ratio, min_volatility):
                """
                This function creates a DataFrame with the portfolio weights for the selected tickers for both the
                maximum Sharpe ratio and minimum volatility portfolios for a single asset class.
                
                Parameters:
                - selected_tickers (list): List of selected tickers for the asset class.
                - max_sharpe_ratio (Series): Series with the maximum Sharpe ratio portfolio weights for the asset class.
                - min_volatility (Series): Series with the minimum volatility portfolio weights for the asset class.
                
                Returns:
                - Tuple of two DataFrames: one for the maximum Sharpe ratio portfolio weights and another for the minimum volatility
                portfolio weights.
                """
                
                # Create a dictionary with the portfolio weights for each portfolio
                portfolio_weights_dict = {
                    'Max Sharpe Ratio': max_sharpe_ratio,
                    'Min Volatility': min_volatility
                }
                
                # Create empty dictionaries for the portfolio weights
                max_sharpe_ratio_weights = {}
                min_volatility_weights = {}
                
                # Iterate over each portfolio
                for portfolio_name, portfolio_weights in portfolio_weights_dict.items():
                    
                    # Create an empty dictionary to store the portfolio weights for each ticker
                    ticker_weights = {}
                    
                    # Iterate over each ticker
                    for i, ticker in enumerate(selected_tickers):
                        
                        # Get the weight for the ticker from the portfolio weights
                        ticker_weight = portfolio_weights[i]
                        
                        # Add the ticker weight to the ticker_weights dictionary
                        ticker_weights[ticker] = ticker_weight
                    
                    # Convert the ticker_weights dictionary to a Series and add it to the corresponding weights dictionary
                    if portfolio_name == 'Max Sharpe Ratio':
                        max_sharpe_ratio_weights = pd.Series(ticker_weights, name=portfolio_name)
                    else:
                        min_volatility_weights = pd.Series(ticker_weights, name=portfolio_name)
                
                # Create DataFrames from the weights dictionaries
                max_sharpe_ratio_weights_df = pd.DataFrame(max_sharpe_ratio_weights)
                min_volatility_weights_df = pd.DataFrame(min_volatility_weights)
                
                # # Add a column to the DataFrames for the total weight of the portfolio
                # max_sharpe_ratio_weights_df['Total Weight'] = max_sharpe_ratio_weights_df.sum(axis=1)
                # min_volatility_weights_df['Total Weight'] = min_volatility_weights_df.sum(axis=1)
                
                # # Add a column to the DataFrames for the weight of each ticker as a percentage of the total portfolio weight
                # max_sharpe_ratio_weights_df = max_sharpe_ratio_weights_df.apply(lambda x: x/max_sharpe_ratio_weights_df['Total Weight'])
                # min_volatility_weights_df = min_volatility_weights_df.apply(lambda x: x/min_volatility_weights_df['Total Weight'])
                
                return max_sharpe_ratio_weights_df, min_volatility_weights_df




            # Create DataFrame with portfolio weights for each asset class
            crypto_max_sharpe_ratio_weights, crypto_min_volatility_weights = create_portfolio_weights_df(crypto_selected_tickers, crypto_max_sharpe_ratio['Portfolio Weights'], crypto_min_volatility['Portfolio Weights'])
            stocks_max_sharpe_ratio_weights, stocks_min_volatility_weights = create_portfolio_weights_df(stocks_selected_tickers, stocks_max_sharpe_ratio['Portfolio Weights'], stocks_min_volatility['Portfolio Weights'])
            commodities_max_sharpe_ratio_weights, commodities_min_volatility_weights = create_portfolio_weights_df(commodities_selected_tickers, commodities_max_sharpe_ratio['Portfolio Weights'], commodities_min_volatility['Portfolio Weights'])
            bonds_max_sharpe_ratio_weights, bonds_min_volatility_weights = create_portfolio_weights_df(bonds_selected_tickers, bonds_max_sharpe_ratio['Portfolio Weights'], bonds_min_volatility['Portfolio Weights'])

            #Assigns the maximum Sharpe ratio weights for all assets
            crypto_max_weights = crypto_max_sharpe_ratio.loc['Portfolio Weights']
            stocks_max_weights = stocks_max_sharpe_ratio.loc['Portfolio Weights']
            commodities_max_weights = commodities_max_sharpe_ratio.loc['Portfolio Weights']
            bonds_max_weights = bonds_max_sharpe_ratio.loc['Portfolio Weights']    


            #Assigns the Min volatility weights for all assets
            crypto_min_vol_weights = crypto_min_volatility.loc['Portfolio Weights']
            stocks_min_vol_weights = stocks_min_volatility.loc['Portfolio Weights']
            commodities_min_vol_weights = commodities_min_volatility.loc['Portfolio Weights']
            bonds_min_vol_weights = bonds_min_volatility.loc['Portfolio Weights']


            # Assign weights to asset classes for max sharpe ratio portfolio 
            crypto_max_comb_df = crypto_results @ crypto_max_weights
            stocks_max_comb_df = stocks_results @ stocks_max_weights
            commodities_max_comb_df = commodities_results @ commodities_max_weights
            bonds_max_comb_df = bonds_results @ bonds_max_weights
    
            # Assign weights to asset classes for min volatility portfolio
            crypto_min_comb_df = crypto_results @ crypto_min_vol_weights
            stocks_min_comb_df = stocks_results @ stocks_min_vol_weights
            commodities_min_comb_df = commodities_results @ commodities_min_vol_weights
            bonds_min_comb_df = bonds_results @ bonds_min_vol_weights


            # Concatenate the four asset classes dataframes for the maximum Sharpe ratio portfolio
            max_comb_df = pd.concat([crypto_max_comb_df, stocks_max_comb_df, commodities_max_comb_df, bonds_max_comb_df], axis=1)

            # Add prefixes to the column names to signify the corresponding asset class
            max_comb_df.columns = ['Crypto', 'Stocks', 'Commodities', 'Bonds']

            # Concatenate the four asset class dataframes for the minimum volatility portfolio
            min_comb_df = pd.concat([crypto_min_comb_df, stocks_min_comb_df, commodities_min_comb_df, bonds_min_comb_df], axis=1)

            # Add prefixes to the column names to signify the corresponding asset class
            min_comb_df.columns = ['Crypto', 'Stocks', 'Commodities', 'Bonds']

            # Fill NaN values in a DataFrame using forward-fill method.
            max_comb_df = max_comb_df.fillna(method='ffill')
            min_comb_df = min_comb_df.fillna(method='ffill')

            # Define the number of selected asset classes as number of columns 
            max_comb_num_selected_tickers = max_comb_df.shape[1]
            min_comb_num_selected_tickers = min_comb_df.shape[1]

            # Define the df that will be passed to MC simulation
            max_comb_results = max_comb_df
            min_comb_results = min_comb_df

            # Perform Monte Carlo simulation for maximum Sharpe ratio portfolio
            max_comb_all_weights, max_comb_ret_arr, max_comb_vol_arr, max_comb_sharpe_ratio_arr = monte_carlo_simulation(num_of_portfolios, max_comb_num_selected_tickers, max_comb_results)

            # Perform Monte Carlo simulation for minimum volatility portfolio
            min_comb_all_weights, min_comb_ret_arr, min_comb_vol_arr, min_comb_sharpe_ratio_arr = monte_carlo_simulation(num_of_portfolios, min_comb_num_selected_tickers, min_comb_results)


            # Create data frame with the weights, the returns, the volatility, and the Sharpe Ratio arrays
            # for minimum volatility and maximun sharpe ratio portfoilos for asset classes
            max_comb_sim_data = [max_comb_ret_arr, max_comb_vol_arr, max_comb_sharpe_ratio_arr, max_comb_all_weights]
            min_comb_sim_data = [min_comb_ret_arr, min_comb_vol_arr, min_comb_sharpe_ratio_arr, min_comb_all_weights]


            # Create a DataFrame from sim data and Transpose, so it will look like our original one.
            max_comb_simulations_df = pd.DataFrame(data=max_comb_sim_data).T
            min_comb_simulations_df = pd.DataFrame(data=min_comb_sim_data).T


            #Call the process_simulations_df() function for minimum volatility and maximun sharpe ratio portfoilos for asset classes
            max_comb_max_sharpe_ratio, max_comb_min_volatility, max_comb_processed_df = process_simulations_df(max_comb_simulations_df)
            min_comb_max_sharpe_ratio, min_comb_min_volatility, min_comb_processed_df = process_simulations_df(min_comb_simulations_df)

            # Define list of asset classes for allocation 
            max_comb_selected_tickers = max_comb_df.columns.tolist()
            min_comb_selected_tickers = min_comb_df.columns.tolist()
            
            # Create DataFrame with portfolio weights for each asset class
            max_comb_sharpe_ratio_weights, max_comb_min_volatility_weights = create_portfolio_weights_df(max_comb_selected_tickers, max_comb_max_sharpe_ratio['Portfolio Weights'], max_comb_min_volatility['Portfolio Weights'])
            min_comb_sharpe_ratio_weights, min_comb_min_volatility_weights = create_portfolio_weights_df(min_comb_selected_tickers, min_comb_max_sharpe_ratio['Portfolio Weights'], min_comb_min_volatility['Portfolio Weights'])


            max_sharpe_portfolio = {
                "asset_class_weights": max_comb_sharpe_ratio_weights.T,
                "asset_classes": {
                    "crypto": crypto_max_sharpe_ratio_weights,
                    "stocks": stocks_max_sharpe_ratio_weights,
                    "commodities": commodities_max_sharpe_ratio_weights,
                    "bonds": bonds_max_sharpe_ratio_weights
                }
            }

            min_volatility_portfolio = {
                "asset_class_weights": min_comb_sharpe_ratio_weights.T,
                "asset_classes" : {
                        'crypto': crypto_min_volatility_weights,
                        'stocks': stocks_min_volatility_weights,
                        'commodities': commodities_min_volatility_weights,
                        'bonds': bonds_min_volatility_weights
                }
            }


            def calculate_final_allocation(portfolio):
                final_allocation = {}
                asset_class_weights = portfolio["asset_class_weights"]
                
                for asset_class, ticker_weights in portfolio["asset_classes"].items():
                    # Match the asset class key with the corresponding column name
                    asset_class_column = asset_class.title()
                    
                    # Calculate the final allocation for each ticker by multiplying the ticker weight
                    # with the respective asset class weight
                    asset_class_weight = asset_class_weights.loc[:, asset_class_column].values[0]
                    final_ticker_weights = ticker_weights.multiply(asset_class_weight, axis=0)
                    # Add the final ticker weights to the final_allocation dictionary
                    final_allocation[asset_class] = final_ticker_weights

                return final_allocation
            
            max_sharpe_final_allocation = calculate_final_allocation(max_sharpe_portfolio)
            min_volatility_final_allocation = calculate_final_allocation(min_volatility_portfolio)

            # Maximum Sharpe Ratio portfolio DataFrames.
            max_sharpe_crypto_df = max_sharpe_final_allocation['crypto']
            max_sharpe_stocks_df = max_sharpe_final_allocation['stocks']
            max_sharpe_commodities_df = max_sharpe_final_allocation['commodities']
            max_sharpe_bonds_df = max_sharpe_final_allocation['bonds']

            # Minimum Volatility portfolio DataFrames
            min_volatility_crypto_df = min_volatility_final_allocation['crypto']
            min_volatility_stocks_df = min_volatility_final_allocation['stocks']
            min_volatility_commodities_df = min_volatility_final_allocation['commodities']
            min_volatility_bonds_df = min_volatility_final_allocation['bonds']

            def format_allocation(df, old_column, new_column):
                """
                This function formats the allocation values in a given DataFrame by renaming the specified column, converting
                the values to percentages, and rounding them to 4 decimal places.
                
                Parameters:
                df (pd.DataFrame): The input DataFrame containing the allocation values.
                old_column (str): The name of the column in the input DataFrame to be renamed.
                new_column (str): The new name for the column after renaming.
                
                Returns:
                pd.DataFrame: A new DataFrame with the specified column renamed and the allocation values formatted as percentages.
                """
                
                # Create a copy of the input DataFrame to avoid modifying the original data
                formatted_df = df.copy()
                
                # Rename the specified column
                formatted_df = formatted_df.rename(columns={old_column: new_column})
                
                # Convert the values in the renamed column to percentages and round them to 4 decimal places
                formatted_df[new_column] = formatted_df[new_column].apply(lambda x: f'{x * 100:1.4f}%')
                
                # Return the formatted DataFrame
                return formatted_df            


            # Format the allocation values for each asset class by renaming the 'Max Sharpe Ratio'
            # column and converting the allocation values to percentages
            max_sharpe_crypto_form_df = format_allocation(
                max_sharpe_final_allocation['crypto'].copy(),
                'Max Sharpe Ratio',
                'Portfolio Allocation'
            )

            max_sharpe_stocks_form_df = format_allocation(
                max_sharpe_final_allocation['stocks'].copy(),
                'Max Sharpe Ratio',
                'Portfolio Allocation'
            )

            max_sharpe_commodities_form_df = format_allocation(
                max_sharpe_final_allocation['commodities'].copy(),
                'Max Sharpe Ratio',
                'Portfolio Allocation'
            )

            max_sharpe_bonds_form_df = format_allocation(
                max_sharpe_final_allocation['bonds'].copy(),
                'Max Sharpe Ratio',
                'Portfolio Allocation'
            )

            


            # st.markdown("""

            # <h3 style="text-align: center;">
            # Create portfolio weights for each ticker in asset class</h3>

            # <div style="text-align:justify;">
            # This function creates a DataFrame with the portfolio weights for the selected tickers for both the
            # maximum Sharpe ratio and minimum volatility portfolios for a single asset class.


            # """,
            # unsafe_allow_html=True,
            # )


            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for Crypto Max Sharpe ratio portfolio:')
            # st.dataframe(crypto_max_sharpe_ratio_weights, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for Stocks Max Sharpe ratio portfolio:')
            # st.dataframe(stocks_max_sharpe_ratio_weights, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for Commodities Max Sharpe ratio portfolio:')
            # st.dataframe(commodities_max_sharpe_ratio_weights, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for Bonds Max Sharpe ratio portfolio:')
            # st.dataframe(bonds_max_sharpe_ratio_weights, width=500, height=300)


            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for Crypto Minimum volatility portfolio:')
            # st.dataframe(crypto_min_volatility_weights, width=500, height=300)


            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for Stocks Minimum volatility portfolio:')
            # st.dataframe(stocks_min_volatility_weights, width=500, height=300)


            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for Commodities Minimum volatility portfolio:')
            # st.dataframe(commodities_min_volatility_weights, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Weights for BondsMinimum volatility portfolio:')
            # st.dataframe(bonds_min_volatility_weights, width=500, height=300)

            st.markdown(
                """
                <h4 style="text-align: center;">
                Final allocation weights for Maximum Sharpe Ratio Portfolio</h4>

                """, 
                unsafe_allow_html=True
            )            



            # Create an HTML table to arrange the DataFrames side by side
            table = """
            <table>
                <tr>
                    <td>{0}</td>
                    <td>{1}</td>
                </tr>
                <tr>
                    <td>{2}</td>
                    <td>{3}</td>
                </tr>
            </table>
            """.format(
                max_sharpe_crypto_form_df.to_html(),
                max_sharpe_stocks_form_df.to_html(),
                max_sharpe_commodities_form_df.to_html(),
                max_sharpe_bonds_form_df.to_html()
            )


            st.markdown(table, unsafe_allow_html=True)
            # Display the HTML table with the DataFrames in Streamlit

            

            def create_pie_chart(df, title):
                fig = go.Figure()

                fig.add_trace(go.Pie(
                    labels=df.index,
                    values=df.iloc[:, 0],
                    hole=0.7,
                    marker=dict(colors=['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'magenta', 'cyan'])
                ))

                fig.update_layout(
                    title_text=title,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="black"
                    ),
                    showlegend=True
                )

                # Display the plot in Streamlit app
                st.plotly_chart(fig)

            # Create scatter pie plot for crypto.
            create_pie_chart(max_sharpe_crypto_df, 'Maximum Sharpe Ratio portfolio for crypto assets')

            # Create scatter pie plot for crypto.
            create_pie_chart(max_sharpe_stocks_df, 'Maximum Sharpe Ratio portfolio for stock assets')

            # Create scatter pie plot for crypto.
            create_pie_chart(max_sharpe_commodities_df, 'Maximum Sharpe Ratio portfolio for commodities')

            # Create scatter pie plot for crypto.
            create_pie_chart(max_sharpe_bonds_df, 'Maximum Sharpe Ratio portfolio for bonds')

            # Format the allocation values for each asset class in the Minimum Volatility portfolio by renaming the 'Min Volatility'
            # column and converting the allocation values to percentages
            min_volatility_crypto_form_df = format_allocation(
                min_volatility_final_allocation['crypto'].copy(),
                'Min Volatility',
                'Portfolio Allocation'
            )

            min_volatility_stocks_form_df = format_allocation(
                min_volatility_final_allocation['stocks'].copy(),
                'Min Volatility',
                'Portfolio Allocation'
            )

            min_volatility_commodities_form_df = format_allocation(
                min_volatility_final_allocation['commodities'].copy(),
                'Min Volatility',
                'Portfolio Allocation'
            )

            min_volatility_bonds_form_df = format_allocation(
                min_volatility_final_allocation['bonds'].copy(),
                'Min Volatility',
                'Portfolio Allocation'
            )    

            st.markdown(
                """
                <h4 style="text-align: center;">
                Final allocation weights for Minimum Volatility Portfolio</h4>

                """, 
                unsafe_allow_html=True
            )            



            # Create an HTML table to arrange the DataFrames side by side
            table = """
            <table>
                <tr>
                    <td>{0}</td>
                    <td>{1}</td>
                </tr>
                <tr>
                    <td>{2}</td>
                    <td>{3}</td>
                </tr>
            </table>
            """.format(
                min_volatility_crypto_form_df.to_html(),
                min_volatility_stocks_form_df.to_html(),
                min_volatility_commodities_form_df.to_html(),
                min_volatility_bonds_form_df.to_html()
            )


            st.markdown(table, unsafe_allow_html=True)
            # Display the HTML table with the DataFrames in Streamlit


            # Create scatter pie plot for crypto.
            create_pie_chart(min_volatility_crypto_df, 'Minimum Volatility portfolio for crypto assets')

            # Create scatter pie plot for stocks.
            create_pie_chart(min_volatility_stocks_df, 'Minimum Volatility portfolio for stock assets')

            # Create scatter pie plot for stocks.
            create_pie_chart(min_volatility_commodities_df, 'Minimum Volatility portfolio for commodities')

            # Create scatter pie plot for stocks.
            create_pie_chart(min_volatility_bonds_df, 'Minimum Volatility portfolio for bonds')


            # Define benchmark tickers for each asset class
            benchmark_tickers = {
                'crypto': 'BTC-USD',  # Bitcoin as a benchmark for cryptocurrencies
                'stocks': 'SPY',      # S&P 500 ETF (SPY) as a benchmark for stocks
                'commodities': 'GSG', # S&P GSCI Commodity Index ETF (GSG) as a benchmark for commodities
                'bonds': 'AGG'        # iShares Core U.S. Aggregate Bond ETF (AGG) as a benchmark for bonds
            }

            # Download historical data for benchmark tickers
            benchmarks_data = yf.download(list(benchmark_tickers.values()), start=selected_start_date, end=selected_end_date)['Adj Close']

            # Fill NaN values with the previous data point (forward fill)
            benchmarks_data.fillna(method='ffill', inplace=True)

            # Calculate daily log returns for benchmark_data
            benchmarks_log_returns = np.log(1 + benchmarks_data.pct_change().dropna())

            # Define the weights for each asset class in the benchmark portfolio
            crypto_weight = 0.25
            stocks_weight = 0.25
            commodities_weight = 0.25
            bonds_weight = 0.25

            # Calculate the weighted returns for each asset class
            crypto_weighted_bench_ret = benchmarks_log_returns[benchmark_tickers['crypto']] * crypto_weight
            stocks_weighted_bench_ret = benchmarks_log_returns[benchmark_tickers['stocks']] * stocks_weight
            commodities_weighted_bench_ret = benchmarks_log_returns[benchmark_tickers['commodities']] * commodities_weight
            bonds_weighted_bench_ret = benchmarks_log_returns[benchmark_tickers['bonds']] * bonds_weight

            # Combine the weighted returns into a single benchmark portfolio return
            bench_portfolio_weighted_ret = (
                crypto_weighted_bench_ret 
                + stocks_weighted_bench_ret 
                + commodities_weighted_bench_ret 
                + bonds_weighted_bench_ret
            )

            # Calculate the weighted returns for each asset class in Maximum Sharpe Ratio Portfolio
            max_sharpe_crypto_weighted_returns = crypto_results.mul(
                crypto_max_sharpe_ratio_weights.T.values, axis=1)
            max_sharpe_stocks_weightedreturns = stocks_results.mul(
                stocks_max_sharpe_ratio_weights.T.values, axis=1)
            max_sharpe_weighted_commodities_returns = commodities_results.mul(
                commodities_max_sharpe_ratio_weights.T.values, axis=1)
            max_sharpe_weighted_bonds_returns = bonds_results.mul(
                bonds_max_sharpe_ratio_weights.T.values, axis=1)

            # Calculate the weighted returns for each asset class in Minimum Volatility Portfolio
            min_vol_crypto_weighted_returns = crypto_results.mul(
                crypto_min_volatility_weights.T.values, axis=1)
            min_vol_stocks_weightedreturns = stocks_results.mul(
                stocks_min_volatility_weights.T.values, axis=1)
            min_vol_weighted_commodities_returns = commodities_results.mul(
                commodities_min_volatility_weights.T.values, axis=1)
            min_vol_weighted_bonds_returns = bonds_results.mul(
                bonds_min_volatility_weights.T.values, axis=1)




            def calculate_total_weighted_returns(crypto_weighted_returns, stocks_weighted_returns, commodities_weighted_returns, bonds_weighted_returns):
                """
                Calculate the total weighted returns for a portfolio consisting of four asset classes.

                :weighted_returns: DataFrame containing the weighted returns for each asset class
                
                :return: Series containing the total weighted returns for the given portfolio
                """

                # Combine the DataFrames
                combined_weighted_returns = pd.concat(
                    [
                        crypto_weighted_returns,
                        stocks_weighted_returns,
                        commodities_weighted_returns,
                        bonds_weighted_returns,
                    ],
                    axis=1,
                )

                # Fill NaN values with 0
                combined_weighted_returns.fillna(0, inplace=True)

                # Sum the returns across the columns (tickers) for each day
                total_weighted_returns = combined_weighted_returns.sum(axis=1)

                return total_weighted_returns



            # Call the function for Maximum Sharpe Ratio portfolio
            max_sharpe_total_weighted_returns = calculate_total_weighted_returns(
                max_sharpe_crypto_weighted_returns,
                max_sharpe_stocks_weightedreturns,
                max_sharpe_weighted_commodities_returns,
                max_sharpe_weighted_bonds_returns,
            )

            # Call the function for Minimum Volatility portfolio
            min_vol_total_weighted_returns = calculate_total_weighted_returns(
                min_vol_crypto_weighted_returns,
                min_vol_stocks_weightedreturns,
                min_vol_weighted_commodities_returns,
                min_vol_weighted_bonds_returns,
            )

            #Convert the index of the returns to a DatetimeIndex
            max_sharpe_total_weighted_returns.index = pd.to_datetime(max_sharpe_total_weighted_returns.index)
            min_vol_total_weighted_returns.index = pd.to_datetime(min_vol_total_weighted_returns.index)
            bench_portfolio_weighted_ret.index = pd.to_datetime(bench_portfolio_weighted_ret.index)




            # Create the tear sheet
            st.header("Maximum Sharpe Ratio Portfolio Performance Tear Sheet")
            with st.empty():
                

                # Capture the tear sheet plot and save it as an image
                plt.figure()
                pf.create_simple_tear_sheet(max_sharpe_total_weighted_returns)
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = Image.open(buf)

                # Display the tear sheet image in the Streamlit app
                st.image(image, caption="Tear Sheet", use_column_width=True)

            # Create the tear sheet
            st.header("Minimum Volatility Portfolio Performance Tear Sheet:")
            with st.empty():
                

                # Capture the tear sheet plot and save it as an image
                plt.figure()
                pf.create_simple_tear_sheet(min_vol_total_weighted_returns)
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = Image.open(buf)

                # Display the tear sheet image in the Streamlit app
                st.image(image, caption="Tear Sheet", use_column_width=True)

                        # Create the tear sheet
            st.header("Benchmark Portfolio Performance Tear Sheet:")
            with st.empty():
                

                # Capture the tear sheet plot and save it as an image
                plt.figure()
                pf.create_simple_tear_sheet(bench_portfolio_weighted_ret)
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = Image.open(buf)

                # Display the tear sheet image in the Streamlit app
                st.image(image, caption="Tear Sheet", use_column_width=True)



            # # Display Data Frames of closing pices por each asset class
            # st.write('Maximum Sharpe Ratio Portfolio - Crypto Allocations:')
            # st.dataframe(max_sharpe_crypto_df, width=500, height=300)


            # # Display Data Frames of closing pices por each asset class
            # st.write('Maximum Sharpe Ratio Portfolio - Commodities Allocations:')
            # st.dataframe(max_sharpe_commodities_df, width=500, height=300)



            # # Display Data Frames of closing pices por each asset class
            # st.write('Maximum Sharpe Ratio Portfolio - Bonds Allocations:')
            # st.dataframe(max_sharpe_bonds_df, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Minimum Volatility Portfolio - Crypto Allocations:')
            # st.dataframe(min_volatility_crypto_df, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Minimum Volatility Portfolio - Stocks Allocations:')
            # st.dataframe(min_volatility_stocks_df, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Minimum Volatility Portfolio - Commodities Allocations:')
            # st.dataframe(min_volatility_commodities_df, width=500, height=300)

            # # Display Data Frames of closing pices por each asset class
            # st.write('Minimum Volatility Portfolio - Bonds Allocations:')
            # st.dataframe(min_volatility_bonds_df, width=500, height=300)


            # st.write('First 5 rows of closing prices for stocks:')
            # st.dataframe(stocks_price_df.head(), width=500, height=300)

            # st.write('First 5 rows of closing prices for commodities:')
            # st.dataframe(commodities_price_df.head(), width=500, height=300)

            # # Call function to calculate the log returns for each asset class 
            # crypto_log_returns, stocks_log_returns, commodities_log_returns = calculate_log_returns(crypto_price_df, stocks_price_df, commodities_price_df)

            # # Display Data Frames for logarithmic returns of the assets for chosen time period.
            # st.write('First 5 rows of logarithmic returns for crypto assets:')
            # st.dataframe(crypto_log_returns.head(), width=500, height=300)

            # st.write('First 5 rows of logarithmic returns for stocks:')
            # st.dataframe(stocks_log_returns.head(), width=500, height=300)

            # st.write('First 5 rows of logarithmic returns for commodities:')
            # st.dataframe(commodities_log_returns.head(), width=500, height=300)
            
            

            # st.write('First 5 rows of simulated portfolios for crypto asset class:')
            # st.dataframe(crypto_simulations_df.head(), width=500, height=300)





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
        # img_header = Image.open('Capital_Allocation_Optimization/streamlit_front_end_cap_alloc/data/images/page_2.png')
        # st.image(img_header, width=None)
    

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
        # img_header = Image.open('Capital_Allocation_Optimization/streamlit_front_end_cap_alloc/data/images/bot_page.jpeg')
        # st.image(img_header, use_column_width=True)

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



