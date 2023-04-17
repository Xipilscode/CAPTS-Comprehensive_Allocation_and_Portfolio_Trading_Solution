# CAPTS: Comprehensive Allocation and Portfolio Trading Solution 

![CAPTS](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/logo.gif)

This repository contains an application that utilizes Modern Portfolio Theory (MPT) to construct optimized portfolios based on historical data, with a focus on either Maximum Sharpe Ratio or Minimum Volatility. The step-by-step process includes:

1. Selecting asset classes and tickers: Choose the assets and asset classes for analysis (crypto, stocks, commodities, and bonds).
2. Data acquisition: Download historical price data for the chosen assets from Yahoo Finance.
3. Data preprocessing: Clean and preprocess the data to calculate returns and prepare it for further analysis.

4. ![PCA and K-means clustering:](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/ml_algorithms.png)
Perform Principal Component Analysis (PCA) for dimensionality reduction and K-means clustering for asset classification. The application includes an automated function that leverages the elbow method to identify the optimal number of clusters for K-means clustering. This automated approach streamlines the clustering process and enhances the asset classification efficiency. 
![Automated clustering function](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/automated_function9%20PM.png)
5. Split assets into asset classes: Assign the assets to their respective asset classes based on the clustering results.
6. Monte Carlo simulation (first layer): Run Monte Carlo simulations for each asset class to determine optimal portfolio weights.
![Monte Carlo simulation plot](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/monte_carlo_simulation_plot.png)
7. Monte Carlo simulation (second layer): Run another layer of Monte Carlo simulations for the combined asset classes. 
8. Portfolio optimization: Optimize the portfolios based on Maximum Sharpe Ratio and Minimum Volatility.
9. Asset allocation visualization: Display the final allocation for each asset in pie charts. (Screenshot: !![Pie_charts_screenshot](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/pie_chart_1.png)
10. Benchmark portfolio construction: Define benchmark tickers and download historical data for comparison with the optimized portfolios.
11. Performance analysis: Calculate the returns for the optimized portfolios and benchmark portfolio.
12. Generate performance tear sheets: Create performance tear sheets for the Maximum Sharpe Ratio portfolio, Minimum Volatility portfolio, and benchmark portfolio. 
Screenshots:
![Maximum Sharpe Ratio portfolio Tear Sheet](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/max_sharpe_ratio_portfolio.png)
![Minimum Volatility portfolio Tear Sheet](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/minimum_volatility_portfolio.png)
![Benchmark portfolio Tear Sheet](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/benchmark_portfolio.png)

The next step for the development of this application is to choose between the Maximum Sharpe Ratio portfolio and Minimum Volatility portfolio and use it for algorithmic trading. The repository also includes code in development for a trading strategy based on Volume Exponential Moving Average (EMA) signals, which will be used to implement and backtest the chosen portfolio on the Alpaca paper trading platform.

The trading algorithm will rebalance the portfolio weekly, with the potential for adjusting the rebalancing schedule as needed. The application is implemented using Streamlit, providing an interactive and user-friendly interface. 
The Streamlit app can be found here:
[CAPTS](https://xipilscode-capts-comprehensive-allocation-and--streamlit-1966nx.streamlit.app/)
![Screenshots of the web application](https://github.com/Xipilscode/CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution/blob/main/images/web_page_shot1.png)

## Technologies
This project leverages the following technologies:

* [Python 3.7.13](https://www.python.org/downloads/release/python-385/) - The programming language used in the project.
* [Streamlit](https://streamlit.io/) - A fast and easy way to create and share data apps.
* [Pandas](https://pandas.pydata.org/) - A Python library used for efficient data manipulation.
* [NumPy](https://numpy.org/) - A powerful Python library for numerical computing.
* [Scikit-learn](https://scikit-learn.org/stable/index.html) - A Python library containing efficient tools for machine learning and statistical modeling, including classification, regression, clustering, and dimensionality reduction.
* [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - A powerful unsupervised machine learning algorithm used to solve complex problems.
* [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - A statistical technique used to speed up machine learning algorithms when dealing with a large number of features or dimensions.
* [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) - A class for basic strategies for imputing missing values.
* [yfinance](https://pypi.org/project/yfinance/) - A simple and convenient way to download Yahoo Finance data directly into Python.
* [Plotly](https://plotly.com/python/) - A graphing library that makes interactive, publication-quality graphs.
* Plotly Express - A high-level interface for data visualization.
* Plotly Graph Objects - A low-level interface to create fully customizable figures.
* Seaborn - A Python data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
* [Matplotlib](https://matplotlib.org/) - A Python 2D plotting library used for creating static, animated, and interactive visualizations.
* [pyfolio](https://pypi.org/project/pyfolio/) - A Python library for performance and risk analysis of financial portfolios.
* IPython - A command shell for interactive computing in multiple programming languages.
* [warnings](https://docs.python.org/3/library/warnings.html) - A built-in Python module for issuing warning messages.

 ## Installation 
 
 To run this project, you will need to install the following technologies:
 
1. To run this project, you'll need to install [Python 3.7.13](https://www.python.org/downloads/release/python-385/)
2. To install Pandas, run the following command in your terminal or command prompt:
```
pip install pandas
```
3. To install Jupyter Lab, run the following command in your terminal or command prompt:
```
pip install jupyterlab
```
4. To install the PyViz packages, check that your development environment is active, and then run the following command:
```
conda install -c pyviz hvplot
```

5. Installing Scikit-learn
 * To install the Scikit-learn, check that your development environment is active, and then run the following command:
```
pip install -U scikit-learn
```
 * To check if scikit-learn is already installed, you can run the following code on your dev environment:
```
conda list scikit-learn
```

6. To install pyfolio, run the following command in your terminal or command prompt:

```
pip install pyfolio
```
7. To install Plotly, run the following command in your terminal or command prompt:
```
pip install plotly
```
8. To install Seaborn, run the following command in your terminal or command prompt:
```
pip install plotly
```
9. To install Matplotlib, run the following command in your terminal or command prompt:
```
pip install plotly
```
10.  After installing the technologies, you can launch Jupyter Lab by running the following command in your terminal or command prompt:
```
jupyter lab
```

11. Run the file **capts.ipynb** in the Jupyter Notebook

## Contributors
[Alexander Likhachev](https://github.com/Xipilscode)
[Alphonso Logan](https://github.com/fonzeon)
[Firas Obeid](https://github.com/firobeid)

## License
MIT
