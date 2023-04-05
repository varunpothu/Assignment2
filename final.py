# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:08:22 2023

@author: Varun
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the data 
data = pd.read_csv('Climate_Data1.csv')  


def electricity_data(filename):
    
    
    """
    Reads a dataframe in World bank format from a file and returns two dataframes:
    one with years as columns and one with countries as columns for processing the data.

    Args:
    filename (str): The name of the file containing the data.

    Returns:
    years_data (DataFrame): A dataframe with years as columns and countries as rows.
    countries_data (DataFrame): A dataframe with countries as columns and years as rows.
    """
    
    # Load the data into a DataFrame
    #data = pd.read_csv(filename)

    # Extract the data for the years 2010-2019
    years_data = data.loc[:, 'Country Name':'2014']
    years_data.columns = [col if not col.isdigit() else str(col) for col in years_data.columns]
    
    # Transpose the DataFrame to get a country-centric view
    countries_data = years_data.transpose()

    # Replace empty values with 0
    countries_data = countries_data.fillna(0)

    # Set the column names for the countries DataFrame
    countries_data.columns = countries_data.iloc[0]
    countries_data = countries_data.iloc[1:]
    countries_data.index.name = 'Year'

    # Set the column names for the years DataFrame
    years_data = years_data.rename(columns={'Country Name': 'Year'})
    years_data = years_data.set_index('Year')

    # Describe the years_data dataframe
    years_data_description = years_data.describe()

    return years_data, countries_data


years_data, countries_data = electricity_data("Climate_Data1.csv")
years_data_description = years_data.describe()
print(years_data_description)
print(years_data.info)
print(years_data.head())
print(countries_data.head())



def plot_electricity_use_by_country(data):  
    
    
    """
    Calculates electricity use by various countries 
    Arg: Data
    
    """
    
    #years_data, countries_data = electricity_data(data)
    # Select the data for the specified countries and years
    countries = ["India","United States", "China", "United Kingdom", "Russian\
                 Federation", "France", "Germany", "Canada"]
    years = ["2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", \
             "2012", "2013", "2014"]
    indicator = "EG.USE.ELEC.KH.PC"
    data_x = data.loc[data["Country Name"].isin(countries) & \
             data["Indicator Code"].eq(indicator), ["Country Name"] + years]

    # Create the grouped bar chart
    ax = data_x.plot(kind="bar", x="Country Name", figsize=(10, 6))
    ax.set_xlabel("Country")
    ax.set_ylabel("Electricity Use (kWh per capita)")
    ax.set_title("Electricity Use by Country (2004-2014)")
    plt.legend(title="Year", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.show()

plot_electricity_use_by_country(data)


def plot_gdp_per_energy_use(data): 
    
    
    """
    Calculates GDP per energy use
    
    """
    
    
    # Select the data for the specified countries and years
    countries = ["India","United States", "China", "United Kingdom", "Russian \
                 Federation", "France", "Germany", "Canada"]
    years = ["2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", \
             "2012", "2013", "2014"]
    indicator = "EG.GDP.PUSE.KO.PP"
    data_y = data.loc[data["Country Name"].isin(countries) & \
             data["Indicator Code"].eq(indicator), ["Country Name"] + years]

    # Convert the data to numeric values
    data_y[years] = data_y[years].apply(pd.to_numeric)

    # Create the line chart
    ax = data_y.set_index("Country Name").T.plot(figsize=(10, 6))
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP per unit of energy use (PPP $ per kg of oil equivalent)")
    ax.set_title("GDP per unit of energy use (PPP $ per kg of oil equivalent) \
                (2004-2014)")
    plt.legend(title="Country", loc="upper left")
    plt.show()
    
plot_gdp_per_energy_use(data)


def plot_electricity_production(data):
    
    
    """
    Calculates electricity production
    
    """
   
    # Define the indicators, years, and countries of interest
    indicators = ['EG.ELC.COAL.ZS', 'EG.ELC.HYRO.ZS', 'EG.ELC.NGAS.ZS', \
                  'EG.ELC.NUCL.ZS', 'EG.ELC.PETR.ZS', 'EG.ELC.RNWX.ZS']
    
    years = ["2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", \
             "2012", "2013", "2014"]
    countries = ["India", "United States", "China", "United Kingdom", \
                 "Russian Federation", "Germany", "Canada"]

    # Filter the data to select the rows that correspond to the countries and indicators of interest, and calculate mean values for the indicators over the 8 countries
    data = data.loc[data["Country Name"].isin(countries) & \
           data["Indicator Code"].isin(indicators), ["Country Name",\
            "Indicator Code"] + years]
    grouped_data = data.groupby("Indicator Code").mean()

    # Create the grouped bar chart
    ax = grouped_data.T.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity Production (% of total)")
    ax.set_title("Electricity Production by Source ({0}-{1})".format(years[0], \
                                                                     years[-1]))
    plt.legend(title="Indicator", labels=['Coal', 'Hydroelctric', 'Natural Gas',\
              'Nuclear', 'Petroleum','Renewable'], loc="upper left", \
               bbox_to_anchor=(1.02, 1))
    plt.show()
    
plot_electricity_production(data)


def filter_data(data, indicator_name, countries):
    
    
    """
    Filter the data to select a specific indicator and countries of interest.
    Parameters:
        data (pandas.DataFrame): The raw data to filter.
        indicator_name (str): The name of the indicator to select.
        countries (list of str): The names of the countries to select. 
    Returns:
        pandas.DataFrame: The filtered data with only the selected indicator and countries.
    """
    
    
    data_filtered = data[data['Indicator Name'] == indicator_name]
    data_filtered = data_filtered[data_filtered['Country Name'].isin(countries)]
    data_filtered = data_filtered.loc[:, '2004':'2014']
    return data_filtered


def calculate_stats(data_filtered, countries):
    
    
    """
    Calculate the mean and standard deviation for each country in the filtered data.
    Parameters:
        data_filtered (pandas.DataFrame): The filtered data to calculate statistics for.
        countries (list of str): The names of the countries in the data.
    Returns:
        dict: A dictionary with the country names as keys and a tuple of the mean and standard deviation as values.
    """
    
    country_stats = {}
    for country in countries:
        country_data = data_filtered.loc[data_filtered.index == country]
        if not country_data.empty:
            country_data_list = country_data.values.tolist()[0]
            mean = stats.mean(country_data_list)
            stdev = stats.stdev(country_data_list)
            country_stats[country] = (mean, stdev)
    return country_stats


def plot_data(data_filtered, countries, indicator_name):
    
    
    """
    Plot the filtered data for the selected countries and indicator.
    Parameters:
        data_filtered (pandas.DataFrame): The filtered data to plot.
        countries (list of str): The names of the countries in the data.
        indicator_name (str): The name of the indicator to plot.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(data_filtered.transpose(), linestyle='--')
    plt.legend(countries, loc='upper left', fontsize='small')
    plt.xlabel('Year')
    plt.title(indicator_name)
    plt.show()

# Define the countries and years of interest
countries = ["Canada", "China", "Germany", "France", "United Kingdom", "India",\
             "Russian Federation", "United States"]
years = ['2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',\
         '2013', '2014']

# Filter the data for the desired indicator and countries
indicator_name = 'Electricity production from natural gas sources (% of total)'
data_filtered = filter_data(data, indicator_name, countries)

# Calculate the statistics for each country in the filtered data
country_stats = calculate_stats(data_filtered, countries)

# Print the statistics for each country
for country, (mean, stdev) in country_stats.items():
    print(f'{country}: Mean = {mean}, Standard Deviation = {stdev}')

# Plot the filtered data for the selected countries and indicator
plot_data(data_filtered, countries, indicator_name)


def filter_data(data, indicator_name, countries, start_year, end_year):
    
    
    """
    Filter the input data to include only the specified indicator, countries, and years.
    Parameters:
    data (DataFrame): Input data containing indicator values for multiple countries and years
    indicator_name (str): Name of the indicator to filter on
    countries (list): List of country names to include in the filtered data
    start_year (str): First year to include in the filtered data (format: 'YYYY')
    end_year (str): Last year to include in the filtered data (format: 'YYYY')
    Returns:
    DataFrame: Filtered data containing only the specified indicator, countries, and years
    """    
    
    # Filter data by indicator name and countries
    data_filtered = data[data['Indicator Name'] == indicator_name]
    data_filtered = data_filtered[data_filtered['Country Name'].isin(countries)]
    
    # Filter data by start and end year
    data_filtered = data_filtered.loc[:, start_year:end_year]
    return data_filtered


def calculate_stats(data, countries):
    
    
    """
    Calculate the median and standard deviation for each country in the input data.
    Parameters:
    data (DataFrame): Input data containing indicator values for multiple countries and years
    countries (list): List of country names to include in the analysis
    Returns:
    dict: Dictionary containing the median and standard deviation for each country
    """
    
    country_stats = {}
    for country in countries:
        country_data = data.loc[data.index == country]
        if not country_data.empty:
            country_data_list = country_data.values.tolist()[0]
            median = np.median(country_data_list)
            stdev = stats.stdev(country_data_list)
            country_stats[country] = {'Median': median, 'Standard Deviation': stdev}
    return country_stats


def plot_data(data, countries, title):
    
    
    """
    Plot the input data for the specified countries.
    Parameters:
    data (DataFrame): Input data containing indicator values for multiple countries and years
    countries (list): List of country names to include in the plot
    title (str): Title of the plot
    Returns:
    None
    """
    
    plt.figure(figsize = (10,6))
    plt.plot(data.transpose(), linestyle='--')
    plt.legend(countries, loc='upper left', fontsize='small')
    plt.xlabel('Year')
    plt.title(title)
    plt.show()

# Define the parameters for filtering the data
indicator_name = 'Electricity production from coal sources (% of total)'
countries = ["Canada", "China", "Germany", "France", "United Kingdom", "India", \
             "Russian Federation", "United States"]
start_year = '2004'
end_year = '2014'

# Filter the data and calculate statistics for each country
filtered_data = filter_data(data, indicator_name, countries, start_year, end_year)
country_stats = calculate_stats(filtered_data, countries)

# Print the results
for country in country_stats:
    print(country + ": " + str(country_stats[country]))

# Plot the filtered data for the specified countries
title = 'Electricity production from coal sources (% of total)'
plot_data(filtered_data, countries, title)
