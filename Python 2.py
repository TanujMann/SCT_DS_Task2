import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the Dataset

df=pd.read_csv("D:/World Economic Indicators.csv")

#Display Basic Info
print(df.info())
print(df.head())

#Check for missing Values
missing_values = df.isnull().sum()

# Drop rows where essential values are missing (e.g., GDP and GDP per capita)
df.dropna(subset=['GDP (USD)', 'GDP per capita (USD)'], inplace=True)

# Loop through all columns with missing values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['float64', 'int64']:
            # For numerical columns, use mode if available, else use mean
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
        else:
            # For categorical columns, use mode
            df[col].fillna(df[col].mode()[0], inplace=True)

# Objective 1: Relationship Between GDP and GDP per Capita

plt.figure(figsize=(8, 6))
sns.scatterplot(x='GDP (USD)', y='GDP per capita (USD)', hue='Region', data=df)
plt.title('GDP vs GDP per Capita')
plt.xlabel('GDP (USD)')
plt.ylabel('GDP per capita (USD)')
plt.show()

# Objective 2: Explore Regional GDP Differences

plt.figure(figsize=(10, 6))
region_gdp = df.groupby('Region')['GDP (USD)'].sum().sort_values(ascending=False)
sns.barplot(x=region_gdp.values, y=region_gdp.index, palette='viridis')
plt.title('Total GDP by Region')
plt.xlabel('Total GDP (USD)')
plt.ylabel('Region')
plt.show()

# Objective 3: GDP per capita trend for a country
country = 'India'  

df_country = df[df['Country Name'] == country]

plt.figure(figsize=(10, 6))
plt.plot(df_country['Year'], df_country['GDP per capita (USD)'], marker='o', linestyle='-')
plt.title(f'GDP per Capita Trend of {country}')
plt.xlabel('Year')
plt.ylabel('GDP per capita (USD)')
plt.grid(True)
plt.show()

# Objective 4: Compare the spread of Population Density among different Regions.

plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='Population Density ', data=df, palette='Set2')  # <== here!
plt.title('Population Density Distribution Across Regions')
plt.xlabel('Region')
plt.ylabel('Population Density ')
plt.xticks(rotation=45)
plt.show()

# Objective 5: Find correlations between GDP per capita, and Population Density.

selected_columns = ['GDP per capita (USD)', 'Population Density ']

corr_matrix = df[selected_columns].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap: GDP per Capita, and Population Density ')
plt.show()

# Objective 6: Distribution of countries across different regions

# Count the number of countries per region
region_counts = df['Region'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title('Distribution of Countries by Region')
plt.axis('equal')
plt.show()




