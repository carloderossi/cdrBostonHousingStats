import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Load the modified CSV file
boston_url = "boston_housing.csv"
boston_df=pd.read_csv(boston_url)

# Step 2: Summary Statistics
print("Summary Statistics:\n", boston_df.describe())

# Filter the properties near the Charles River
properties_near_river = boston_df[boston_df['CHAS'] == 1]

# List the values of these properties (assuming you want to list 'MEDV' or any other column)
values_near_river = properties_near_river['MEDV'].tolist()

# Print the values
print("\nValues of properties near the Charles River (CHAS=1):")
print(values_near_river)
print(len(values_near_river))


# Step 3: Missing Values Check
print("\nMissing Values:\n", boston_df.isnull().sum())

# Step 4: Data Visualization

# Histogram for each feature

# Exclude the first column
boston_df_excluded = boston_df.iloc[:, 1:]  # Select all columns except the first one

# Plot the histogram of all features except the first column
boston_df_excluded.hist(bins=20, figsize=(15, 10))
pyplot.suptitle('Histogram of All Features')
 # Show the plot
manager = pyplot.get_current_fig_manager()
manager.set_window_title('Histogram of All Features')
pyplot.show()

# Boxplot to see the distribution and outliers
pyplot.figure(figsize=(15, 10))
sns.boxplot(data=boston_df_excluded)
pyplot.title('Boxplot of All Features')
pyplot.xticks(rotation=90)
 # Show the plot
manager = pyplot.get_current_fig_manager()
manager.set_window_title('Boxplot of All Features')
pyplot.show()

# Step 5: Correlation Matrix
pyplot.figure(figsize=(15, 10))
correlation_matrix = boston_df_excluded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
pyplot.title('Correlation Matrix')
pyplot.show()

# Boxplot for the Median value of owner-occupied homes
#  This boxplot will show the distribution of the median value of owner-occupied homes (MEDV). 
#  It will help identify any outliers and understand the spread and central tendency of the values.
pyplot.figure(figsize=(10, 6))
sns.boxplot(y=boston_df['MEDV'])
pyplot.title('Boxplot of Median Value of Owner-Occupied Homes')
pyplot.ylabel('Median Value of Owner-Occupied Homes (in $1000s)')
 # Show the plot
manager = pyplot.get_current_fig_manager()
manager.set_window_title('Boxplot of Median Value of Owner-Occupied Homes')
pyplot.show()


# Bar plot for the Charles River variable
#  This bar plot shows the count of properties next to the Charles River (CHAS = 1) versus those that are not (CHAS = 0). 
#  This helps us understand how many properties in the dataset are located near the river.
pyplot.figure(figsize=(10, 6))

# Define a color palette 
palette = sns.color_palette("pastel", 2)  # Assuming there are 2 unique values in 'CHAS'

# Create the countplot
ax = sns.countplot(x=boston_df['CHAS'], hue=boston_df['CHAS'], palette=palette)

# Annotate each bar with the corresponding value
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Customize the plot
pyplot.title('Bar Plot of Charles River Variable')
pyplot.xlabel('Charles River (0 = No, 1 = Yes)')
pyplot.ylabel('Count')

# Show the plot
manager = pyplot.get_current_fig_manager()
manager.set_window_title('Bar Plot of Charles River Variable')
pyplot.show()


# Boxplot for MEDV vs Age Group (Discretized)
#  This boxplot shows the median value of owner-occupied homes (MEDV) across different age groups of the homes. 
#  It helps us see if there's a relationship between the age of homes and their median value.
# Discretize the AGE variable
boston_df['Age Group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['0-35', '36-70', '71-100'])
pyplot.figure(figsize=(10, 6))
palette = sns.color_palette("Set2")
sns.boxplot(x=boston_df['Age Group'], y=boston_df['MEDV'], hue=boston_df['Age Group'], palette=palette)
pyplot.title('Boxplot of MEDV vs Age Group')
pyplot.xlabel('Age Group')
pyplot.ylabel('Median Value of Owner-Occupied Homes (in $1000s)')
 # Show the plot
manager = pyplot.get_current_fig_manager()
manager.set_window_title('Boxplot of MEDV vs Age Group')
pyplot.show()

# Scatter Plot for Nitric Oxide Concentrations (NOX) vs Proportion of Non-Retail Business Acres (INDUS)


# Scatter plot for NOX vs INDUS
#  This scatter plot shows the relationship between the proportion of non-retail business acres per town (INDUS) and the nitric oxide concentrations (NOX). 
#  It helps us observe if there is any correlation between industrial activities and pollution levels. Typically, we might expect to see a positive correlation if industrial areas contribute to higher pollution.
pyplot.figure(figsize=(10, 6))
# https://matplotlib.org/stable/gallery/color/named_colors.html
sns.scatterplot(x=boston_df['INDUS'], y=boston_df['NOX'], color='indianred', marker='D')
pyplot.title('Scatter Plot of Nitric Oxide Concentrations vs Proportion of Non-Retail Business Acres')
pyplot.xlabel('Proportion of Non-Retail Business Acres (INDUS)')
pyplot.ylabel('Nitric Oxide Concentrations (NOX)')
 # Show the plot
manager = pyplot.get_current_fig_manager()
manager.set_window_title('Scatter Plot of Nitric Oxide Concentrations vs Proportion of Non-Retail Business Acres')
pyplot.show()


# Histogram for the pupil to teacher ratio
#  This histogram shows the distribution of the pupil to teacher ratio (PTRATIO) in the dataset. 
#  It helps us understand the spread and central tendency of this variable. Histograms are useful for visualizing the frequency distribution of a dataset.
pyplot.figure(figsize=(10, 6))
pyplot.hist(boston_df['PTRATIO'], bins=20, edgecolor='k')
pyplot.title('Histogram of Pupil to Teacher Ratio')
pyplot.xlabel('Pupil to Teacher Ratio (PTRATIO)')
pyplot.ylabel('Frequency')
 # Show the plot
manager = pyplot.get_current_fig_manager()
manager.set_window_title('Histogram of Pupil to Teacher Ratio')
pyplot.show()

# B1. Is there a significant difference in median value of houses bounded by the Charles River or not? (T-test for independent samples)
print("\n B1. Is there a significant difference in median value of houses bounded by the Charles River or not? (T-test for independent samples)")
from scipy.stats import ttest_ind
# Separate the median values based on the Charles River variable
medv_river = boston_df[boston_df['CHAS'] == 1]['MEDV']
medv_no_river = boston_df[boston_df['CHAS'] == 0]['MEDV']

# Perform the t-test for independent samples
t_stat, p_value = ttest_ind(medv_river, medv_no_river)

print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
# Enhanced summary
mean_river = medv_river.mean()
mean_no_river = medv_no_river.mean()
std_river = medv_river.std()
std_no_river = medv_no_river.std()

print(f"Mean value of houses near the Charles River: {mean_river:.2f}")
print(f"Mean value of houses not near the Charles River: {mean_no_river:.2f}")
print(f"Standard deviation of houses near the Charles River: {std_river:.2f}")
print(f"Standard deviation of houses not near the Charles River: {std_no_river:.2f}")

# Visualization: Violin plot
pyplot.figure(figsize=(12, 6))
sns.violinplot(x='CHAS', y='MEDV', data=boston_df, hue=boston_df['CHAS'], palette="Set2")
pyplot.title('Violin Plot of Median House Values by Proximity to Charles River')
pyplot.xlabel('Proximity to Charles River (0 = No, 1 = Yes)')
pyplot.ylabel('Median Value of Owner-Occupied Homes (in $1000s)')
pyplot.show()

# B2. Is there a difference in Median values of houses (MEDV) for each proportion of owner-occupied units built prior to 1940 (AGE)? (ANOVA)
print("\n B2. Is there a difference in Median values of houses (MEDV) for each proportion of owner-occupied units built prior to 1940 (AGE)? (ANOVA)")
from scipy.stats import f_oneway

# Discretize the AGE variable
boston_df['Age Group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['0-35', '36-70', '71-100'])

# Separate the median values by age groups
medv_group1 = boston_df[boston_df['Age Group'] == '0-35']['MEDV']
medv_group2 = boston_df[boston_df['Age Group'] == '36-70']['MEDV']
medv_group3 = boston_df[boston_df['Age Group'] == '71-100']['MEDV']

# Perform the ANOVA test
f_stat, p_value = f_oneway(medv_group1, medv_group2, medv_group3)

print(f"F-Statistic: {f_stat}, P-Value: {p_value}")

# Enhanced summary statistics
mean_group1 = medv_group1.mean()
mean_group2 = medv_group2.mean()
mean_group3 = medv_group3.mean()

std_group1 = medv_group1.std()
std_group2 = medv_group2.std()
std_group3 = medv_group3.std()

print(f"Group '0-35' - Mean: {mean_group1:.2f}, Std: {std_group1:.2f}")
print(f"Group '36-70' - Mean: {mean_group2:.2f}, Std: {std_group2:.2f}")
print(f"Group '71-100' - Mean: {mean_group3:.2f}, Std: {std_group3:.2f}")

# Visualization: Violin Plot
pyplot.figure(figsize=(12, 6))
sns.violinplot(x='Age Group', y='MEDV', data=boston_df, hue=boston_df['Age Group'], palette="Accent")
pyplot.title('Violin Plot of Median House Values by Age Group')
pyplot.xlabel('Age Group')
pyplot.ylabel('Median Value of Owner-Occupied Homes (in $1000s)')
pyplot.show()

import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Discretize the AGE variable
boston_df['Age Group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['0-35', '36-70', '71-100'])

# Separate the median values by age groups
medv_group1 = boston_df[boston_df['Age Group'] == '0-35']['MEDV']
medv_group2 = boston_df[boston_df['Age Group'] == '36-70']['MEDV']
medv_group3 = boston_df[boston_df['Age Group'] == '71-100']['MEDV']

# Combine all groups for Tukey's HSD test
medv_values = boston_df['MEDV']
age_groups = boston_df['Age Group']

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(endog=medv_values, groups=age_groups, alpha=0.05)

print(tukey_result)

# Plot the results
tukey_result.plot_simultaneous()
pyplot.title('Tukey HSD Test Results')
pyplot.xlabel('Mean Differences')
pyplot.ylabel('Age Groups')
pyplot.show()


# B3. Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)
print("\n B3. Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)")

from scipy.stats import pearsonr

# Calculate the Pearson correlation coefficient and p-value
corr_coefficient, p_value = pearsonr(boston_df['NOX'], boston_df['INDUS'])

# Enhanced print statements
print(f"Pearson Correlation Coefficient: {corr_coefficient:.2f}")
print(f"P-Value: {p_value:.4e}")

# Interpretation of the results
if p_value < 0.05:
    print("The p-value is less than 0.05, indicating a statistically significant relationship between NOX and INDUS.")
    if corr_coefficient > 0:
        print("The positive correlation coefficient suggests that higher proportions of non-retail business acres are associated with higher nitric oxide concentrations.")
    else:
        print("The negative correlation coefficient suggests that higher proportions of non-retail business acres are associated with lower nitric oxide concentrations.")
else:
    print("The p-value is greater than 0.05, indicating no statistically significant relationship between NOX and INDUS.")

# Visualization: Scatter plot with regression line
pyplot.figure(figsize=(10, 6))
sns.regplot(x='INDUS', y='NOX', data=boston_df, scatter_kws={'color':'indianred'}, line_kws={'color':'blue'})
pyplot.title('Scatter Plot of NOX vs INDUS with Regression Line')
pyplot.xlabel('Proportion of Non-Retail Business Acres (INDUS)')
pyplot.ylabel('Nitric Oxide Concentrations (NOX)')
pyplot.show()

# Visualization: Heatmap of the correlation matrix
pyplot.figure(figsize=(8, 6))
sns.heatmap(boston_df[['NOX', 'INDUS']].corr(), annot=True, cmap='coolwarm')
pyplot.title('Correlation Matrix Heatmap')
pyplot.show()

# Additional Descriptive Statistics
print("\nDescriptive Statistics:")
print(boston_df[['NOX', 'INDUS']].describe())




# B4. What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner-occupied homes? (Regression analysis)
print("\n B4. What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner-occupied homes? (Regression analysis)")

import statsmodels.api as sm

# Define the dependent and independent variables
X = boston_df['DIS']
y = boston_df['MEDV']

# Add a constant to the independent variable
X = sm.add_constant(X)

# Perform the regression analysis
model = sm.OLS(y, X).fit()

#print(model.summary())


# Enhanced output: Extract important statistics
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
f_statistic = model.fvalue
p_value_f = model.f_pvalue
coefficients = model.params

print(f"R-squared: {r_squared:.4f}")
print(f"Adjusted R-squared: {adj_r_squared:.4f}")
print(f"F-statistic: {f_statistic:.4f}")
print(f"F-statistic P-value: {p_value_f:.4e}")
print(f"Coefficients:\n{coefficients}")
print(model.summary())

# Visualization: Scatter plot with regression line
pyplot.figure(figsize=(10, 6))
sns.regplot(x='DIS', y='MEDV', data=boston_df, scatter_kws={'color':'indianred'}, line_kws={'color':'blue'})
pyplot.title('Scatter Plot of Median House Values vs. Distance to Employment Centres')
pyplot.xlabel('Weighted Distance to Employment Centres (DIS)')
pyplot.ylabel('Median Value of Owner-Occupied Homes (in $1000s)')
pyplot.show()

# Visualization: Residual plot
pyplot.figure(figsize=(10, 6))
sns.residplot(x='DIS', y='MEDV', data=boston_df, lowess=True, scatter_kws={'color':'indianred'}, line_kws={'color':'blue'})
pyplot.title('Residual Plot of Median House Values vs. Distance to Employment Centres')
pyplot.xlabel('Weighted Distance to Employment Centres (DIS)')
pyplot.ylabel('Residuals')
pyplot.show()
