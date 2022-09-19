#import libraries

import pandas as pd
from scipy import stats
import numpy as np
from pandas.plotting import scatter_matrix # for plotting data
from statsmodels.formula.api import ols # for multiple regressions
import seaborn ## for statistical visualizations
from matplotlib import pyplot as plt


## Load In Data
df = pd.read_csv('data\sizes_brain.csv', sep=';', na_values=".") # load in csv file
df # view output

######################################


#### CREATE DATAFRAME FROM NUMPY ARRAYS ####
# using numpy
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

df.shape # list # rows by columns: 40 x 8
df.columns # list column names 
df.dtypes # check for different data types 
print(df['Height']) #list values of Height column


### EXERCISE 1 ###
# What is the mean value for VIQ for the full population?
VIQ_mean = df['VIQ'].mean()
VIQ_mean # view output: 112.35

# How many males/females were included in this study?
groupby_gender = df.groupby('Gender') # categorizes # males/females of each column
groupby_gender.count() # view output

# What is the average value of MRI counts expressed in log units, for males and females?
MRI_log2_mean = df['logarithim_base2'] #define variable for log2 mean
df['logarithim_base2'] = np.log2(df['MRI_Count']) # using log 2
MRI_log2_mean # view output

MRI_log10_mean = df['logarithim_base10'] # define variable for log10 mean
df['logarithim_base10'] = np.log2(df['MRI_Count']) # using log10
MRI_log10_mean # view output 


######################################


#### PLOTTING AND MANIPULATING DATA ####
#using pandas.plotting
scatter_matrix(df[['Weight', 'Height', 'MRI_Count']])
scatter_matrix(df[['PIQ', 'VIQ', 'FSIQ']]) # IQ metrics are bimodal, as if there are 2 sub-populations


### EXERCISE 2 ###
# Plot the scatter matrix for males only, and for females only. Do you think that the 2 sub-populations correspond to gender?
scatter_matrix(df[['PIQ', 'VIQ', 'FSIQ']], c=(df['Gender'] == 'Male'), figsize=(5,5), alpha= 1, marker='o', cmap='spring')
#alpha = # for transparency
#figsize = (20,20) for wxh size
#cmap = color map
#marker = 'shape' ('o', 'x')


######################################


#### HYPOTHESIS TESTING: COMPARING 2 GROUPS ####
#using scipy t-test to create simple tests to compare 2 groups


### EXERCISE 3 ###
# Test the difference between weights in males and females
## tests if population of mean of data is likely to be equal to a given value 

stats.ttest_1samp(df.dropna()['Weight'], 0) ## Resolved NA error by uding df.dropna to remove rows with NULL values
female_weight = df.dropna()[df['Gender'] == 'Female']['Weight']
male_weight = df.dropna()[df['Gender'] == 'Male']['Weight']
stats.ttest_ind(female_weight, male_weight) 

#Use non parametrics statistics to test the difference between VIQ in males and females 
female_viq = df[df['Gender'] == 'Female']['VIQ']
male_viq = df[df['Gender'] == 'Male']['VIQ']
stats.mannwhitneyu(female_viq, male_viq)


######################################


#### LINEAR MODELS, MULTIPLE FACTORS, AND ANALYSIS OF VARIANCE ####

### EXERCISE 4 ###
# Retrieve the estimated parameters from the model above using numpy
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
df = pd.DataFrame({'x': x, 'y': y})
df

# using statsmodels.formula.api to specify an OLS model and fit it
model = ols("y ~ x", df).fit()
print(model.summary())



######################################

#### MULTIPLE REGRESSION: INLCUDING MULTIPLE FACTORS ####
### POST-HOC HYPOTHESIS TESTING: ANALYSIS OF VARIANCE (ANOVA) ###

### EXERCISE 5 ###
#Going back to the brain size + IQ data, test if the VIQ of male and female are different after removing the effect of brain size, height and weight.
df = pd.read_csv(r"C:\Users\Alice\Documents\GitHub\descriptives-scipy\data\sizes_brain.csv", sep=';', na_values=".")
#write vector of contrast
model = ols('VIQ ~ Gender + MRI_Count + Height', df).fit() 
print(model.summary())
print (model.f_test([0, 1, -1,0]))