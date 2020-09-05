#!/usr/bin/env python
# coding: utf-8

# # Cleaning and Analyzing Employee Surveys
# by Nicholas Archambault
# 
# This project explores exit surveys from the Department of Education, Training and Employment (DETE) and the Technical and Further Education (TAFE) institute in Queensland, Australia.
# 
# We seek to determine whether resignations are typically motivated by some type of dissatisfaction, and whether disparities exist between the resignation circumstances of employees of varying ages and experience.
# 
# Prior to addressing these questions, we'll need to combine the two datasets and clean the accumulated data to render it fit for analysis.

# ## Introduction
# We'll first read in the data from both surveys and do some initial exploration.

# In[1]:


# Import packages and explore each dataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd

dete = pd.read_csv("dete_survey.csv")
tafe = pd.read_csv("tafe_survey.csv")


# ### DETE Information

# In[2]:


dete.head()


# In[3]:


dete.info()


# In[4]:


dete.isnull().sum()


# The `dete` dataframe contains 822 entries and 56 variable columns.  Examination of its null values shows that the `Classification` and `Business Unit` columns contain significant missing data.  In the final columns pertaining to demographic information like race and disability status, most information is absent.  

# ### TAFE Information

# In[5]:


tafe.head()


# In[6]:


tafe.info()


# In[7]:


tafe.isnull().sum()


# The `tafe` dataframe contains 702 entries and 72 columns.  We observe that its columns have more cumulative null values than the columns of the `dete` dataframe.  A conspicuous number of them have exactly the same number of missing null entries.  There seem to be lots of values that are not stated, but not listed as `NaN`.

# # Identifying Missing Values and Dropping Unnecessary Columns
# 
# Both datasets contain a number of columns that will not be necessary for our analysis.  Each contains many of the same columns under different names. We'll first correct all `Not Stated` values and drop columns not needed for analysis. 

# In[8]:


dete = pd.read_csv("dete_survey.csv", na_values = "Not Stated") # Read in survey again, but `Not Stated` becomes NaN


# In[9]:


# Drop columns from both datasets
dete = dete.drop(dete.columns[28:49], axis = 1)
tafe = tafe.drop(tafe.columns[17:66], axis = 1)


# We must clean and standardize the column names for each dataset, since we will eventually want to combine them. The `dete` columns can be cleaned, and the names of the `tafe` columns can be changed to match.

# In[10]:


dete.columns = dete.columns.str.lower().str.strip().str.replace("\s+", "_")
tafe = tafe.rename({'CESSATION YEAR':"cease_date", 
                    "Record ID":"id", 
                    "Reason for ceasing employment":"separationtype", 
                    "Gender. What is your Gender?":"gender", 
                    "CurrentAge. Current Age":"age", 
                    "Employment Type. Employment Type":"employment_status", 
                    "Classification. Classification":"position", 
                    "LengthofServiceOverall. Overall Length of Service at Institute (in years)":"institute_service", 
                    "LengthofServiceCurrent. Length of Service at current workplace (in years)":"role_service"}, 
                   axis = 1)


# # Filter the Data
# 
# For this project, we'll only analyze survey respondents who resigned, so we'll only select separation types containing the string 'Resignation'.

# In[11]:


dete["separationtype"].value_counts()


# In[12]:


tafe["separationtype"].value_counts()


# In[13]:


# Create new dataframe with only separations driven by resignations
dete_resignations = dete[dete["separationtype"].str.contains("Resignation")].copy()
tafe_resignations = tafe[tafe["separationtype"] == "Resignation"].copy()


# ## Verify the Data
# 
# Below, we clean and explore the `cease_date` and `dete_start_date` columns to make sure all of the years make sense. We'll use the following criteria:
# 
#    * Since the cease_date is the last year of the person's employment and the `dete_start_date` is the person's first year of employment, it wouldn't make sense to have years after the current date.
#    * Given that most people in this field start working in their 20s, it's also unlikely that the `dete_start_date` was before the year 1940.

# ### DETE `cease_date`

# In[14]:


dete_resignations["cease_date"].value_counts()


# In[15]:


# Extract years; convert to float in order to match `tafe` years
dete_resignations["cease_date"] = dete_resignations["cease_date"].str.split("/").str[-1]
dete_resignations["cease_date"] = dete_resignations["cease_date"].astype(float)


# In[16]:


dete_resignations["cease_date"].value_counts()


# ### DETE `dete_start_date`

# In[17]:


dete_resignations["dete_start_date"].value_counts().sort_values(ascending = False)


# ### TAFE `cease_date`

# In[18]:


tafe_resignations["cease_date"].value_counts().sort_values(ascending = False)


# ### Findings
# 
# The years in each dataset don't fully align. The `tafe_resignations` dataframe contains some cease dates in 2009, but the `dete_resignations` dataframe does not. The `tafe_resignations` dataframe also contains many more cease dates in 2010 than the `dete_resignations` dataframe. Since we aren't concerned with analyzing the results by year, we'll leave them as is.

# # Create New Column
# Since one of our goals is to analyze the association between employees' employment durations and the terms of their resignations, we need to create a new variable that displays the amount of time each employee spent in their position.
# 
# The `tafe_resignation` dataframe already contains an `institute_service` column with this information.
# 
# We can create the same column in the `dete_resignations` dataframe by subtracting the `dete_start_date` column from the `cease_date` column.

# In[19]:


dete_resignations["institute_service"] = dete_resignations["cease_date"] - dete_resignations["dete_start_date"]
dete_resignations["institute_service"].head()


# # Identify Dissatisfied Employees
# 
# Next, we'll identify any employees who resigned because they were dissatisfied. Below are the columns we'll use to categorize employees as "dissatisfied" from each dataframe:
# 
#    * tafe_resignations:
#         * Contributing Factors. Dissatisfaction
#         * Contributing Factors. Job Dissatisfaction
#    * dete_resignations:
#         * job_dissatisfaction
#         * dissatisfaction_with_the_department
#         * physical_work_environment
#         * lack_of_recognition
#         * lack_of_job_security
#         * work_location
#         * employment_conditions
#         * work_life_balance
#         * workload
# 
# If the employee indicated any of the factors above caused them to resign, we'll mark them as dissatisfied in a new column. After our changes, the new dissatisfied column will contain just the following values:
# 
#    * True: indicates a person resigned because they were dissatisfied in some way
#    * False: indicates a person resigned because of a reason other than dissatisfaction with the job
#    * NaN: indicates the value is missing

# We'll update the data so that the contributing factors columns are either True, False, or NaN.

# In[20]:


def update_vals(row):
    # Return NaN for null rows
    if pd.isnull(row):
        return np.nan
    # Return False for unanswered rows, indicating satisfaction
    elif row == "-":
        return False
    else:
        return True

tafe_resignations["dissatisfied"] = tafe_resignations[["Contributing Factors. Dissatisfaction", "Contributing Factors. Job Dissatisfaction"]].applymap(update_vals).any(1, skipna = False)

dete_resignations['dissatisfied'] = dete_resignations[['job_dissatisfaction',
       'dissatisfaction_with_the_department', 'physical_work_environment',
       'lack_of_recognition', 'lack_of_job_security', 'work_location',
       'employment_conditions', 'work_life_balance',
       'workload']].any(1, skipna=False)

# Create new dataframes containing the cleaned and updated data
tafe_resignations_up = tafe_resignations.copy()
dete_resignations_up = dete_resignations.copy()


# In[21]:


# Freqency tables of dissatisfaction for each dataset
tafe_resignations_up["dissatisfied"].value_counts(dropna = False)


# In[22]:


dete_resignations_up["dissatisfied"].value_counts(dropna = False)


# # Combining the Data
# We need to add an institute column to allow us to differentiate the data from each survey after they are combined.  Then, we'll combine the dataframes and drop remaining values we don't need.

# In[23]:


# Add column identify where response came from
dete_resignations_up["institute"] = "DETE"
tafe_resignations_up["institute"] = "TAFE"


# In[24]:


# Combine data
combined = pd.concat([dete_resignations_up, tafe_resignations_up], ignore_index = True)


# In[25]:


# Drop columns with less than 500 non-null values
combined_updated = combined.dropna(thresh = 500, axis = 1).copy()


# # Clean the Service Column
# Next, we'll clean the `institute_service` column and categorize employees according to the following definitions:
# 
#    * New: Less than 3 years in the workplace
#    * Experienced: 3-6 years in the workplace
#    * Established: 7-10 years in the workplace
#    * Veteran: 11 or more years in the workplace

# In[26]:


combined_updated["institute_service"].value_counts(dropna = False)


# In[27]:


# Extract years; convert to float
combined_updated["institute_service"] = combined_updated["institute_service"].astype(str).str.extract(r"(\d+)")
combined_updated["institute_service"] = combined_updated["institute_service"].astype(float)

combined_updated["institute_service"].value_counts()


# A simple function will convert the years to categories. We can apply this function to the `institute_service` column and create a new categorical variable.

# In[28]:


# Classify each employee based on years of service
def transform(row):
    if row >= 11:
        return "Veteran"
    elif 10 >= row >= 7:
        return "Established"
    elif 6 >= row >= 3:
        return "Experienced"
    elif pd.isnull(row):
        return np.nan
    else:
        return "New"

# Create new column
combined_updated["service_cat"] = combined_updated["institute_service"].apply(transform)


# In[29]:


# Frequency table
combined_updated["service_cat"].value_counts()


# # Perform Initial Analysis
# Finally, we can replace the missing values in the `dissatisfied` column with the most common value, `False`.  Then, we can calculate the percentages of `service_cat` column and plot the results.

# In[30]:


combined_updated["dissatisfied"].value_counts(dropna = False)


# In[31]:


# Replace missing values
combined_updated["dissatisfied"] = combined_updated["dissatisfied"].fillna(False)


# In[32]:


# Pivot table of results
combined_pv = combined_updated.pivot_table(index = "service_cat", values = "dissatisfied")
combined_pv


# The above pivot table reveals the percentage of employees in each service category who resigned due to job dissatisfaction. We see that Veteran and Established employees, the two longer-tenured tiers, tended to resign over dissatisfaction at a greater rate.

# In[33]:


# Plot results
get_ipython().run_line_magic('matplotlib', 'inline')
combined_pv.plot(kind = "bar", rot = 30, legend = False, title = "Percentage of Dissatisfied Resignations by Service Category")


# ## Conclusion
# 
# This project extensively cleaned responses from two Australian employee surveys, then combined in order to assess whether resignations are typically motivated by job dissatisfaction.
# 
# From the initial analysis above, we can tentatively conclude that employees with 7 or more years of service are more likely to resign due to some kind of dissatisfaction with the job than employees with less than 7 years of service. However, we need to handle the rest of the missing data to finalize our analysis.
