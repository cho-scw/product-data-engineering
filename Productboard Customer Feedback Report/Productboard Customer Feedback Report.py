#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import pandasql as psql
import numpy as np


# # Create Path & Collect Files

# ## Requirements for Import/Exporting Raw Data
# 
# 1. Please ensure there is a "Data Import" and "Data Exports" folder in the same location where this Jupyter Notebook is running from.
# 
# > E.g. If this Jupyter Notebook sits in `\folder..\Productboard\` then you need a `\folder..\Productboard\Data Import`
# > You will also need a `\folder..\Productboard\Data Exports`
# 
# 2. Rename Productboard raw data to `Productboard Notes Data Export.csv`
# 3. Rename ChurnZero raw data to `ChurnZero - List of Accounts.csv`

# In[2]:


strImportFolder='Productboard Customer Feedback Report'
FileNameChurnZero = 'ChurnZero - List of Accounts.csv'
FileNameProductboard = 'Productboard Notes Data Export.csv'

ImportFolderLocation = os.path.join(os.getcwd(),'Data Import','Productboard Customer Feedback')
ExportFolderLocation = os.path.join(os.getcwd(),'Data Exports','Productboard Customer Feedback')

FilePathChurnZero = os.path.join(ImportFolderLocation,FileNameChurnZero)
FilePathProductboard = os.path.join(ImportFolderLocation,FileNameProductboard)


# # Import Files into Pandas DataFrame

# In[3]:


dfProductboardRaw = pd.read_csv(FilePathProductboard)
dfChurnZeroRaw = pd.read_csv(FilePathChurnZero)


# # Raw Data Schema:
# 
# |Data Source|Field|Additional Notes|
# |-|-|-|
# |Productboard|id||
# |Productboard|created_at||
# |Productboard|note_title||
# |Productboard|note_text||
# |Productboard|state||
# |Productboard|creator_name||
# |Productboard|creator_email||
# |Productboard|owner_name||
# |Productboard|owner_email||
# |Productboard|user_name||
# |Productboard|user_email||
# |Productboard|company_name|To ```JOIN``` against ChurnZero data to get Salesforce Company attributes|
# |Productboard|company_domain||
# |Productboard|source_id|Will be dropped in the Transformed output|
# |Productboard|source_url|Will be dropped in the Transformed output|
# |Productboard|tags|
# |ChurnZero|Name|To ```JOIN``` against Productboard data to provide Salesforce Company attributes|
# |ChurnZero|Tags|Will be dropped in the Transformed output|
# |ChurnZero|Owner||
# |ChurnZero|Next Renewal Date||
# |ChurnZero|Total Contract Amount||
# |ChurnZero|External Id|Will be dropped in the Transformed output|
# |ChurnZero|SSO|Will be dropped in the Transformed output|
# |ChurnZero|Is Active||
# |ChurnZero|Tier||
# |ChurnZero|Industry||
# |ChurnZero|Industry Detail||
# 
# # JOIN Key(s):
# 
# Productboard.company_name -> ChurnZero.Name

# # Create Tall Table for Tags from Productboard Raw Data

# In[4]:


#dfProductboardTags = dfProductboardRaw['tags'].str.split(',', expand=True)
dfProductboardTags = pd.DataFrame(dfProductboardRaw.tags.str.split(', ').tolist(), index=dfProductboardRaw.id)
dfProductboardTags = dfProductboardTags.reset_index()[['id',0]]
dfProductboardTags = dfProductboardTags.rename(columns={
    0: 'tags'
})
dfProductboardTags = dfProductboardTags.apply(lambda x: pd.Series(x['tags']), axis=1).stack().reset_index(level=1, drop=True)
dfProductboardTags.name = 'tags'
dfProductboardTags = pd.DataFrame(dfProductboardTags)


# # Create dfProductboardTransformed DataFrame

# In[5]:


dfProductboardTransformed = dfProductboardRaw.drop(['tags','source_id','source_url'], axis=1).merge(dfProductboardTags, how='left', left_index=True, right_index=True)

#drop note_text column as it can take up too much space
dfProductboardTransformed = dfProductboardTransformed.drop('note_text',axis=1)


# # Calculate Day Difference of when Insights were submitted vs Today

# In[6]:


dfProductboardTransformed['DayDifference'] = pd.to_datetime('now') - dfProductboardTransformed['created_at'].astype('datetime64[ms]')
dfProductboardTransformed['DayDifference'] = dfProductboardTransformed['DayDifference']/np.timedelta64(1,'D')
dfProductboardTransformed['DayDifference'] = dfProductboardTransformed['DayDifference'].astype(int)


# # Create TimeBucket Field

# In[7]:


dfProductboardTransformed['TimeBucket'] = np.select(
    [
        dfProductboardTransformed['DayDifference'].between(-5,1, inclusive=False),
        dfProductboardTransformed['DayDifference'].between(1,5, inclusive=True),
        dfProductboardTransformed['DayDifference'].between(5,10, inclusive=True),
        dfProductboardTransformed['DayDifference'].between(10,30, inclusive=True),
        dfProductboardTransformed['DayDifference'].between(30,60, inclusive=True),
        dfProductboardTransformed['DayDifference'].between(60,90, inclusive=True)
        
    ],
    [
        '1. < 1 Day(s)',
        '2. 1 - 5 Day(s)',
        '3. 6 - 10 Day(s)',
        '4. 11 - 30 Day(s)',
        '5. 31 - 60 Day(s)',
        '6. 61 - 90 Day(s)'
    ],
    default = '7. >90 Day(s)'
)


# # Create Product Groups TRUE/FALSE Filter

# In[8]:


dfProductboardTransformed['ProductGroup'] = np.select(
    [
        dfProductboardTransformed.tags.str.lower() == 'buildsecurely',
        dfProductboardTransformed.tags.str.lower() == 'contentcoverage',
        dfProductboardTransformed.tags.str.lower() == 'grow&innovate',
        dfProductboardTransformed.tags.str.lower() == 'play&learn',
        dfProductboardTransformed.tags.str.lower() == 'visualise&measure',
        dfProductboardTransformed.tags.str.lower() == 'enable&extend'
    ],
    [
        True,
        True,
        True,
        True,
        True,
        True
    ],
    default = False
)


# # Milestone 2 Placeholder
# 
# To create a hierarchy for reporting.
# 
# - E.g. Drilling down from Play&Learn to Courses or Assessments in the end result..

# # JOIN ChurnZero with Productboard data

# In[9]:


dfChurnZeroTransformed = dfChurnZeroRaw
dfChurnZeroTransformed = dfChurnZeroTransformed.drop(['Tags','External Id','SSO'], axis=1)


# In[10]:


dfDataExport = dfProductboardTransformed.merge(dfChurnZeroTransformed, how='left', left_on = 'company_name', right_on = 'Name')

dfDataExport.rename(columns = {'Region':'RegionChurnZero'}, inplace = True)

dfDataExport.loc[dfDataExport['Name'].isnull(),'IsSCWCustomer'] = False
dfDataExport.loc[dfDataExport['Name'].notnull(),'IsSCWCustomer'] = True


# # Create Mapping Table by Requestor x ChurnZero Data

# A.k.a. Popularity contest to resolve feedback from customers that has not signed on with SCW (e.g. trial accounts or currently under sales)
# 
# For more information, please refer to the [Technical Documentation](https://securecodewarrior.atlassian.net/l/c/7Tw0T1nf) in Confluence.

# In[11]:


df = dfDataExport.loc[dfDataExport.owner_email.notnull()]
df = df.loc[df.RegionChurnZero.notnull()]
df.rename(columns = {'RegionChurnZero':'RegionHack'}, inplace = True)
df = pd.DataFrame(df[['creator_email','RegionHack','id']].groupby(['creator_email','RegionHack']).id.nunique())
df.rename(columns = {'id':'RecordCount'}, inplace = True)
sSQL = 'select creator_email, RegionHack, RecordCount, ROW_NUMBER() OVER (PARTITION BY creator_email ORDER BY RecordCount DESC) RankId FROM df GROUP BY creator_email, RegionHack'
df = psql.sqldf(sSQL)
dfRegionMappingTable = df.loc[df.RankId == 1].drop(['RecordCount','RankId'], axis=1)


# # Apply & Collate "Hack" Region Attributes to Main Dataset

# If `Region` in ChurnZero is null, then apply the "popularity contest" compiled in `dfRegionMappingTable`.
# 
# The objective is to have a single field that can be used to easily filter Regions by.
# 
# PS. All staging fields are retained so troubleshooting can be perfomed in the dataset rather than to dig into this code.

# In[12]:


df = dfDataExport.merge(dfRegionMappingTable, how='left', left_on='creator_email', right_on='creator_email')
df['Region'] = np.select(
    [
        df['RegionChurnZero'].isnull()
    ],
    [
        df['RegionHack']
    ],
    default = df['RegionChurnZero']
)
dfDataExport = df


# # Add Timestamp

# Add timestamp to show when the report was generated in this process - to be displayed in GDS report.

# In[15]:


dfDataExport['TimeStamp'] = pd.to_datetime('now')


# # Export CSV File..

# In[17]:


dfDataExport.to_csv(os.path.join(ExportFolderLocation,'Productboard Customer Feedback Data Export Test.csv') , index=False)


# # Data QC Checks

# ## What are the Companies that has no ChurnZero Company Attributes?
# 
# > Basically, this is not an SCW customer that is registered in Salesforce.

# In[18]:


#dfDataExport.loc[dfDataExport.IsSCWCustomer == False]
sSQL = "SELECT DISTINCT company_name FROM dfDataExport where IsSCWCustomer = False ORDER BY 1"
df = psql.sqldf(sSQL)
#df.to_csv(os.path.join(ExportFolderLocation,'Missing Companies.csv'),index=False)


# In[85]:


#dfDataExport.loc[dfDataExport.IsSCWCustomer == False]
sSQL = "SELECT * FROM dfChurnZeroRaw where name like '%webjet%' ORDER BY 1"
df = psql.sqldf(sSQL)
df
#df.to_csv(os.path.join(ExportFolderLocation,'Missing Companies.csv'),index=False)


# # Old Codes

# ## TimeBucket Build..

# In[33]:


dfProductboardTransformed['TimeBucket'] = np.where(
    dfProductboardTransformed['DayDifference'].between(0,1, inclusive=True), '1. < 1 Day(s)',
    np.where(
        dfProductboardTransformed['DayDifference'].between(1,5, inclusive=True), '2. 1 - 5 Day(s)',
        np.where(
            dfProductboardTransformed['DayDifference'].between(5,10, inclusive=True), '3. 6 - 10 Day(s)',
            np.where(
                dfProductboardTransformed['DayDifference'].between(10,30, inclusive=True), '4. 11 - 30 Day(s)',
                np.where(
                    dfProductboardTransformed['DayDifference'].between(30,60, inclusive=True), '5. 31 - 60 Day(s)',
                        np.where(
                            dfProductboardTransformed['DayDifference'].between(60,90, inclusive=True), '6. 61 - 90 Day(s)', '>90 Day(s)'
                    )
                )
            )
        )
    )
)


# ## Count total records

# In[59]:


df = dfDataExport.loc[dfDataExport.owner_email.notnull()]
df = df.loc[df.Region.notnull()]

df[['owner_email','Region','id']].groupby(['owner_email','Region']).agg(['count'])

#df.loc[df.IsSCWCustomer==True]

