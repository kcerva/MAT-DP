#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns
import plotly
import plotly_express as px
# import plotly.plotly as py
from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go
# import matplotlib.patches as mpatches
# from shapely.geometry import Polygon
# import geopandas
# import geoplot
import plotly.io as pio
from os import listdir
from os.path import isfile, join
from pathlib import Path
from os import listdir
from os.path import isfile, join
# import country_converter as coco
# import scipy
# from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
import ipywidgets as widgets
# import statsmodels.formula.api as smf
# import statsmodels.api as sm
# from patsy import dmatrices

# import spacy

get_ipython().run_line_magic('matplotlib', 'inline')


# # Load E+M data

# In[624]:


# CHECK THE DATA TAKEN HERE IS THE RIGHT ONE AFTER THE CHANGES. SAME FOR EXCEL FILE.

# Define matrices and load data
# country energy projection [kWh]
mypath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/"
C = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "C:I", nrows = 1)
#  material per energy technology [g/kWh]
M = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "K:AG", nrows = 10)
M = M.rename(columns={'[g/kWh]':'tech'}).set_index('tech')
# Ref = xlsread('Excel model - material implications energy systems','Matrices','L4:AG12') #reference included

#  embodied emissions (GHG) per material [gCO2e/g]
E = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AJ", nrows = 22)
#  water usage per material [l/kg]
W = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AK", nrows = 22)

#  recycling rate in current supply per material [%]
R = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AL", nrows = 22)
#  costs per material [â‚¬/kg]
K = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AM", nrows = 22)
K = K/1000

# Calculating all effects
# E: emissions per energy technology [gCO2/kWh]
# W: water usage per energy technology
# R: recycling rate per energy technology 
# K: costs per energy technology 
for ef, ne in zip([E, W, R, K],['E', 'W', 'R', 'K'] ):
    #  effect per energy technology e.g. [gCO2/kWh]
    globals()['{}_tech'.format(ne)] = M.dot(ef.values)
    globals()['{}_tech_sep'.format(ne)] = M.multiply(ef.T.values)
    
#     later add the index to C so the tech's names are used in calc
    #  total factor of country e.g. embodied emissions [gCO2]
    globals()['{}_country'.format(ne)] = C.dot(M[0:len(C.T)].values).dot(ef.values)
    globals()['{}_country_sep'.format(ne)] = globals()['{}_tech_sep'.format(ne)][0:len(C.T)].multiply(C.T.values)

    globals()['country_tech_{}'.format(ne)] = C.T.values*(globals()['{}_tech'.format(ne)][0:len(C.T)])
    globals()['country_tech_{}'.format(ne)] = globals()['country_tech_{}'.format(ne)].T
    


# # Load E+M data for all countries

# In[1006]:


# Df w all countries and scenarios
dfC = pd.read_excel(mypath+'EnergyProjection.xlsx', sheet_name = 'All countries', 
                skiprows = 2, usecols = "B:R")
#  material per energy technology [g/kWh]
cM = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "R:AN", nrows = 14)
cM = cM.rename(columns={'[g/kWh]':'tech'}).set_index('tech')

#  embodied emissions (GHG) per material [gCO2e/g]
cE = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AQ", nrows = 22)
#  water usage per material [l/kg]
cW = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AR", nrows = 22)

#  recycling rate in current supply per material [%]
cR = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AS", nrows = 22)
#  costs per material [â‚¬/kg]
cK = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AT", nrows = 22)
cK = cK/1000

# Add employment and land use data


# In[1007]:


#  total factor of all countries e.g. embodied emissions [gCO2]
#  df w all countries, years and scenarios
dfC['Scenario'] = dfC['Scenario'].replace(np.nan, 'Baseline')
dfC['Year'] = dfC['Year'].replace(np.nan,0).astype('int')

# Ordering dfC columns for the same order as M
coln = list(cM.index)

for ef, ne in zip([cE, cW, cR, cK],['E', 'W', 'R', 'K'] ):
    globals()['df{}_tech'.format(ne)] = cM.fillna(0).dot(cE.values)
    globals()['df{}_tech_sep'.format(ne)]  = cM.fillna(0).multiply(cE.T.values)
    
#     'df{ne}_{country}_{scenario}_{year}_sep'
    for c in dfC['Country'].unique():
        for sc in dfC[dfC['Country']==c]['Scenario'].unique():
            for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
                globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] = globals()['df{}_tech_sep'.format(ne)].reindex(index = coln).multiply(dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)&(dfC['Year']==y)][coln].T.values) 
    #             print('df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y)))

#     'df{}_country'.format(ne) 
    globals()['df{}_country'.format(ne)]  = dfC[['Year','Scenario','Country']]
    globals()['df{}_country'.format(ne)] ['value'] = 0

#     'dfcountry_tech_{}'.format(ne)'
    globals()['dfcountry_tech_{}'.format(ne)] = dfC.copy()
    for col in globals()['dfcountry_tech_{}'.format(ne)].iloc[:,3:].columns:
        globals()['dfcountry_tech_{}'.format(ne)][col] = 0

#     'df{}_country'.format(ne) 
    for index, row in dfC[coln].iterrows():
        val = pd.DataFrame(row).fillna(0).T.dot(cM.fillna(0)[0:len(dfC.iloc[:,3:].T)].values).dot(cE.values)
        globals()['df{}_country'.format(ne)] .loc[index,'value'] = val.values[0]

#     'dfcountry_tech_{}'.format(ne)'
        valt = pd.DataFrame(row).fillna(0).values*(eg_E_tech)
        for col in valt.T.columns:
            globals()['dfcountry_tech_{}'.format(ne)].loc[index,col] = valt.T.loc[0,col]
            


# In[887]:


#Example for a single effect calculation
# for c in dfC['Country'].unique():
#     eg_E_tech = cM.fillna(0).dot(cE.values)
#     eg_E_tech_sep = cM.fillna(0).multiply(cE.T.values)
    
#     for sc in dfC[dfC['Country']==c]['Scenario'].unique():
#         for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
#             globals()['egE_{0}_{1}_{2}_sep'.format(c,sc[0:3],str(y))] = eg_E_tech_sep.reindex(index = coln).multiply(dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)&(dfC['Year']==y)][coln].T.values) 
#             print('egE_{0}_{1}_{2}_sep'.format(c,sc[0:3],str(y)))           
            
# egE_country = dfC[['Year','Scenario','Country']]
# egE_country['value'] = 0

# eg_country_tech_E = dfC.copy()
# for col in eg_country_tech_E.iloc[:,3:].columns:
#     eg_country_tech_E[col] = 0

# for index, row in dfC[coln].iterrows():
#     val = pd.DataFrame(row).fillna(0).T.dot(cM.fillna(0)[0:len(dfC.iloc[:,3:].T)].values).dot(cE.values)
#     egE_country.loc[index,'value'] = val.values[0]
    
#     valt = pd.DataFrame(row).fillna(0).values*(eg_E_tech)
#     for col in valt.T.columns:
#         eg_country_tech_E.loc[index,col] = valt.T.loc[0,col]


# # Errors

# In[321]:


# country energy projection [kWh]
e_C = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AQ:AW", nrows = 1)
# material per energy technology [g/kWh]
e_M = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AY:BU", nrows = 10)
# embodied emissions (GHG) per material [gCO2e/g]
e_E = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "BX:BX", nrows = 22)
# water usage per material [l/kg]
e_W = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "BY:BY", nrows = 22)
# recycling rate in current supply per material [%]
e_R = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "BZ:BZ", nrows = 22)
# costs per material [â‚¬/kg]
e_K = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "CA:CA", nrows = 22)
e_K = e_K/1000

for ef in [e_C, e_M, e_E, e_W, e_R, e_K]:
    eco = ef.columns.values
    ef.columns = [str(x).replace('.1','') for x in eco]
e_M = e_M.rename(columns={'[g/kWh]':'tech'}).set_index('tech')

# clean calculations trying .dot
for ef, ne in zip([e_E, e_W, e_R, e_K],['E', 'W', 'R', 'K'] ):
    #seperate errors per material per kWh for different impacts 
    globals()['em_'.format(ne)] = ((ef/globals()['{}'.format(ne)]).pow(2)).T
    globals()['em_'.format(ne)].columns = e_cM.columns
    globals()['em_'.format(ne)] = globals()['em_'.format(ne)].reset_index().drop(columns='index', axis = 1)
    
    t1 = pd.DataFrame([x+y for x in ((e_cM/cM).pow(2)).values for y in globals()['em_'.format(ne)].values]).pow(0.5)
    t1.columns = globals()['{}_tech_sep'.format(ne)].columns
    t1.index = e_M.index
    
#     e_E_tech_sep = E_tech_sep.*sqrt((e_M./M).^2+(e_E'./E').^2)
    globals()['e_{}_tech_sep'.format(ne)] = globals()['{}_tech_sep'.format(ne)].mul(t1, fill_value=0)
    globals()['e_{}_tot'.format(ne)] = globals()['e_{}_tech_sep'.format(ne)].T.sum()
# Do I need to remove any rows with this? [0:len(C.T)]

for ef, ne in zip([e_E, e_W, e_R, e_K],['E', 'W', 'R', 'K'] ):
    #seperate errors per material per kWh for different impacts 
    globals()['em_'.format(ne)] = ((ef/globals()['{}'.format(ne)]).pow(2)).T
    globals()['em_'.format(ne)].columns = e_M.columns
    globals()['em_'.format(ne)] = globals()['em_'.format(ne)].reset_index().drop(columns='index', axis = 1)
    
    # add index of tech to t2, otherwise the calc gives error
    tc = (e_C/C).T
    t2 = pd.DataFrame([x+y+z for x in ((e_M[0:len(C.T)]/M[0:len(C.T)]).pow(2)).values for y in globals()['em_'.format(ne)].values for z in (tc.pow(2))]).pow(0.5)
    t2.columns =  globals()['e_{}_tech_sep'.format(ne)].columns
    t2.index = tc.index
    
    globals()['e_{}_country_sep'.format(ne)] = globals()['{}_country_sep'.format(ne)].mul(t2, fill_value=0)
    globals()['e_{}c_tot'.format(ne)] = globals()['e_{}_country_sep'.format(ne)].T.sum()


# # Errors for all countries

# In[1008]:


# Error or all countries and scenarios
e_dfC = pd.read_excel(mypath+'EnergyProjection.xlsx', sheet_name = 'All countries (e)', 
                skiprows = 2, usecols = "B:R")
# material per energy technology [g/kWh]
e_cM = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "BM:CI", nrows = 14)
# embodied emissions (GHG) per material [gCO2e/g]
e_cE = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CL", nrows = 22)
# water usage per material [l/kg]
e_cW = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CM", nrows = 22)
# recycling rate in current supply per material [%]
e_cR = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CN", nrows = 22)
# costs per material [â‚¬/kg]
e_cK = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CO", nrows = 22)
e_cK = e_cK/1000

# e_dfC
for ef in [ e_cM, e_cE, e_cW, e_cR, e_cK]:
    eco = ef.columns.values
    ef.columns = [str(x).replace('.1','') for x in eco]
e_cM = e_cM.rename(columns={'[g/kWh]':'tech'}).set_index('tech')


# In[1059]:



# Ordering dfC columns for the same order as M
coln = list(cM.index)
techs = ['Wind (Onshore)', 'Solar CSP', 'Solar PV', 'Hydro', 'Geothermal',
       'Gas CCS', 'Oil', 'Gas', 'Coal', 'Wind (Offshore)', 'Biomass',
       'Nuclear', 'BECCS', 'Hydrogen']
mats = list(cM.columns)

for ef, ne in zip([e_cE, e_cW, e_cR, e_cK],['E', 'W', 'R', 'K'] ):
    #seperate errors per material per kWh for different impacts 
    globals()['em_'.format(ne)] = ((ef/globals()['{}'.format(ne)]).pow(2)).T
    globals()['em_'.format(ne)].columns = e_cM.columns
    globals()['em_'.format(ne)] = globals()['em_'.format(ne)].reset_index().drop(columns='index', axis = 1)
    
    t1 = pd.DataFrame([x+y for x in ((e_cM/cM).pow(2)).values for y in globals()['em_'.format(ne)].values]).pow(0.5)
    t1.columns = globals()['df{}_tech_sep'.format(ne)].columns
    t1.index = e_cM.index
    
#     e_E_tech_sep = E_tech_sep.*sqrt((e_M./M).^2+(e_E'./E').^2)
    globals()['e_df{}_tech_sep'.format(ne)] = globals()['df{}_tech_sep'.format(ne)].mul(t1, fill_value=0)
    globals()['e_df{}_tot'.format(ne)] = globals()['e_df{}_tech_sep'.format(ne)].T.sum()
# Do I need to remove any rows with this? [0:len(C.T)]

for ef, ne in zip([e_cE, e_cW, e_cR, e_cK],['E', 'W', 'R', 'K'] ):
    #seperate errors per material per kWh for different impacts 
    globals()['em_'.format(ne)] = ((ef/globals()['{}'.format(ne)]).pow(2)).T
    globals()['em_'.format(ne)].columns = e_cM.columns
    globals()['em_'.format(ne)] = globals()['em_'.format(ne)].reset_index().drop(columns='index', axis = 1)
    
    globals()['e_df{0}_countries_sep'.format(ne)] = pd.DataFrame(columns =['Country','Year','Scenario','tech']+mats )
    globals()['e_df{0}c_tot'.format(ne)] = pd.DataFrame(columns =['Country','Year','Scenario']+techs )
    for c in dfC['Country'].unique():
        for sc in dfC[dfC['Country']==c]['Scenario'].unique():
            for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
                # add index of tech to t2, otherwise the calc gives error
                tc = (e_dfC[(e_dfC['Country']==c)&(e_dfC['Scenario']==sc)&(e_dfC['Year']==y)][coln]/dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)&(dfC['Year']==y)])[coln].T
                t2 = pd.DataFrame([x+y+z for x in ((e_cM/cM).pow(2)).values for y in globals()['em_'.format(ne)].values for z in (tc.pow(2))]).pow(0.5)
                t2.columns =  globals()['e_{}_tech_sep'.format(ne)].columns
                t2.index = tc.index
                
                news = pd.DataFrame(globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].mul(t2, fill_value=0))
                news = news.reset_index()
                news['Country'] = c
                news['Scenario'] = sc
                news['Year'] = y
                globals()['e_df{0}_countries_sep'.format(ne)] = globals()['e_df{0}_countries_sep'.format(ne)].append(news)
                globals()['e_df{0}_countries_sep'.format(ne)] = globals()['e_df{0}_countries_sep'.format(ne)][['Country','Year','Scenario','tech']+mats]
                
                new = pd.DataFrame(globals()['e_df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].T.sum()).T
                new['Country'] = c
                new['Scenario'] = sc
                new['Year'] = y
                globals()['e_df{0}c_tot'.format(ne)] = globals()['e_df{0}c_tot'.format(ne)].append(new)
                globals()['e_df{0}c_tot'.format(ne)] =globals()['e_df{0}c_tot'.format(ne)][['Country','Year','Scenario']+techs]
                


# # Employment and land-use

# In[1134]:


# Employment by stage [job-years/MW] Manuf and C&I and Dec, [jobs/MW] for O&M, [jobs/PJ] for fuel
jobs = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Employment', 
                skiprows = 37, usecols = "B:G,T", nrows = 23)
jobs.columns = ['tech', 'Manufacturing', 'Construction and Installation',
       'Operation and Maintenance', 'Fuel', 'Decommissioning', 'Total']

# In [jobs-lifetime/kWh]
op_jobs = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Employment', 
                skiprows = 65, usecols = "B:E", nrows = 23)
op_jobs.columns = ['tech','Capacity (MW)', 'kWh/lifetime','Operation and Maintenance']

jobs = jobs.merge(op_jobs[['tech','Capacity (MW)','kWh/lifetime']], on = 'tech')
# Ram2020 has Regional Employment multipliers that should be useful for country evaluations

# Land required [km^2/1000 MW]
land = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Employment', 
                skiprows = 19, usecols = "B:C", nrows = 14)
land.columns = ['tech','land']

jobs['Operation and Maintenance (jobs)'] = jobs['Capacity (MW)'] * jobs['Operation and Maintenance']


# In[ ]:


# employment * kWh/tech/country


# # Plots for single country values

# ## Per kWh

# In[616]:


#  per KWH
colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Embodied Material Emissions [gCO<sub>2</sub>/kWh]','Embodied Material Water Usage [L/kWh]',
          'Material Costs [Euro‚cent/kWh]','Material Recycling Rate [g/kWh]']

for ne, l in zip(['E', 'W', 'K', 'R'], labels):
    fig = go.Figure()
    
    if ne == 'E' or ne == 'R':
        data = globals()['{}_tech_sep'.format(ne)]
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        
        datae.loc[datae.index,'Stainless Steel'] = error.values/2
        datae.index = data.index
    if ne == 'W':
        data = globals()['{}_tech_sep'.format(ne)]/1000
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values/2000
        datae.index = data.index
        
    if ne == 'K':
        data = globals()['{}_tech_sep'.format(ne)]*1000
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values/0.002
        datae.index = data.index
                
    for i, c in zip(data.columns, colors): 
#         print(i)
        fig.add_trace(go.Bar(x = data[i], 
                             y = data.index,
                             name = i, 
                             marker_color = c, orientation = 'h',
                             error_x=dict(type='data', 
                                      array=datae[i])
                            ))

    fig.update_layout( xaxis_title =  l, 
                      template = 'simple_white+presentation', barmode='stack')

    pio.write_image(fig, figpath+"kWh_{}.pdf".format(ne), width = 500, height = 450)
    pio.write_image(fig, figpath+"kWh_{}.eps".format(ne), width = 500, height = 450)
    plotly.offline.plot(fig, filename = figpath+"kWh_{}.html".format(ne), auto_open=False)
#     fig.show()


# ## For Country, all factors

# In[559]:


# for Country
colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Embodied Material Emissions [MtCO<sub>2</sub>]','Embodied Material Water Usage [L]',
          'Material Costs [million Euro]','Material Recycling Rate [t]']

for  ne, l in zip(['E', 'W', 'K', 'R'], labels):
    fig = go.Figure()
    
    if ne == 'E':
        data = globals()['{}_country_sep'.format(ne)]/1e12
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_country_sep'.format(ne)])), 
                     columns=globals()['{}_country_sep'.format(ne)].columns)
        error = globals()['e_{}c_tot'.format(ne)]
        
        datae.loc[datae.index,'Stainless Steel'] = error.values/1e12/2
        datae.index = data.index
    if ne == 'W':
        data = globals()['{}_country_sep'.format(ne)]/1000
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_country_sep'.format(ne)])), 
                     columns=globals()['{}_country_sep'.format(ne)].columns)
        error = globals()['e_{}c_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values/1000/2
        datae.index = data.index
        
    if ne == 'K'  or ne =='R':
        data = globals()['{}_country_sep'.format(ne)]/1e6
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_country_sep'.format(ne)])), 
                     columns=globals()['{}_country_sep'.format(ne)].columns)
        error = globals()['e_{}c_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values/1e6/2
        datae.index = data.index
                
    for i, c in zip(data.columns, colors): 
#         print(i)
        fig.add_trace(go.Bar(x = data[i], 
                             y = data.index,
                             name = i, 
                             marker_color = c, orientation = 'h',
                             error_x=dict(type='data', 
                                      array=datae[i])
                            ))

    fig.update_layout( xaxis_title =  l, 
                      legend=dict(orientation="h", yanchor="bottom", 
#                                   traceorder="reversed",
                                   y=1.01,
                                   xanchor="center",
                                   x=0.4),
                                   template = 'simple_white+presentation', barmode='stack')

    pio.write_image(fig, figpath+"country_{}.pdf".format(ne), width = 900, height = 550)
    pio.write_image(fig, figpath+"country_{}.eps".format(ne), width = 900, height = 550)
    plotly.offline.plot(fig, filename = figpath+"country_{}.html".format(ne), auto_open=False)
#     fig.show()


# ## Totals and percentages in country

# In[615]:


#  Bar with totals and total percentages for Country
colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Total material mass accross all technologies for Uganda (%)',
          'Total CO <sub>2</sub> emissions accross all technologies for Uganda (%)',
          'Total material costs accross all technologies for Uganda [million Euro]']
names = ['material_mass', 'total_emiss', 'total_mat']
colu = ['pct','pct','value']

# emissions per energy technology [gCO2/kWh]
mat_country = C.T.values*M[0:len(C.T)]

for l, n, co in zip( labels, names, colu):
    fig = go.Figure()
    if n == 'material_mass':
        totaal = mat_country.sum().reset_index()
        totaal = totaal.rename(columns = {'index':'Material', 0:'value'})
       
    if n == 'total_emiss':
        totaal = E_country_sep.sum().reset_index()   
        totaal = totaal.rename(columns = {'index':'Material', 0:'value'})
        
    if ne == 'total_mat':
        totaal = K_country_sep.sum().reset_index()
        totaal = totaal/1e6.rename(columns = {'index':'Material', 0:'value'})
        
    ts = totaal['value'].sum().round(2)
    totaal.loc[totaal.index,'pct'] = totaal.loc[totaal.index,'value']/ts*100
    tsdf = totaal[['Material','value']]
    tsdf.loc[totaal.index,'value']=np.nan
    tsdf.loc[totaal['Material']=='Stainless Steel','value']=ts
    
    for i, c in zip(totaal.Material.unique(), colors): 
#         print(i)
        fig.add_trace(go.Bar(x = totaal[totaal['Material']==i][co], 
                             y = ['Total'],
                             width=0.5,
                             name = i ,
                             marker_color = c, 
                             orientation = 'h',
                             text = tsdf[totaal['Material']==i], textposition='auto'
                                 ))

    fig.update_layout( xaxis_title =  l, 
                      legend=dict(orientation="h", yanchor="bottom",
#                                   traceorder="reversed",
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                                    template = 'simple_white+presentation', barmode='stack')


#     fig.update_xaxes(range=[0,110])

    pio.write_image(fig, figpath+"country_{}.pdf".format(n), width = 800, height = 500)
    pio.write_image(fig, figpath+"country_{}.eps".format(n), width = 800, height = 500)
    plotly.offline.plot(fig, filename = figpath+"country_{}.html".format(n), auto_open=False)
#     fig.show()


# # Plots comparing countries/scenarios

# ## Per kWh

# In[ ]:


#  per KWH
colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Embodied Material Emissions [gCO_2/kWh]','Embodied Material Water Usage [L/kWh]',
          'Material Costs [Euro‚cent/kWh]','Material Recycling Rate [g/kWh]']

for ne, l in zip(['E', 'W', 'K', 'R'], labels):
    fig = go.Figure()
    
    if ne == 'E' or ne == 'R':
        data = globals()['{}_tech_sep'.format(ne)]
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        
        datae.loc[datae.index,'Stainless Steel'] = error.values/2
        datae.index = data.index
    if ne == 'W':
        data = globals()['{}_tech_sep'.format(ne)]/1000
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values/2000
        datae.index = data.index
        
    if ne == 'K':
        data = globals()['{}_tech_sep'.format(ne)]*1000
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values/0.002
        datae.index = data.index
                
    for i, c in zip(data.columns, colors): 
#         print(i)
        fig.add_trace(go.Bar(x = data[i], 
                             y = data.index,
                             name = i, 
                             marker_color = c, orientation = 'h',
                             error_x=dict(type='data', 
                                      array=datae[i])
                            ))

    fig.update_layout( xaxis_title =  l, 
                      template = 'simple_white+presentation', barmode='stack')

    pio.write_image(fig, figpath+"kWh_{}.pdf".format(ne), width = 500, height = 450)
    pio.write_image(fig, figpath+"kWh_{}.eps".format(ne), width = 500, height = 450)
    plotly.offline.plot(fig, filename = figpath+"kWh_{}.html".format(ne), auto_open=False)
#     fig.show()


# ## For Country, all factors

# In[1072]:


# for Country
# make subplots for all scenario/year combos for each country
# MISSING: ADD ERROR DATA
# for ef, ne in zip([cE, cW, cR, cK],['E', 'W', 'R', 'K'] ):
    
colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/Countries/"

labels = ['Embodied Material Emissions [MtCO<sub>2</sub>]','Embodied Material Water Usage [L]',
          'Material Costs [million Euro]','Material Recycling Rate [t]']
atechs = ['Wind (Onshore)', 'Solar CSP', 'Solar PV', 'Hydro', 'Geothermal',
       'Gas CCS', 'Oil', 'Gas', 'Coal', 'Wind (Offshore)', 'Biomass',
       'Nuclear', 'BECCS', 'Hydrogen']
mats = list(cM.columns)
# e_dfK_countries_sep
# e_dfEc_tot

# make list w names for plot titles
for c in dfC['Country'].unique():
    globals()['l_{}'.format(c)] = []
    for sc, y in zip(dfC[dfC['Country']==c]['Scenario'],dfC[dfC['Country']==c]['Year']):
        globals()['l_{}'.format(c)].append('{0} {1}'.format(y,sc))

                
for  ne, l in zip(['E', 'W', 'K', 'R'], labels):
    #     'df{ne}_{country}_{scenario}_{year}_sep'
    for c in dfC['Country'].unique():
        fig = make_subplots(rows=len(dfC[dfC['Country']==c]['Scenario'].unique()), cols=1,
                           subplot_titles=globals()['l_{}'.format(c)],
                            x_title =  l, shared_xaxes = True
                           )
        print(c)
        h = len(globals()['l_{}'.format(c)])*310
        slegend = [True, False, False, False, False]
        for sc, ns,sl in zip(dfC[dfC['Country']==c]['Scenario'].unique(),range(1,len(dfC[dfC['Country']==c]['Scenario'].unique())+1),slegend):
            print(sc,ns)
            for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
#                 range(1,len(dfC[dfC['Country']==c]['Scenario'].unique())+1)
#                 globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] 

                    if ne == 'E':
                        data = globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] /1e12
                        datae = pd.DataFrame(np.nan, 
                                             index=np.arange(len(globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],
                                                                                                          str(y))])), 
                                     columns=globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].columns)
                #         ERROR
                        error = globals()['e_df{}c_tot'.format(ne)]
                        error = error[(error['Country']==c)&(error['Scenario']==sc)&(error['Year']==y)][atechs]
                        datae.loc[datae.index,'Stainless Steel'] = error.values/1e12/2
                        datae.index = data.index
                    if ne == 'W':
                        data = globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] /1000
                        datae = pd.DataFrame(np.nan, 
                                             index=np.arange(len(globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,
                                                                                                          sc[0:3],str(y))])), 
                                     columns=globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].columns)
#                         ERROR
                        error = globals()['e_df{}c_tot'.format(ne)]
                        error = error[(error['Country']==c)&(error['Scenario']==sc)&(error['Year']==y)][atechs]
                        datae.loc[datae.index,'Stainless Steel'] = error.values/1000/2
                        datae.index = data.index

                    if ne == 'K'  or ne =='R':
                        data = globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] /1e6
                        datae = pd.DataFrame(np.nan, 
                                             index=np.arange(len(globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,
                                                                                                          sc[0:3],str(y))])), 
                                     columns=globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].columns)
                #         ERROR
                        error = globals()['e_df{}c_tot'.format(ne)]
                        error = error[(error['Country']==c)&(error['Scenario']==sc)&(error['Year']==y)][atechs]
                        datae.loc[datae.index,'Stainless Steel'] = error.values/1e6/2
                        datae.index = data.index

                    for i, cl in zip(data.columns, colors): 
                #         print(i)
                        d = data.T.sum().reset_index()
                        techs = d[d[0]>0]['tech']
                        
                        fig.add_trace(go.Bar(x = data.loc[techs][i], 
                                             y = data.loc[techs].index,
                                             name = i, 
                                             marker_color = cl, orientation = 'h',
                                             showlegend = sl,
                                             error_x=dict(type='data', 
                                                      array=datae.loc[techs][i])
                                            ),
                                     row = ns, col = 1)

                    fig.update_layout( 
#                         xaxis_title =  l, 
#                                       legend=dict(
#                                                   orientation="h", 
#                                                   yanchor="bottom", 
#                 #                                   traceorder="reversed",
#                                                    y=1.01,
#                                                    xanchor="center",
#                                                    x=0.4),
                                      template = 'simple_white+presentation', barmode='stack')
                    fig.update_layout(legend = dict(font = dict(size = 15)))
                    

        pio.write_image(fig, figpath+"{0}_{1}.pdf".format(c,ne), width = 900, height = h)
        pio.write_image(fig, figpath+"{0}_{1}.eps".format(c,ne), width = 900, height = h)
        plotly.offline.plot(fig, filename = figpath+"{0}_{1}.html".format(c,ne), auto_open=False)
#         fig.show()


# ## Totals and percentages in country

# In[ ]:


#  Bar with totals and total percentages for Country
colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Total material mass accross all technologies for Uganda (%)',
          'Total CO <sub>2</sub> emissions accross all technologies for Uganda (%)',
          'Total material costs accross all technologies for Uganda [million Euro]']
names = ['material_mass', 'total_emiss', 'total_mat']
colu = ['pct','pct','value']

# emissions per energy technology [gCO2/kWh]
mat_country = C.T.values*M[0:len(C.T)]

for l, n, co in zip( labels, names, colu):
    fig = go.Figure()
    if n == 'material_mass':
        totaal = mat_country.sum().reset_index()
        totaal = totaal.rename(columns = {'index':'Material', 0:'value'})
       
    if n == 'total_emiss':
        totaal = E_country_sep.sum().reset_index()   
        totaal = totaal.rename(columns = {'index':'Material', 0:'value'})
        
    if ne == 'total_mat':
        totaal = K_country_sep.sum().reset_index()
        totaal = totaal/1e6.rename(columns = {'index':'Material', 0:'value'})
        
    ts = totaal['value'].sum().round(2)
    totaal.loc[totaal.index,'pct'] = totaal.loc[totaal.index,'value']/ts*100
    tsdf = totaal[['Material','value']]
    tsdf.loc[totaal.index,'value']=np.nan
    tsdf.loc[totaal['Material']=='Stainless Steel','value']=ts
    
    for i, c in zip(totaal.Material.unique(), colors): 
#         print(i)
        fig.add_trace(go.Bar(x = totaal[totaal['Material']==i][co], 
                             y = ['Total'],
                             width=0.5,
                             name = i ,
                             marker_color = c, 
                             orientation = 'h',
                             text = tsdf[totaal['Material']==i], textposition='auto'
                                 ))

    fig.update_layout( xaxis_title =  l, 
                      legend=dict(orientation="h", yanchor="bottom",
#                                   traceorder="reversed",
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                                    template = 'simple_white+presentation', barmode='stack')


#     fig.update_xaxes(range=[0,110])

    pio.write_image(fig, figpath+"country_{}.pdf".format(n), width = 800, height = 500)
    pio.write_image(fig, figpath+"country_{}.eps".format(n), width = 800, height = 500)
    plotly.offline.plot(fig, filename = figpath+"country_{}.html".format(n), auto_open=False)
#     fig.show()


# ## Employment and tech

# In[ ]:





# # Plots for general values

# ## Embodied vs use-phase

# In[1103]:


# Embodied vs use phase

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

EU = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Graph', 
                skiprows = 44, usecols = "B,T:V", nrows = 15)
EU.rename(columns = {'Unnamed: 1':'tech'}, inplace = True)
EU = EU[EU['tech']!= 'Wind']

EE_u = pd.read_excel(mypath+'Excel_model_material_implications_energy_systems.xlsx', 
                   sheet_name = 'Graph', 
                skiprows = 63, usecols = "B:C", nrows = 15)
EE_u.rename(columns = {'Unnamed: 1':'tech'}, inplace = True)

# EE is the same as e_E_tot

colors = px.colors.qualitative.Vivid

df = [EU,E_tech]
dnames = ['EU','E_tech']

datae_u = EU[['tech','STANDARD ERROR']].rename(columns = {'STANDARD ERROR':'error'})
datae_u = datae_u[~datae_u['tech'].isin(['Hydrogen','BECCS'])]
datae_e = datae.append(e_E_tot.reset_index().rename(columns = {0:'error'}))
datae_e = datae_e[~datae_e['tech'].isin(['Hydrogen','BECCS'])]

# sorting data based on use phase

fig = go.Figure()
for d, dn, c in zip(df, dnames, colors[0:2]): 
    if dn == 'EU': 
        d = EU.sort_values(by = 'AVERAGE', ascending = True)
        d = d[~d['tech'].isin(['Hydrogen','BECCS'])]
        fig.add_trace(go.Bar(x = d.AVERAGE, 
                             y = d.tech,
                             name = 'Use-Phase',
                             marker_color = c, 
                             orientation = 'h',
                             error_x=dict(type='data', 
                                      array=datae_u[['error']])
                                 ))
    if dn == 'E_tech': 
        d = E_tech.reset_index().rename(columns = {0:'energy'})
        d = d[~d['tech'].isin(['Hydrogen','BECCS'])]
        fig.add_trace(go.Bar(x = d.energy, 
                             y = d.tech,
                             name = 'Embodied (materials)' ,
                             marker_color = c, 
                             orientation = 'h',
                             error_x=dict(type='data', 
                                      array=datae_e['error'])
                                 ))
fig.update_layout(xaxis_title =  'Emissions per kWh [gCO<sub>2</sub>/kWh]', 
                  template = 'simple_white+presentation', barmode='stack',
                  legend=dict(orientation="h", y=-0.2))

pio.write_image(fig, figpath+"embodied_direct_CO2.pdf", width = 900, height = 700)
pio.write_image(fig, figpath+"embodied_direct_CO2.eps", width = 900, height = 700)
plotly.offline.plot(fig, filename = figpath+"embodied_direct_CO2.html", auto_open=False)
fig.show()


# ## General - Embodied, water and costs

# In[613]:


# Embodied, water and costs

colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Embodied Material Emissions [gCO<sub>2</sub>/kWh]','Water Usage [L/kWh]',
         'Costs [Euro cent/kWh]','Recycling Rate [g/kWh]']

for  ne, l in zip(['E', 'W', 'K', 'R'], labels):
    fig = go.Figure()
    
    if ne == 'E' or ne =='R':
        data = globals()['{}_tech_sep'.format(ne)]
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        
        datae.loc[datae.index,'Stainless Steel'] = error.values/2
        datae.index = data.index
    if ne == 'W':
        data = globals()['{}_tech_sep'.format(ne)]/1000
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values/1000/2
        datae.index = data.index
        
    if ne == 'K':
        data = globals()['{}_tech_sep'.format(ne)]*1000
        datae = pd.DataFrame(np.nan, index=np.arange(len(globals()['{}_tech_sep'.format(ne)])), 
                     columns=globals()['{}_tech_sep'.format(ne)].columns)
        error = globals()['e_{}_tot'.format(ne)]
        datae.loc[datae.index,'Stainless Steel'] = error.values*1000/2
        datae.index = data.index
                
    for i, c in zip(data.columns, colors): 
#         print(i)
        fig.add_trace(go.Bar(x = data[i], 
                             y = data.index,
                             name = i, 
                             marker_color = c, orientation = 'h',
                             error_x=dict(type='data', 
                                      array=datae[i])
                            ))

    fig.update_layout( xaxis_title =  l, 
                      legend=dict(orientation="h", yanchor="bottom", 
#                                   traceorder="reversed",
                                   y=1.01,
                                   xanchor="center",
                                   x=0.4),
                                   template = 'simple_white+presentation', barmode='stack')

    pio.write_image(fig, figpath+"general_{}.pdf".format(ne), width = 1000, height = 700)
    pio.write_image(fig, figpath+"general_{}.eps".format(ne), width = 1000, height = 700)
    plotly.offline.plot(fig, filename = figpath+"general_{}.html".format(ne), auto_open=False)
#     fig.show()


# ## General - Materials per kWh and embodied-material-emissions per kWh

# In[614]:


## Materials per kWh and embodied-material-emissions per kWh
colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Materials [g/kWh]','Embodied Material Emissions [gCO<sub>2</sub>/kWh]']

for  ne, l in zip(['Mat','Em'], labels):
    fig = go.Figure()
    
    if ne == 'Mat':
        data = M# /1e12 this would be for Mtonne
        datae = pd.DataFrame(np.nan, index=np.arange(len(M)), 
                     columns=M.columns)
        error = e_M.T.sum()         
        datae.loc[datae.index,'Stainless Steel'] = error/2 #/1e12 this would be for Mtonne
        datae.index = data.index
#         the Mat error needs checking
        
    if ne == 'Em':
        data = E_tech_sep# /1e12 this would be for Mtonne
        datae = pd.DataFrame(np.nan, index=np.arange(len(E_tech_sep)), 
                     columns=E_tech_sep.columns)
        error = e_E_tech_sep.T.sum()
        datae.loc[datae.index,'Stainless Steel'] = error/2
        datae.index = data.index
            
    data = data.sort_values(by = 'tech')
    for i, c in zip(data.columns, colors): 
        fig.add_trace(go.Bar(x = data[i], 
                             y = data.index,
                             name = i, 
                             marker_color = c, orientation = 'h',
                             error_x=dict(type='data', 
                                      array=datae[i])
                            ))

    fig.update_layout( xaxis_title =  l, 
                      legend=dict(orientation="h", yanchor="bottom", 
#                                   traceorder="reversed",
                                   y=1.01,
                                   xanchor="center",
                                   x=0.4),
                                   template = 'simple_white+presentation', barmode='stack')
    
    pio.write_image(fig, figpath+"general_{}.pdf".format(ne), width = 1000, height = 600)
    pio.write_image(fig, figpath+"general_{}.eps".format(ne), width = 1000, height = 600)
    plotly.offline.plot(fig, filename = figpath+"general_{}.html".format(ne), auto_open=False)
#     fig.show()


# ## General - Bars with total percentage of grams for materials and total percentage of CO2 for materials

# In[608]:


## Bar with total percentage of material mass and total CO2 for materials general

colors = px.colors.qualitative.Alphabet

figpath = "C:/Users/KarlaC/Dropbox (Cambridge University)/CCG/Electricity and material demand model/Figs/"

labels = ['Total material mass accross all technologies (%)',
          'Total CO<sub>2</sub> emissions accross all technologies (%)']
names = ['material_mass', 'total_emiss']
colu = ['pct','pct']

for l, n, co in zip( labels, names, colu):
    fig = go.Figure()
    if n == 'material_mass':
        totaal = M.sum().reset_index()
        totaal = totaal.rename(columns = {'index':'Material', 0:'value'})
       
    if n == 'total_emiss':
        totaal = E_tech_sep.sum().reset_index()   
        totaal = totaal.rename(columns = {'index':'Material', 0:'value'})
        
        
    ts = totaal['value'].sum().round(2)
    totaal.loc[totaal.index,'pct'] = totaal.loc[totaal.index,'value']/ts*100
    tsdf = totaal[['Material','value']]
    tsdf.loc[totaal.index,'value']=np.nan
    tsdf.loc[totaal['Material']=='Stainless Steel','value']=ts
    
    for i, c in zip(totaal.Material.unique(), colors): 
#         print(i)
        fig.add_trace(go.Bar(x = totaal[totaal['Material']==i][co], 
                             y = ['Total'],
                             width=0.5,
                             name = i ,
                             marker_color = c, 
                             orientation = 'h',
                             text = tsdf[totaal['Material']==i], textposition='auto'
                                 ))

    fig.update_layout( xaxis_title =  l, 
                      legend=dict(orientation="h", yanchor="bottom",
#                                   traceorder="reversed",
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                                    template = 'simple_white+presentation', barmode='stack')

#     fig.update_xaxes(range=[0,110])

    pio.write_image(fig, figpath+"general_{}.pdf".format(n), width = 800, height = 500)
    pio.write_image(fig, figpath+"general_{}.eps".format(n), width = 800, height = 500)
    plotly.offline.plot(fig, filename = figpath+"general_{}.html".format(n), auto_open=False)
#     fig.show()

