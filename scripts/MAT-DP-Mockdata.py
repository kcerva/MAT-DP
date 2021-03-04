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
import plotly.io as pio
from os import listdir
from os.path import isfile, join
from pathlib import Path
from os import listdir
from os.path import isfile, join
# import country_converter as coco
from plotly.subplots import make_subplots
import ipywidgets as widgets
import os
import glob


# In[2]:


# Setting working directory
# os.chdir ('C:\\Users\\KarlaC\\MAT-DP\\')


# In[3]:


# Make folders for figures
if not os.path.exists('figures'):
    os.makedirs('figures')
    
if not os.path.exists('figures\countries'):
    os.makedirs('figures\countries')
if not os.path.exists('outputs'):
    os.makedirs('outputs')


# # Load E+M data

# In[9]:


# Define matrices and load data
# country energy projection [kWh]
C = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "C:I", nrows = 1)
#  material per energy technology [g/kWh]
M = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "K:AG", nrows = 10)
M = M.rename(columns={'[g/kWh]':'tech'}).set_index('tech')

#  embodied emissions (GHG) per material [gCO2e/g]
E = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AJ", nrows = 22)
#  water usage per material [l/kg]
W = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AK", nrows = 22)

#  recycling rate in current supply per material [%]
R = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AL", nrows = 22)
#  costs per material [â‚¬/kg]
K = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices', 
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
    
#     improve later: add the index to C so the tech's names are used in calc
    #  total factor of country e.g. embodied emissions [gCO2]
    globals()['{}_country'.format(ne)] = C.dot(M[0:len(C.T)].values).dot(ef.values)
    globals()['{}_country_sep'.format(ne)] = globals()['{}_tech_sep'.format(ne)][0:len(C.T)].multiply(C.T.values)

    globals()['country_tech_{}'.format(ne)] = C.T.values*(globals()['{}_tech'.format(ne)][0:len(C.T)])
    globals()['country_tech_{}'.format(ne)] = globals()['country_tech_{}'.format(ne)].T
    


# # Load E+M data for all countries

# In[10]:


# Df w all countries and scenarios
dfC = pd.read_excel(r'data/EnergyProjection.xlsx', sheet_name = 'All countries', 
                skiprows = 2, usecols = "B:R")
#  material per energy technology [g/kWh]
cM = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "R:AN", nrows = 15)
cM = cM.rename(columns={'[g/kWh]':'tech'}).set_index('tech')

#  embodied emissions (GHG) per material [gCO2e/g]
cE = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AQ", nrows = 22)
#  water usage per material [l/kg]
cW = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AR", nrows = 22)

#  recycling rate in current supply per material [%]
cR = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AS", nrows = 22)
#  costs per material [â‚¬/kg]
cK = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "AT", nrows = 22)
cK = cK/1000

#  total factor of all countries e.g. embodied emissions [gCO2]
#  df w all countries, years and scenarios
dfC['Scenario'] = dfC['Scenario'].replace(np.nan, 'Baseline')
dfC['Year'] = dfC['Year'].replace(np.nan,0).astype('int')

# Ordering dfC columns for the same order as M
coln = list(cM.index)


# In[116]:


# Df w TEMBA results
shnames = ['results_ref','results_1.5deg','results_2.0deg']
power = ['Power Generation (Aggregate)','Power Generation Capacity (Aggregate)','New power generation capacity (Aggregate)']
upow = ['PJ','GW','GW']
results = pd.DataFrame(columns = ['variable', 'scenario', 'country', 'parameter', '2015', '2016', '2017',
                                   '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026',
                                   '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035',
                                   '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044',
                                   '2045', '2046', '2047', '2048', '2049', '2050', '2051', '2052', '2053',
                                   '2054', '2055', '2056', '2057', '2058', '2059', '2060', '2061', '2062',
                                   '2063', '2064', '2065']
                      )
for sn in shnames:
    globals()[sn] = pd.read_csv(r'data/{}.csv'.format(sn))
    globals()[sn] = globals()[sn].drop(columns ='Unnamed: 0')
    globals()[sn] = globals()[sn][globals()[sn]['parameter'].isin(power)]
    results = results.append(globals()[sn])

dtech = pd.DataFrame(list(zip(['Wind (Onshore)', 'Wind (Offshore)', 'Solar CSP', 'Solar PV', 
                             'Hydro', 'Geothermal','Gas CCS', 'Oil', 
                             'Gas', 'Coal', 'Biomass','Nuclear',
                             'BECCS', 'Hydrogen','Coal CCS'],
                            ['Wind','Wind','Solar CSP', 'Solar PV',
                             'Hydro','Geothermal','Gas with ccs', 'Oil',
                             'Gas','Coal','Biomass','Nuclear',
                             'Biomass with ccs', 'Hydrogen' ,'Coal with ccs'])),
                     columns = ['tech','variable']
                    )

dtech = dtech.set_index('variable')['tech'].to_dict()
results['tech']= results['variable'].map(dtech)

# 'power_trade' include as non-country embodied emissions

results_df = pd.melt(results.drop(columns = 'variable'), 
                     id_vars = ['tech', 'scenario','country','parameter'], 
                     var_name = 'Year', value_name = 'Value')

results_piv = pd.pivot_table(results_df, values = 'Value', 
                                      index = ['Year', 'scenario', 'country','parameter'], 
                                      columns = 'tech', aggfunc=np.sum)
results_piv = results_piv.reset_index()

generation_df = results_df[results_df['parameter']== 'Power Generation (Aggregate)']
generation_df['Value'] = [x*277778000 for x in generation_df['Value']]
generation_piv = pd.pivot_table(generation_df, values = 'Value', 
                                      index = ['Year', 'scenario', 'country','parameter'], 
                                      columns = 'tech', aggfunc=np.sum)
generation_piv = generation_piv.reset_index()

generation_piv = generation_piv.drop(columns = 'parameter')
generation_piv.rename(columns ={'scenario':'Scenario','country':'Country'}, inplace = True)


# ## Appending Uganda, UK and TEMBA data

# In[42]:


dfC['Coal CCS'] = 0
dfC = dfC.append(generation_piv[generation_piv['Country'].isin(['KE','RW','ZA'])]) #mockdata
dfC = dfC[['Year', 'Scenario', 'Country', 'Wind (Onshore)', 'Wind (Offshore)',
       'Solar CSP', 'Solar PV', 'Hydro', 'Oil', 'Gas CCS', 'Gas', 'Nuclear',
       'Geothermal', 'Coal', 'Coal CCS', 'Biomass', 'BECCS', 'Hydrogen']]


# ## Calculating mass of materials for all countries

# In[110]:


mat_country = pd.DataFrame(columns = ['Year', 'Scenario', 'Country','tech',
                                      'Aluminium', 'Bentonite', 'Carbon Fiber', 'Cast Iron', 'Cement',
                                      'Ceramics', 'Concrete', 'Copper', 'Epoxy', 'EVA ', 'Fibre Glass',
                                      'Glass', 'Lubricant', 'Non-Ferrous Metal', 'Paint', 'Plastic', 'PVC',
                                      'Resin', 'Sand', 'Silicon', 'Steel', 'Stainless Steel'] )
coln = [ 'Wind (Onshore)', 'Wind (Offshore)',
       'Solar CSP', 'Solar PV', 'Hydro', 'Oil', 'Gas CCS', 'Gas', 'Nuclear',
       'Geothermal', 'Coal', 'Biomass', 'BECCS', 'Hydrogen', 'Coal CCS']
# [coln]
for c in dfC['Country'].unique():
    for sc in dfC[dfC['Country']==c]['Scenario'].unique():
        for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
            mat_c = dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)&(dfC['Year']==y)][coln].                        T.values*cM
#         mat_c = mat_c.sum().reset_index()
        mat_c['Country'] = c
        mat_c['Year'] = y
        mat_c['Scenario'] = sc
        mat_country = mat_country.append(mat_c.reset_index())
#     val = pd.DataFrame(row).fillna(0).T.dot(cM.fillna(0)[0:len(dfC.iloc[:,3:].T)].values).dot(cE.values)
mat_country = mat_country[['Year', 'Scenario', 'Country', 'tech',
                           'Aluminium', 'Bentonite', 'Carbon Fiber', 'Cast Iron', 'Cement',
                           'Ceramics', 'Concrete', 'Copper', 'Epoxy', 'EVA ', 'Fibre Glass',
                           'Glass', 'Lubricant', 'Non-Ferrous Metal', 'Paint', 'Plastic', 'PVC',
                           'Resin', 'Sand', 'Silicon', 'Steel', 'Stainless Steel']]
mat_country.to_csv(r'outputs/massmat_bytech_bycountry.csv',index=False)


# ## Calculating E,W, K, R for all countries

# In[198]:


# Calculating and transforming country data
techcols = ['tech','Country','Scenario','Year','Aluminium', 'Bentonite', 'Carbon Fiber',
            'Cast Iron', 'Cement','Ceramics', 'Concrete','Copper', 'Epoxy', 'EVA ', 'Fibre Glass',
            'Glass', 'Lubricant', 'Non-Ferrous Metal', 'Paint', 'Plastic', 'PVC','Resin', 'Sand',
            'Silicon', 'Steel', 'Stainless Steel']
for ef, ne in zip([cE, cW, cR, cK],['E', 'W', 'R', 'K'] ):
    globals()['df{}_tech'.format(ne)] = cM.fillna(0).dot(cE.values)
    globals()['df{}_tech_sep'.format(ne)]  = cM.fillna(0).multiply(cE.T.values)
    globals()['df{}c_tech_sep'.format(ne)] = pd.DataFrame(columns = techcols)
#     'df{ne}_{country}_{scenario}_{year}_sep'
    for c in dfC['Country'].unique():
        for sc in dfC[dfC['Country']==c]['Scenario'].unique():
            for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
                df_sep = globals()['df{}_tech_sep'.format(ne)].reindex(index = coln).                                                    multiply(dfC[(dfC['Country']==c)&                                                                 (dfC['Scenario']==sc)&                                                                 (dfC['Year']==y)][coln].T.values) 
                df_sep = df_sep.reset_index()
                df_sep['Country'] = c
                df_sep['Scenario'] = sc
                df_sep['Year'] = y
#                 df_sep used to be called globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))]
                globals()['df{}c_tech_sep'.format(ne)] = globals()['df{}c_tech_sep'.format(ne)].append(df_sep)
    globals()['df{}c_tech_sep'.format(ne)] = globals()['df{}c_tech_sep'.format(ne)][techcols]
    globals()['df{}c_tech_sep'.format(ne)].to_csv(r'outputs/{}_matbytech_bycountry.csv'.format(ne))

#     'df{}_country'.format(ne) 
    globals()['df{}_country'.format(ne)] = dfC[['Year','Scenario','Country']]
    globals()['df{}_country'.format(ne)] ['value'] = 0

#     'dfcountry_tech_{}'.format(ne)'
    globals()['dfcountry_tech_{}'.format(ne)] = dfC.copy()
    for col in globals()['dfcountry_tech_{}'.format(ne)].iloc[:,3:].columns:
        globals()['dfcountry_tech_{}'.format(ne)][col] = 0

#     'df{}_country'.format(ne) 
    for index, row in dfC[coln].iterrows():
        val = pd.DataFrame(row).fillna(0).T.dot(cM.fillna(0)[0:len(dfC.iloc[:,3:].T)].values).dot(cE.values)
        globals()['df{}_country'.format(ne)].loc[index,'value'] = val.values[0]

#     'dfcountry_tech_{}'.format(ne)'
        valt = pd.DataFrame(row).fillna(0).values*(globals()['df{}_tech'.format(ne)])
        for col in valt.T.columns:
            globals()['dfcountry_tech_{}'.format(ne)].loc[index,col] = valt.T.loc[0,col]
    
    globals()['df{}_country'.format(ne)].to_csv(r'outputs/df{}_total_bycountry.csv'.format(ne))
    globals()['dfcountry_tech_{}'.format(ne)].to_csv(r'outputs/df{}_tech_bycountry.csv'.format(ne))
            


# In[108]:


# Socio-economic parameters
tecec_df = pd.read_excel(r'data/technoeconomic_params.xlsx')

lf_df = tecec_df[['Technology','Load factor']].drop([0])

sec_techs = ['Diesel (centralised)', 'Diesel 1 kW system (decentralised)',
           'HFO', 'OCGT', 'CCGT', 'CCGT - CCS', 'Supercritical coal',
           'Coal + CCS', 'Hydro (large scale)', 'Hydro (small scale)',
           'Hydro (med. scale)', 'Biomass', 'Biomass (CHP small)',
           'Biomass CCS', 'Nuclear', 'Geothermal', 'Wind onshore',
           'Wind offshore', 'Solar PV (centr.)', 'Solar PV (decentralised)',
           'Solar PV with battery', 'Solar CSP', 'Solar CSP with storage']
o_techs = ['Diesel', 'Diesel',
           'Oil', 'Gas', 'Gas', 'Gas CCS', 'Coal',
           'Coal CCS', 'Hydro', 'Hydro',
           'Hydro', 'Biomass', 'Biomass',
           'Biomass CCS', 'Nuclear', 'Geothermal', 'Wind (Onshore)',
           'Wind (Offshore)', 'Solar PV', 'Solar PV',
           'Solar PV', 'Solar CSP', 'Solar CSP']
lf_d = pd.DataFrame(list(zip(sec_techs,o_techs)), columns = ['Technology','tech'])

lf_d = lf_d.set_index('Technology')['tech'].to_dict()
lf_df['tech']= lf_df['Technology'].map(lf_d)
lf_df = lf_df[['tech','Technology','Load factor']]
lf_df.loc[lf_df['Load factor']=='Varies','Load factor'] = 0

lf_df['Load factor'] = lf_df['Load factor'].astype('int')

# Completing Wind values with data from the UK
# Source: https://www.renewableuk.com/page/UKWEDExplained
lf_df.loc[lf_df['tech']=='Wind (Onshore)','Load factor'] = 26.62
lf_df.loc[lf_df['tech']=='Wind (Offshore)','Load factor'] = 38.86
lf_df.loc[lf_df['tech']=='Wind (Offshore)','Load factor'] = 58.4 # new build offshore wind (2023/24/25) is 58.4%

# Averaging load factors (improve this when technologies are more specific)
lf_df = (lf_df[['tech','Load factor']].groupby(['tech'], as_index = False).agg('mean'))
lf_df.to_csv(r'outputs/load_factors.csv')


# # Errors

# In[189]:


# country energy projection [kWh]
e_C = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AQ:AW", nrows = 1)
# material per energy technology [g/kWh]
e_M = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "AY:BU", nrows = 10)
# embodied emissions (GHG) per material [gCO2e/g]
e_E = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "BX:BX", nrows = 22)
# water usage per material [l/kg]
e_W = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "BY:BY", nrows = 22)
# recycling rate in current supply per material [%]
e_R = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "BZ:BZ", nrows = 22)
# costs per material [â‚¬/kg]
e_K = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices', 
                skiprows = 2, usecols = "CA:CA", nrows = 22)
e_K = e_K/1000

for ef in [e_C, e_M, e_E, e_W, e_R, e_K]:
    eco = ef.columns.values
    ef.columns = [str(x).replace('.1','') for x in eco]
e_M = e_M.rename(columns={'[g/kWh]':'tech'}).set_index('tech')


# # Errors for all countries

# In[191]:


# Error or all countries and scenarios
e_dfC = pd.read_excel(r'data/EnergyProjection.xlsx', sheet_name = 'All countries (e)', 
                skiprows = 2, usecols = "B:R")
# material per energy technology [g/kWh]
e_cM = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "BM:CI", nrows = 15)
# embodied emissions (GHG) per material [gCO2e/g]
e_cE = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CL", nrows = 22)
# water usage per material [l/kg]
e_cW = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CM", nrows = 22)
# recycling rate in current supply per material [%]
e_cR = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CN", nrows = 22)
# costs per material [â‚¬/kg]
e_cK = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Matrices (country)', 
                skiprows = 2, usecols = "CO", nrows = 22)
e_cK = e_cK/1000

# e_dfC
for ef in [ e_cM, e_cE, e_cW, e_cR, e_cK]:
    eco = ef.columns.values
    ef.columns = [str(x).replace('.1','') for x in eco]
e_cM = e_cM.rename(columns={'[g/kWh]':'tech'}).set_index('tech')


# ## Temba errors

# In[ ]:


egeneration_df = generation_df
egeneration_df['Value'] = [x*0.2 for x in egeneration_df['Value']]
egeneration_piv = pd.pivot_table(egeneration_df, values = 'Value', 
                                      index = ['Year', 'scenario', 'country','parameter'], 
                                      columns = 'tech', aggfunc=np.sum)
egeneration_piv = egeneration_piv.reset_index()

egeneration_piv = egeneration_piv.drop(columns = 'parameter')
egeneration_piv.rename(columns ={'scenario':'Scenario','country':'Country'}, inplace = True)

e_dfC['Coal CCS'] = 0
e_dfC = dfC.append(egeneration_piv[egeneration_piv['Country'].isin(['KE','RW','ZA'])])
e_dfC = dfC[['Year', 'Scenario', 'Country', 'Wind (Onshore)', 'Wind (Offshore)',
       'Solar CSP', 'Solar PV', 'Hydro', 'Oil', 'Gas CCS', 'Gas', 'Nuclear',
       'Geothermal', 'Coal', 'Coal CCS', 'Biomass', 'BECCS', 'Hydrogen']]


# ## Error calculations

# In[27]:


# clean calculations trying .dot
for ef, ne in zip([e_E, e_W, e_R, e_K],['E', 'W', 'R', 'K'] ):
    #seperate errors per material per kWh for different impacts 
    globals()['em_'.format(ne)] = ((ef/globals()['{}'.format(ne)]).pow(2)).T
    globals()['em_'.format(ne)].columns = e_cM.columns
    globals()['em_'.format(ne)] = globals()['em_'.format(ne)].reset_index().drop(columns='index', axis = 1)
    
    t1 = pd.DataFrame([x+y for x in ((e_cM/cM).pow(2)).values for y in globals()['em_'.format(ne)].values]).pow(0.5)
    t1.columns = globals()['{}_tech_sep'.format(ne)].columns
    t1.index = e_cM.index
    
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


# In[155]:



# Ordering dfC columns for the same order as M
coln = list(cM.index)
techs = ['Wind (Onshore)', 'Solar CSP', 'Solar PV', 'Hydro', 'Geothermal',
       'Gas CCS', 'Oil', 'Gas', 'Coal', 'Wind (Offshore)', 'Biomass',
       'Nuclear', 'BECCS', 'Hydrogen', 'Coal CCS']
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
    
    globals()['e_df{0}_countries_sep'.format(ne)] = pd.DataFrame(columns =['Country','Year',
                                                                           'Scenario','tech']+mats )
    globals()['e_df{0}c_tot'.format(ne)] = pd.DataFrame(columns =['Country','Year','Scenario']+techs )
    for c in dfC['Country'].unique():
        for sc in dfC[dfC['Country']==c]['Scenario'].unique():
            for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
#                 print(c, sc, y)
                # add index of tech to t2, otherwise the calc gives error
                tc = (e_dfC[(e_dfC['Country']==c)&                            (e_dfC['Scenario']==sc)&                            (e_dfC['Year']==y)][coln]/dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)&                                                          (dfC['Year']==y)])[coln].T
                t2 = pd.DataFrame([x+y+z 
                                   for x 
                                   in ((e_cM/cM).pow(2)).values 
                                   for y 
                                   in globals()['em_'.format(ne)].values for z in (tc.pow(2))]).pow(0.5)
                t2.columns =  globals()['e_{}_tech_sep'.format(ne)].columns
                t2.index = tc.index
                
#                 news = pd.DataFrame(globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].mul(t2, fill_value=0))
                news = globals()['df{}c_tech_sep'.format(ne)]
                news = news[(news['Country']==c)&(news['Scenario']==sc)&(news['Year']==y)]
                news = news.drop(columns=['Country','Scenario','Year']).set_index('tech').mul(t2, fill_value=0)
                news = news.reset_index()
                news['Country'] = c
                news['Scenario'] = sc
                news['Year'] = y
                globals()['e_df{0}_countries_sep'.format(ne)] = globals()['e_df{0}_countries_sep'.format(ne)].append(news)
                globals()['e_df{0}_countries_sep'.format(ne)] = globals()['e_df{0}_countries_sep'.format(ne)][['Country',
                                                                                                               'Year',
                                                                                                               'Scenario',
                                                                                                               'tech']+mats]
                globals()['e_df{0}_countries_sep'.format(ne)].to_csv(r'outputs/errors{}_bymat_bycountry.csv'.format(ne))
                
#                 new = pd.DataFrame(globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].mul(t2, fill_value=0))
                new = globals()['df{}c_tech_sep'.format(ne)]
                new = new[(new['Country']==c)&(news['Scenario']==sc)&(news['Year']==y)]
                new = new.drop(columns=['Country','Scenario','Year']).set_index('tech').mul(t2, fill_value=0)
                new = new.T.sum().reset_index().set_index('tech').T
                new['Country'] = c
                new['Scenario'] = sc
                new['Year'] = y
                globals()['e_df{0}c_tot'.format(ne)] = globals()['e_df{0}c_tot'.format(ne)].append(new)
                globals()['e_df{0}c_tot'.format(ne)] =globals()['e_df{0}c_tot'.format(ne)][['Country',
                                                                                            'Year',
                                                                                            'Scenario']+techs]
                globals()['e_df{0}c_tot'.format(ne)].to_csv(r'outputs/errors{}_total_bycountry.csv'.format(ne))
                
                
                


# # Employment and land-use

# In[113]:


# Employment by stage [job-years/MW] Manuf and C&I and Dec, [jobs/MW] for O&M, [jobs/PJ] for fuel
jobs = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Employment', 
                skiprows = 37, usecols = "B:G,T", nrows = 23)
jobs.columns = ['tech', 'Manufacturing', 'Construction and Installation',
       'Operation and Maintenance', 'Fuel', 'Decommissioning', 'Total']

# In [jobs-lifetime/kWh]
op_jobs = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Employment', 
                skiprows = 65, usecols = "B:G", nrows = 23)
op_jobs.columns = ['tech','Capacity (MW)', 'kWh/lifetime', 'Load factor','Lifetime','Operation and Maintenance']

jobs = jobs.merge(op_jobs[['tech','Capacity (MW)','kWh/lifetime']], on = 'tech')
# Ram2020 has Regional Employment multipliers that should be useful for country evaluations

# Land required [km^2/1000 MW]
land = pd.read_excel(r'data/Excel_model_material_implications_energy_systems.xlsx', 
                    sheet_name = 'Employment', 
                skiprows = 19, usecols = "B:C", nrows = 15)
land.columns = ['tech','land']

jobs['Operation and Maintenance (jobs)'] = jobs['Capacity (MW)'] * jobs['Operation and Maintenance']


# In[114]:


# employment * kWh/tech/country
# jobs


# In[118]:


# Calculation of employment based on new capacity
# cap = ['Power Generation Capacity (Aggregate)','New power generation capacity (Aggregate)']
# employment_df = results_df[results_df['parameter'].isin(cap)]

# power = ['Power Generation Capacity (Aggregate)','New power generation capacity (Aggregate)']
# upow = ['GW','GW']


# # Plots for single country values

# ## Per kWh

# In[616]:


#  per KWH
colors = px.colors.qualitative.Alphabet

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

    pio.write_image(fig, r'figures/kWh_{}.pdf'.format(ne), width = 500, height = 450)
    pio.write_image(fig, r'figures/kWh_{}.eps'.format(ne), width = 500, height = 450)
#     plotly.offline.plot(fig, filename = r'figures/kWh_{}.html'.format(ne), auto_open=False)
#     fig.show()


# ## For Country, all factors

# In[559]:


# for Country
colors = px.colors.qualitative.Alphabet

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

    pio.write_image(fig, r'figures/country_{}.pdf'.format(ne), width = 900, height = 550)
    pio.write_image(fig, r'figures/country_{}.eps'.format(ne), width = 900, height = 550)
#     plotly.offline.plot(fig, filename = r'figures/country_{}.html'.format(ne), auto_open=False)
#     fig.show()


# ## Totals and percentages in country

# In[615]:


#  Bar with totals and total percentages for Country
colors = px.colors.qualitative.Alphabet

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

    pio.write_image(fig, r'figures/country_{}.pdf'.format(n), width = 800, height = 500)
    pio.write_image(fig, r'figures/country_{}.eps'.format(n), width = 800, height = 500)
#     plotly.offline.plot(fig, filename = r'figures/country_{}.html'.format(n), auto_open=False)
#     fig.show()


# # Plots comparing countries/scenarios

# ## Per kWh

# In[ ]:


#  per KWH
colors = px.colors.qualitative.Alphabet

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

    pio.write_image(fig, r'figures/kWh_{}.pdf'.format(ne), width = 500, height = 450)
    pio.write_image(fig, r'figures/kWh_{}.eps'.format(ne), width = 500, height = 450)
#     plotly.offline.plot(fig, filename = r'figures/kWh_{}.html'.format(ne), auto_open=False)
#     fig.show()


# ## For Country, all factors

# In[202]:


# for Country
# make subplots for all scenario/year combos for each country
# MISSING: ADD ERROR DATA
# for ef, ne in zip([cE, cW, cR, cK],['E', 'W', 'R', 'K'] ):
    
colors = px.colors.qualitative.Alphabet

labels = ['Embodied Material Emissions [MtCO<sub>2</sub>]','Embodied Material Water Usage [L]',
          'Material Costs [million Euro]','Material Recycling Rate [t]']
atechs = ['Wind (Onshore)', 'Solar CSP', 'Solar PV', 'Hydro', 'Geothermal',
       'Gas CCS', 'Oil', 'Gas', 'Coal', 'Wind (Offshore)', 'Biomass',
       'Nuclear', 'BECCS', 'Hydrogen','Coal CCS']
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
        h = len(globals()['l_{}'.format(c)])*330
        slegend = [True, False, False, False, False]
        for sc, ns,sl in zip(dfC[dfC['Country']==c]['Scenario'].unique(),range(1,len(dfC[dfC['Country']==c]['Scenario'].unique())+1),slegend):
            print(sc,ns)
            for y in dfC[(dfC['Country']==c)&(dfC['Scenario']==sc)]['Year'].unique():
#                 range(1,len(dfC[dfC['Country']==c]['Scenario'].unique())+1)
#                 globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] 

                    if ne == 'E':
#                         data = globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] /1e12
                        data = globals()['df{}c_tech_sep'.format(ne)]
                        data = data[(data['Country']==c)&(data['Scenario']==sc)&(data['Year']==y)]
                        data=data.drop(columns=['Country','Year','Scenario']).set_index('tech')
                        data = data/1e12
                        datae = pd.DataFrame(np.nan, 
#                                              index=np.arange(len(globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],
#                                                                                                           str(y))])), 
                                             index=np.arange(len(data)), 
#                                      columns=globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))].columns
                                             columns=data.columns
                                            )
                #         ERROR
                        error = globals()['e_df{}c_tot'.format(ne)]
                        error = error[(error['Country']==c)&(error['Scenario']==sc)&(error['Year']==y)][atechs]
                        datae.loc[datae.index,'Stainless Steel'] = error.values/1e12/2
                        datae.index = data.index
                    if ne == 'W':
#                         data = globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] /1000
                        data = globals()['df{}c_tech_sep'.format(ne)]
                        data = data[(data['Country']==c)&(data['Scenario']==sc)&(data['Year']==y)]
                        data=data.drop(columns=['Country','Year','Scenario']).set_index('tech')
                        data = data/1000
                        datae = pd.DataFrame(np.nan, 
                                             index=np.arange(len(data)), 
                                     columns=data.columns)
#                         ERROR
                        error = globals()['e_df{}c_tot'.format(ne)]
                        error = error[(error['Country']==c)&(error['Scenario']==sc)&(error['Year']==y)][atechs]
                        datae.loc[datae.index,'Stainless Steel'] = error.values/1000/2
                        datae.index = data.index

                    if ne == 'K'  or ne =='R':
#                         data = globals()['df{0}_{1}_{2}_{3}_sep'.format(ne,c,sc[0:3],str(y))] /1e6
                        data = globals()['df{}c_tech_sep'.format(ne)]
                        data = data[(data['Country']==c)&(data['Scenario']==sc)&(data['Year']==y)]
                        data=data.drop(columns=['Country','Year','Scenario']).set_index('tech')
                        data = data/1e6
                        datae = pd.DataFrame(np.nan, 
                                             index=np.arange(len(data)), 
                                     columns=data.columns)
                #         ERROR
                        error = globals()['e_df{}c_tot'.format(ne)]
                        error = error[(error['Country']==c)&(error['Scenario']==sc)&(error['Year']==y)][atechs]
                        datae.loc[datae.index,'Stainless Steel'] = error.values/1e6/2
                        datae.index = data.index

                    for i, cl in zip(data.columns, colors): 
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
                                      template = 'simple_white+presentation', barmode='stack')
                    fig.update_layout(legend = dict(font = dict(size = 15)))
                    

        pio.write_image(fig, r'figures/countries/{0}_{1}.pdf'.format(c,ne), width = 900, height = h)
        pio.write_image(fig, r'figures/countries/{0}_{1}.eps'.format(c,ne), width = 900, height = h)
#         plotly.offline.plot(fig, filename = r'figures/countries/{0}_{1}.html'.format(c,ne), auto_open=False)
#         fig.show()


# ## Totals and percentages in country

# In[ ]:


#  Bar with totals and total percentages for Country
colors = px.colors.qualitative.Alphabet

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
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                                    template = 'simple_white+presentation', barmode='stack')


    pio.write_image(fig, r'figures/country_{}.pdf'.format(n), width = 800, height = 500)
    pio.write_image(fig, r'figures/country_{}.eps'.format(n), width = 800, height = 500)
#     plotly.offline.plot(fig, filename = r'figures/country_{}.html'.format(n), auto_open=False)
#     fig.show()


# ## Employment and tech

# In[ ]:





# # Plots for general values

# ## Embodied vs use-phase

# In[1103]:


# Embodied vs use phase

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
datae_u = datae_u[~datae_u['tech'].isin(['Hydrogen','BECCS','Coal CCS'])]
datae_e = datae.append(e_E_tot.reset_index().rename(columns = {0:'error'}))
datae_e = datae_e[~datae_e['tech'].isin(['Hydrogen','BECCS','Coal CCS'])]

# sorting data based on use phase

fig = go.Figure()
for d, dn, c in zip(df, dnames, colors[0:2]): 
    if dn == 'EU': 
        d = EU.sort_values(by = 'AVERAGE', ascending = True)
        d = d[~d['tech'].isin(['Hydrogen','BECCS','Coal CCS'])]
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
        d = d[~d['tech'].isin(['Hydrogen','BECCS','Coal CCS'])]
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

pio.write_image(fig, r'figures/embodied_direct_CO2.pdf', width = 900, height = 700)
pio.write_image(fig, r'figures/embodied_direct_CO2.eps', width = 900, height = 700)
# plotly.offline.plot(fig, filename = r'figures/embodied_direct_CO2.html', auto_open=False)
fig.show()


# ## General - Embodied, water and costs

# In[613]:


# Embodied, water and costs

colors = px.colors.qualitative.Alphabet

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

    pio.write_image(fig, r'figures/general_{}.pdf'.format(ne), width = 1000, height = 700)
    pio.write_image(fig, r'figures/general_{}.eps'.format(ne), width = 1000, height = 700)
#     plotly.offline.plot(fig, filename = r'figures/general_{}.html'.format(ne), auto_open=False)
#     fig.show()


# ## General - Materials per kWh and embodied-material-emissions per kWh

# In[614]:


## Materials per kWh and embodied-material-emissions per kWh
colors = px.colors.qualitative.Alphabet

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
    
    pio.write_image(fig, r'figures/general_{}.pdf'.format(ne), width = 1000, height = 600)
    pio.write_image(fig, r'figures/general_{}.eps'.format(ne), width = 1000, height = 600)
#     plotly.offline.plot(fig, filename = r'figures/general_{}.html'.format(ne), auto_open=False)
#     fig.show()


# ## General - Bars with total percentage of grams for materials and total percentage of CO2 for materials

# In[608]:


## Bar with total percentage of material mass and total CO2 for materials general

colors = px.colors.qualitative.Alphabet

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

    pio.write_image(fig, r'figures/general_{}.pdf'.format(n), width = 800, height = 500)
    pio.write_image(fig, r'figures/general_{}.eps'.format(n), width = 800, height = 500)
#     plotly.offline.plot(fig, filename = r'figures/general_{}.html'.format(n), auto_open=False)
#     fig.show()

