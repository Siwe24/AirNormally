import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow
# from tensorflow.keras.models import Sequential, save_model
# from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


############################Aircrafts
######Load and clean aircrafts dataset 
df_NTSBAircrafts = pd.read_csv('NTSBaircraft.csv', delimiter=';', encoding='latin-1',on_bad_lines='skip',low_memory=False)

print(df_NTSBAircrafts.head())

#####Select columns required 
columns_kept = ['ev_id', 'Aircraft_Key', 'acft_make', 'acft_model', 'acft_missing', 'acft_year',
    'acft_category', 'cert_max_gr_wt', 'num_eng', 'fuel_on_board', 'type_last_insp',
    'date_last_insp', 'afm_hrs', 'afm_hrs_last_insp', 'afm_hrs_since', 'certs_held',
    'oper_code', 'second_pilot', 'oper_sched', 'dprt_apt_id', 'dest_apt_id',
    'dprt_time', 'dprt_timezn', 'phase_flt_spec', 'flt_plan_filed',
    'flight_plan_activated', 'oper_dom_int', 'oper_pax_cargo', 'damage',
    'acft_fire', 'acft_expl', 'far_part'
]


dfaircrafts_processed = df_NTSBAircrafts[columns_kept].copy()

# df_filtered.to_csv('filtered_events.csv', index=False)

print(dfaircrafts_processed.shape)
print(dfaircrafts_processed.head())
print(dfaircrafts_processed.columns.tolist())

###Select numeric columns float/int, where it is null/empty just fill with median
numeric_columns = dfaircrafts_processed.select_dtypes(include=['int64', 'float64']).columns
numeric_columns_median = dfaircrafts_processed[numeric_columns].median()
dfaircrafts_processed[numeric_columns] = dfaircrafts_processed[numeric_columns].fillna(numeric_columns_median)


######Select text columns object and fill num/empty cols with unknown
text_columns = dfaircrafts_processed.select_dtypes(include=['object']).columns
dfaircrafts_processed[text_columns] = dfaircrafts_processed[text_columns].fillna('Unknown')

#####go through eaxh column remove white spaces and capitalize
for col in text_columns:
    dfaircrafts_processed[col] = dfaircrafts_processed[col].str.strip().str.upper()

#####Drop duplicate rows/columns 
dfaircrafts_processed = dfaircrafts_processed.drop_duplicates()

####Save to csv 
dfaircrafts_processed.to_csv('NTSBAircrafts_processed.csv', index=False)
print(dfaircrafts_processed[['ev_id', 'Aircraft_Key']].dtypes)


########################Events

####Load NTSB dataset
df_NTSBEvents = pd.read_csv('NTSBevents.csv', delimiter=';', encoding='latin-1',on_bad_lines='skip',low_memory=False)
print(df_NTSBEvents.head())

#######Select columns to keep 
columns_kept1 = [ 'ev_id','ev_type','ev_year', 'ev_time','ev_highest_injury',
    'ev_nr_apt_loc','latitude','longitude','mid_air','on_ground_collision',
    'light_cond','wx_cond_basic','sky_cond_ceil', 'sky_ceil_ht','vis_sm', 'wx_temp',
    'wx_dew_pt','wind_dir_deg','wind_vel_kts','gust_kts', 'wx_int_precip', 'altimeter'
]

####Copy the dataset only with the kept columns 
dfevents_processed = df_NTSBEvents[columns_kept1].copy()


print(dfevents_processed.shape)
print(dfevents_processed.head())
print(dfevents_processed.columns.tolist())


####Selecting columns expected to be numeric 
numeric_columns = ['ev_year', 'latitude', 'longitude', 'sky_ceil_ht', 'vis_sm', 'wx_temp', 
                'wx_dew_pt', 'wind_dir_deg', 'wind_vel_kts', 'gust_kts', 'altimeter']


####Make sure if the column is not int/flot convert it to numeric 
for col in numeric_columns:
    if col in dfevents_processed.columns:
        dfevents_processed[col] = pd.to_numeric(dfevents_processed[col], errors='coerce')

####set median for ampty/null columns
for col in numeric_columns:
    if col in dfevents_processed.columns:
        numeric_columns_median = dfevents_processed[col].median()
        dfevents_processed[col] = dfevents_processed[col].fillna( numeric_columns_median)

######Selecting text columns 
text_columns = ['ev_type', 'ev_highest_injury', 'ev_nr_apt_loc', 'mid_air', 'on_ground_collision',
             'light_cond', 'wx_cond_basic', 'sky_cond_ceil', 'wx_int_precip']
####Empty/null columns set to unknown 
for col in text_columns:
    if col in dfevents_processed.columns:
        dfevents_processed[col] = dfevents_processed[col].fillna('Unknown')

#####if event time is null/empty set default to midnight
if 'ev_time' in dfevents_processed.columns:
    dfevents_processed['ev_time'] = dfevents_processed['ev_time'].fillna('00:00')

####Remove white space and capitilize the text columns
for col in text_columns:
    if col in dfevents_processed.columns:
        dfevents_processed[col] = dfevents_processed[col].str.strip().str.upper()

###drop any duplicates 
dfevents_processed = dfevents_processed.drop_duplicates()


####Save to csv
dfevents_processed.to_csv('NTSBEvents_processed.csv', index=False)


###############Flight Crew

####Load dataset 
df_NTSBFlightCrew = pd.read_csv('NTSBFlight_Crew.csv', delimiter=';', encoding='latin-1',on_bad_lines='skip',low_memory=False)
print(df_NTSBFlightCrew.head())


####Select columns to keep 
columns_kept2 = ['ev_id','Aircraft_Key', 'crew_no', 'crew_category', 'crew_age', 'med_certf','med_crtf_vldty', 'date_lst_med',
    'crew_rat_endorse', 'crew_tox_perf', 'pc_profession', 'bfr', 'bfr_date', 'ft_as_of','pilot_flying'
]

dfflightcrew_processed = df_NTSBFlightCrew[columns_kept2].copy()

print(dfflightcrew_processed.shape)
print(dfflightcrew_processed.head())
print(dfflightcrew_processed.columns.tolist())


###Select numeric columns
numeric_columns = ['crew_age', 'med_crtf_vldty', 'ft_as_of']

###If columns is supposed to be numeric but not, force to numeric field
for col in numeric_columns:
    if col in dfflightcrew_processed.columns:
        dfflightcrew_processed[col] = pd.to_numeric(dfflightcrew_processed[col], errors='coerce')

#######Check if no missing values and set the median else use 0 as median if missing values
for col in numeric_columns:
    if col in dfflightcrew_processed.columns:
        if dfflightcrew_processed[col].notna().any():
            median_val = dfflightcrew_processed[col].median()
        else: 
            median_val = 0
        dfflightcrew_processed[col] = dfflightcrew_processed[col].fillna(median_val)

####Select text columns; any null/empty value replaced with Unknown 
text_columns = ['crew_category', 'med_certf', 'crew_rat_endorse', 'crew_tox_perf', 
             'pc_profession', 'bfr', 'pilot_flying']
for col in text_columns:
    if col in dfflightcrew_processed.columns:
        dfflightcrew_processed[col] = dfflightcrew_processed[col].fillna('Unknown')


####Select date columns and if empty/null set default date
date_columns = ['date_lst_med', 'bfr_date']
for col in date_columns:
    if col in dfflightcrew_processed.columns:
        dfflightcrew_processed[col] = dfflightcrew_processed[col].fillna('2008-01-01')


#####Remove white space and ensure captilization 
for col in text_columns:
    if col in dfflightcrew_processed.columns:
        dfflightcrew_processed[col] = dfflightcrew_processed[col].str.strip().str.upper()

####drop duplicates 
dfflightcrew_processed = dfflightcrew_processed.drop_duplicates()

####Save to CSV
dfflightcrew_processed.to_csv('NTSBFlightCrew_processed.csv', index=False)


#############################Flight Time
df_NTSBFlightTime = pd.read_csv('NTSBFlight_time.csv', delimiter=';', encoding='latin-1',on_bad_lines='skip',low_memory=False)

print(df_NTSBFlightTime.head())

columns_kept3 = ['ev_id','Aircraft_Key', 'crew_no', 'flight_type',
    'flight_craft', 'flight_hours'
]
dfflightTime_processed = df_NTSBFlightTime[columns_kept3].copy()


print(dfflightTime_processed.shape)
print(dfflightTime_processed.head())
print(dfflightTime_processed.columns.tolist())

#####Numeric and text columns
numeric_columns = ['flight_hours']
text_columns = ['flight_type', 'flight_craft']

####Numeric columns expected to have numeric dtype otherwise force 
for col in numeric_columns:
    if col in dfflightTime_processed.columns:
        dfflightTime_processed[col] = pd.to_numeric(dfflightTime_processed[col], errors='coerce')

###Calcuate median if columns not empty ottheriwise fill with 0
for col in numeric_columns:
    if col in dfflightTime_processed.columns:
        if dfflightTime_processed[col].notna().any():
            median_val = dfflightTime_processed[col].median()
        else:
            median_val = 0
        dfflightTime_processed[col] = dfflightTime_processed[col].fillna(median_val)

####If text column is empty or null fill with Unknown
for col in text_columns:
    if col in dfflightTime_processed.columns:
        dfflightTime_processed[col] = dfflightTime_processed[col].fillna('Unknown')

####Remove white space and capitalize 
for col in text_columns:
    if col in dfflightTime_processed.columns:
        dfflightTime_processed[col] = dfflightTime_processed[col].str.strip().str.upper()

####drop duplicates
dfflightTime_processed = dfflightTime_processed.drop_duplicates()

####save to csv 
dfflightTime_processed.to_csv('NTSBFlightTime_processed.csv', index=False)


###############################Narratives 
df_NTSBNarratives = pd.read_csv('NTSB_Narratives.csv', delimiter=';', encoding='latin-1',on_bad_lines='skip',low_memory=False)

print(df_NTSBNarratives.head())

######Select columns to keep 
columns_kept4 = ['ev_id','Aircraft_Key', 'narr_accp',
    'narr_accf', 'narr_cause','narr_inc'
]

dfNTSBNarratives_processed = df_NTSBNarratives[columns_kept4].copy()

print(dfNTSBNarratives_processed.shape)
print(dfNTSBNarratives_processed.head())
print(dfNTSBNarratives_processed.columns.tolist())


#####Text columns 
text_columns = ['narr_accp','narr_accf', 'narr_cause','narr_inc']

###Fill empty/null columns with unknown
for col in text_columns:
    if col in dfNTSBNarratives_processed.columns:
        dfNTSBNarratives_processed[col] = dfNTSBNarratives_processed[col].fillna('Unknown')

####Remove white spaces and capitalize 
for col in text_columns:
    if col in dfNTSBNarratives_processed.columns:
        dfNTSBNarratives_processed[col] = dfNTSBNarratives_processed[col].str.strip().str.upper()

####drop duplicates
dfNTSBNarratives_processed = dfNTSBNarratives_processed.drop_duplicates()

####Save as csv
dfNTSBNarratives_processed.to_csv('NTSBFlightNarratives_processed.csv', index=False)


print(dfNTSBNarratives_processed.head())



#########Merge all datasets into one using event id and aircraft key 
df_aircrafts = pd.read_csv('NTSBAircrafts_processed.csv')
df_aircrafts[['ev_id', 'Aircraft_Key']] = df_aircrafts[['ev_id', 'Aircraft_Key']].astype(str)
df_events = pd.read_csv('NTSBEvents_processed.csv')
df_events[['ev_id']] = df_events[['ev_id']].astype(str)
df_crew = pd.read_csv('NTSBFlightCrew_processed.csv')
df_crew[['ev_id', 'Aircraft_Key']] = df_crew[['ev_id', 'Aircraft_Key']].astype(str)
df_time = pd.read_csv('NTSBFlightTime_processed.csv')
df_time[['ev_id', 'Aircraft_Key']] = df_time[['ev_id', 'Aircraft_Key']].astype(str)
df_narratives = pd.read_csv('NTSBFlightNarratives_processed.csv')
df_narratives[['ev_id', 'Aircraft_Key']] = df_narratives[['ev_id', 'Aircraft_Key']].astype(str)

####left join the datasets on event id and aircraft key 
merged_df = pd.merge(df_aircrafts, df_events, on=['ev_id'], how='left')
merged_df = pd.merge(merged_df, df_crew, on=['ev_id', 'Aircraft_Key'], how='left')
merged_df = pd.merge(merged_df, df_time, on=['ev_id', 'Aircraft_Key'], how='left')
merged_df = pd.merge(merged_df, df_narratives, on=['ev_id', 'Aircraft_Key'], how='left')

####Drop duplicates based on ev id and aircraft key 
merged_df = merged_df.drop_duplicates(subset=['ev_id', 'Aircraft_Key'])

print(merged_df.shape)
print(merged_df.head())
print(merged_df.columns.tolist())
print(len(merged_df))

print(merged_df['ev_type'].value_counts())

####Saved merged dataset in csv
merged_df.to_csv('NTSBMerged_processed.csv', index=False)

####Read from csv 
df= pd.read_csv('NTSBMerged_processed.csv', delimiter=';', encoding='latin-1',on_bad_lines='skip', engine='python')



