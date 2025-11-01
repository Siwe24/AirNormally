import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


######Load and clean aircrafts dataset 
df_aircrafts = pd.read_csv('Aircrafts.csv',on_bad_lines='skip', delimiter=';', encoding='latin-1',low_memory=False)
print(df_aircrafts.head())

#####Select columns required 
columns_kept = ['ev_id', 'Aircraft_Key', 'acft_make', 'acft_model', 'acft_missing', 'acft_year', 'flight_plan_activated',
    'acft_category', 'cert_max_gr_wt', 'num_eng', 'fuel_on_board', 'type_last_insp', 'acft_fire',
    'date_last_insp', 'afm_hrs', 'afm_hrs_last_insp', 'afm_hrs_since', 'certs_held', 'acft_expl',
    'oper_code', 'second_pilot', 'oper_sched', 'dprt_apt_id', 'dest_apt_id', 'far_part', 'damage',
    'dprt_time', 'dprt_timezn', 'phase_flt_spec', 'flt_plan_filed', 'oper_pax_cargo', 'oper_dom_int'
]
dfprocessed_aircrafts = df_aircrafts[columns_kept].copy()

print(dfprocessed_aircrafts.shape)
print(dfprocessed_aircrafts.head())
print(dfprocessed_aircrafts.columns.tolist())

###Select numeric columns float/int, where it is null/empty just fill with median
numeric = dfprocessed_aircrafts.select_dtypes(include=['int64', 'float64']).columns
numeric_median = dfprocessed_aircrafts[numeric].median()
dfprocessed_aircrafts[numeric] = dfprocessed_aircrafts[numeric].fillna(numeric_median)


######Select text columns object and fill num/empty cols with unknown
text = dfprocessed_aircrafts.select_dtypes(include=['object']).columns
dfprocessed_aircrafts[text] = dfprocessed_aircrafts[text].fillna('Unknown')

#####go through each column remove white spaces and capitalize
for col in text:
    dfprocessed_aircrafts[col] = dfprocessed_aircrafts[col].str.strip()
    dfprocessed_aircrafts[col] = dfprocessed_aircrafts[col].str.upper()

#####Drop duplicate rows/columns 
dfprocessed_aircrafts = dfprocessed_aircrafts.drop_duplicates()

####Save to csv 
dfprocessed_aircrafts.to_csv('ProcessedAircrafts.csv', index=False)
print(dfprocessed_aircrafts[['ev_id', 'Aircraft_Key']].dtypes)


########################Events
####Load NTSB dataset
df_events = pd.read_csv('Events.csv', on_bad_lines='skip', delimiter=';', encoding='latin-1',low_memory=False)
print(df_events.head())

#######Select columns to keep 
columns_kept1 = [ 'ev_id','ev_type','ev_year', 'ev_nr_apt_loc', 'latitude', 'ev_time','ev_highest_injury',
    'longitude','mid_air','on_ground_collision','light_cond','wx_cond_basic', 'wx_dew_pt','wind_dir_deg',
    'wind_vel_kts','gust_kts', 'wx_int_precip', 'altimeter', 'sky_cond_ceil', 'sky_ceil_ht','vis_sm', 'wx_temp'
]
####Copy the dataset only with the kept columns 
dfproccessed_events = df_events[columns_kept1].copy()
print(dfproccessed_events.shape)
print(dfproccessed_events.head())
print(dfproccessed_events.columns.tolist())

####Selecting columns expected to be numeric 
numeric = ['ev_year', 'vis_sm', 'wx_temp', 'wind_vel_kts', 'gust_kts','longitude',
                'wx_dew_pt', 'wind_dir_deg', 'altimeter', 'latitude', 'sky_ceil_ht']

######Selecting text columns 
text = ['ev_type', 'ev_highest_injury', 'wx_cond_basic', 'sky_cond_ceil', 'ev_nr_apt_loc',
             'light_cond', 'wx_int_precip', 'mid_air', 'on_ground_collision']
####Empty/null columns set to unknown 
for col in text:
    if col in dfproccessed_events.columns:
        dfproccessed_events[col] = dfproccessed_events[col].fillna('Unknown')

####Make sure if the column is not int/flot convert it to numeric 
for col in numeric:
    if col in dfproccessed_events.columns:
        dfproccessed_events[col] = pd.to_numeric(dfproccessed_events[col], errors='coerce')

####set median for ampty/null columns
for col in numeric:
    if col in dfproccessed_events.columns:
        numeric_median = dfproccessed_events[col].median()
        dfproccessed_events[col] = dfproccessed_events[col].fillna(numeric_median)
#####if event time is null/empty set default to midnight
if 'ev_time' in dfproccessed_events.columns:
    dfproccessed_events['ev_time'] = dfproccessed_events['ev_time'].fillna('00:00')
####Remove white space and capitilize the text columns
for col in text:
    if col in dfproccessed_events.columns:
        dfproccessed_events[col] = dfproccessed_events[col].str.strip()
        dfproccessed_events[col] = dfproccessed_events[col].str.upper()

###drop any duplicates 
dfproccessed_events = dfproccessed_events.drop_duplicates()

####Save to csv
dfproccessed_events.to_csv('ProcessedEvents.csv', index=False)


###############Flight Crew
####Load dataset 
dfflightcrew = pd.read_csv('FlightCrews.csv', on_bad_lines='skip', delimiter=';', encoding='latin-1',low_memory=False)
print(dfflightcrew.head())

####Select columns to keep 
columns_kept2 = ['ev_id','Aircraft_Key', 'crew_no', 'crew_rat_endorse', 'crew_tox_perf', 'pc_profession', 'pilot_flying',
    'bfr', 'bfr_date', 'ft_as_of', 'crew_category', 'crew_age', 'med_certf','med_crtf_vldty', 'date_lst_med'
]
dfprocessed_flightcrew = dfflightcrew[columns_kept2].copy()
print(dfprocessed_flightcrew.shape)
print(dfprocessed_flightcrew.head())
print(dfprocessed_flightcrew.columns.tolist())


####Select text columns; any null/empty value replaced with Unknown 
text = ['crew_category', 'pc_profession', 'bfr', 'crew_tox_perf', 
            'pilot_flying','med_certf', 'crew_rat_endorse']
for col in text:
    if col in dfprocessed_flightcrew.columns:
        dfprocessed_flightcrew[col] = dfprocessed_flightcrew[col].fillna('Unknown')

#####Remove white space and ensure captilization 
for col in text:
    if col in dfprocessed_flightcrew.columns:
        dfprocessed_flightcrew[col] = dfprocessed_flightcrew[col].str.strip()
        dfprocessed_flightcrew[col] = dfprocessed_flightcrew[col].str.upper()

####Select date columns and if empty/null set default date
date = ['date_lst_med', 'bfr_date']
for col in date:
    if col in dfprocessed_flightcrew.columns:
        dfprocessed_flightcrew[col] = dfprocessed_flightcrew[col].fillna('2008-01-01')


###Select numeric columns
numeric = ['crew_age', 'med_crtf_vldty', 'ft_as_of']
###If columns is supposed to be numeric but not, force to numeric field
for col in numeric:
    if col in dfprocessed_flightcrew.columns:
        dfprocessed_flightcrew[col] = pd.to_numeric(dfprocessed_flightcrew[col], errors='coerce')

#######Check if no missing values and set the median else use 0 as median if missing values
for col in numeric:
    if col in dfprocessed_flightcrew.columns:
        if dfprocessed_flightcrew[col].notna().any():
            median_val = dfprocessed_flightcrew[col].median()
        else: 
            median_val = 0
        dfprocessed_flightcrew[col] = dfprocessed_flightcrew[col].fillna(median_val)

####drop duplicates 
dfprocessed_flightcrew = dfprocessed_flightcrew.drop_duplicates()
####Save to CSV
dfprocessed_flightcrew.to_csv('ProcessedFlightCrew.csv', index=False)


#############################Flight Time
dfflighttime = pd.read_csv('FlightTimes.csv', on_bad_lines='skip', delimiter=';', encoding='latin-1',low_memory=False)
print(dfflighttime.head())

columns_kept3 = ['ev_id','Aircraft_Key', 'flight_craft',
    'crew_no', 'flight_type', 'flight_hours'
]
dfprocessed_flighttime = dfflighttime[columns_kept3].copy()

print(dfprocessed_flighttime.shape)
print(dfprocessed_flighttime.head())
print(dfprocessed_flighttime.columns.tolist())

#####Numeric and text columns
numeric = ['flight_hours']
text = ['flight_type', 'flight_craft']

####If text column is empty or null fill with Unknown
for col in text:
    if col in dfprocessed_flighttime.columns:
        dfprocessed_flighttime[col] = dfprocessed_flighttime[col].fillna('Unknown')

####Remove white space and capitalize 
for col in text:
    if col in dfprocessed_flighttime.columns:
        dfprocessed_flighttime[col] = dfprocessed_flighttime[col].str.strip()
        dfprocessed_flighttime[col] = dfprocessed_flighttime[col].str.upper()

####Numeric columns expected to have numeric dtype otherwise force 
for col in numeric:
    if col in dfprocessed_flighttime.columns:
        dfprocessed_flighttime[col] = pd.to_numeric(dfprocessed_flighttime[col], errors='coerce')

###Calcuate median if columns not empty ottheriwise fill with 0
for col in numeric:
    if col in dfprocessed_flighttime.columns:
        if dfprocessed_flighttime[col].notna().any():
            median_val = dfprocessed_flighttime[col].median()
        else:
            median_val = 0
        dfprocessed_flighttime[col] = dfprocessed_flighttime[col].fillna(median_val)

####drop duplicates
dfprocessed_flighttime = dfprocessed_flighttime.drop_duplicates()

####save to csv 
dfprocessed_flighttime.to_csv('ProcessedFlightTime.csv', index=False)


###############################Narratives 
df_narratives = pd.read_csv('Narratives.csv', on_bad_lines='skip', delimiter=';', encoding='latin-1',low_memory=False)
print(df_narratives.head())

######Select columns to keep 
columns_kept4 = ['ev_id','Aircraft_Key','narr_accf', 'narr_cause',
     'narr_accp','narr_inc'
]
dfprocessed_narratives = df_narratives[columns_kept4].copy()

print(dfprocessed_narratives.shape)
print(dfprocessed_narratives.head())
print(dfprocessed_narratives.columns.tolist())

#####Text columns 
text = ['narr_cause','narr_accp','narr_accf','narr_inc']

###Fill empty/null columns with unknown
for col in text:
    if col in dfprocessed_narratives.columns:
        dfprocessed_narratives[col] = dfprocessed_narratives[col].fillna('Unknown')
####Remove white spaces and capitalize 
for col in text:
    if col in dfprocessed_narratives.columns:
        dfprocessed_narratives[col] = dfprocessed_narratives[col].str.strip()
        dfprocessed_narratives[col] = dfprocessed_narratives[col].str.upper()

####drop duplicates
dfprocessed_narratives = dfprocessed_narratives.drop_duplicates()
####Save as csv
dfprocessed_narratives.to_csv('ProcessedNarratives.csv', index=False)
print(dfprocessed_narratives.head())


#########Merge all datasets into one using event id and aircraft key 
df_aircrafts = pd.read_csv('ProcessedAircrafts.csv')
df_aircrafts[['ev_id', 'Aircraft_Key']] = df_aircrafts[['ev_id', 'Aircraft_Key']].astype(str)
df_crew = pd.read_csv('ProcessedFlightCrew.csv')
df_crew[['ev_id', 'Aircraft_Key']] = df_crew[['ev_id', 'Aircraft_Key']].astype(str)
df_events = pd.read_csv('ProcessedEvents.csv')
df_events[['ev_id']] = df_events[['ev_id']].astype(str)
df_narratives = pd.read_csv('ProcessedNarratives.csv')
df_narratives[['ev_id', 'Aircraft_Key']] = df_narratives[['ev_id', 'Aircraft_Key']].astype(str)
df_time = pd.read_csv('ProcessedFlightTime.csv')
df_time[['ev_id', 'Aircraft_Key']] = df_time[['ev_id', 'Aircraft_Key']].astype(str)

####left join the datasets on event id and aircraft key 
merged_df = pd.merge(df_aircrafts, df_events, on=['ev_id'], how='left')
merged_df = pd.merge(merged_df, df_crew, on=['ev_id', 'Aircraft_Key'], how='left')
merged_df = pd.merge(merged_df, df_narratives, on=['ev_id', 'Aircraft_Key'], how='left')
merged_df = pd.merge(merged_df, df_time, on=['ev_id', 'Aircraft_Key'], how='left')

####Drop duplicates based on ev id and aircraft key 
merged_df = merged_df.drop_duplicates(subset=['ev_id', 'Aircraft_Key'])

print(merged_df.shape)
print(merged_df.head())
print(merged_df.columns.tolist())
print(len(merged_df))

print(merged_df['ev_type'].value_counts())
####Saved merged dataset in csv
merged_df.to_csv('ProcessedMerged.csv', index=False)




