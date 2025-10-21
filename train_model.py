import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


#####Use merged and processed dataset and assign to df
df_new = pd.read_csv('NTSBMerged_processed.csv', low_memory=False)

#####View coloumns and content of df
print(df_new.columns.tolist())
print(df_new.head(10))

print(df_new['acft_make'].unique())
print(df_new['acft_make'].value_counts().head(50))

makes = ['BOEING', 'AIRBUS', 'EMBRAER', 'BOMBARDIER', 'BOMBARDIER INC', 'CESSNA', 'MCDONNELL DOUGLAS', 'DEHAVILLAND',
'DE HAVILLAND', 'PIPER', 'BEECH', 'CIRRUS']


######Slecting new df consisting of the makes specified above
df= df_new[df_new['acft_make'].isin(makes)].copy()

#####Comparison 
print(df_new.shape)
print(df.shape)
print(df.head())
print(df.columns.tolist())

######Overall Anomaly detection with 5 key features 
def anomaly_detection(df):
    Narratives = ['narr_accp', 'narr_accf', 'narr_cause', 'narr_inc']
     
    ####Ensure that dtype for narrative columns is string 
    for col in Narratives:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    ########different plane ranges 
    com_large = ['BOEING', 'AIRBUS', 'MCDONNELL DOUGLAS']
    com_small = ['EMBRAER', 'BOMBARDIER', 'BOMBARDIER INC']
    private_plane = ['BEECH', 'CIRRUS', 'DEHAVILLAND', 'DE HAVILLAND']
    small_plane = ['PIPER', 'CESSNA']

    df['speed_anomaly'] = 0
    df['speed_risk_score'] = 0

    ####Ensure columns with numbers are numeric 
    numeric = ['wind_vel_kts', 'cert_max_gr_wt', 'knots']
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    #####Ensure that the column is only string and that if empty/null fill with unknown otherwise estimate
    if 'ev_nr_apt_loc' in df.columns:
        phase_col = 'ev_nr_apt_loc'
        df[phase_col] = df[phase_col].fillna('UNKNOWN')
        df[phase_col] = df[phase_col].astype(str).str.upper().str.strip()
    else:
        df['inferred_phase'] = 'UNKNOWN'
        df.loc[df['knots'] < 150, 'inferred_phase'] = 'ONAP'
        df.loc[df['knots'] >= 150, 'inferred_phase'] = 'OFAP'
        phase_col = 'inferred_phase'

    ####Cross wind, make and weight based speed detection 
    if 'wind_vel_kts' in df.columns:
        df.loc[df['acft_make'].isin(com_large) & (df['cert_max_gr_wt'] > 110000) & (df['wind_vel_kts'] > 30), 'speed_anomaly'] = 1
        df.loc[df['acft_make'].isin(com_small) & (df['cert_max_gr_wt'] > 45000) & (df['wind_vel_kts'] > 28), 'speed_anomaly'] = 1
        df.loc[df['acft_make'].isin(private_plane) & (df['cert_max_gr_wt'] > 1300) & (df['wind_vel_kts'] > 27), 'speed_anomaly'] = 1
        df.loc[df['acft_make'].isin(small_plane) & (df['cert_max_gr_wt'] > 7000) & (df['wind_vel_kts'] > 25), 'speed_anomaly'] = 1
    
   #####Stall detection based on narrative, mask returns bool
    stall_words = ['nose', 'nose dive', 'stall', 'stalled', 'stab trim', 'aerodynamic', 'loss of lift']
    for col in Narratives:
        if col in df.columns:
            for keyword in stall_words:
                mask = df[col].str.contains(keyword, case=False, na=False)
                #####Finds all rows where stall is true and sets speed_anomaly col to 1
                df.loc[mask, 'speed_anomaly'] = 1
    ########all masking and loc ref https://youtu.be/WfYub6ILhfQ?si=ZOXj0HA_76Ij-Eot; https://youtu.be/naRQyRZrXCE?si=0Cmhzz04NoWNvTgE

    #####Speed thresholds based on make and weight etc
    df['knots'] = 120
    if 'acft_make' in df.columns and 'knots' in df.columns:
        on_apt_mask = df[phase_col].isin(['ONAP', 'ON'])
        
        #######Expecting speed on ground to be below this
        df.loc[on_apt_mask & (df['knots'] > 100), 'speed_anomaly'] = 1
        
        #####Speed on ground for aircraft makes range
        df.loc[on_apt_mask & df['acft_make'].isin(com_large) & (df['knots'] > 80), 'speed_anomaly'] = 1
        df.loc[on_apt_mask & df['acft_make'].isin(com_small) & (df['knots'] > 70), 'speed_anomaly'] = 1
        df.loc[on_apt_mask & df['acft_make'].isin(private_plane) & (df['knots'] > 60), 'speed_anomaly'] = 1
        df.loc[on_apt_mask & df['acft_make'].isin(small_plane) & (df['knots'] > 50), 'speed_anomaly'] = 1
        
        off_apt_mask = df[phase_col].isin(['OFAP', 'OFF', 'OF'])
        
        #######The actual airspeed thresholds for planes in flight  
        boeing_mask = df['acft_make'].str.contains('BOEING', case=False, na=False)
        df.loc[off_apt_mask & boeing_mask & (df['knots'] < 350), 'speed_anomaly'] = 1
        df.loc[off_apt_mask & boeing_mask & (df['knots'] > 580), 'speed_anomaly'] = 1
        
        airbus_mask = df['acft_make'].str.contains('AIRBUS', case=False, na=False)
        df.loc[off_apt_mask & airbus_mask & (df['knots'] < 340), 'speed_anomaly'] = 1
        df.loc[off_apt_mask & airbus_mask & (df['knots'] > 570), 'speed_anomaly'] = 1
        
        med_com_mask = df['acft_make'].isin(['EMBRAER', 'BOMBARDIER', 'BOMBARDIER INC', 'MCDONNELL DOUGLAS'])
        df.loc[off_apt_mask & med_com_mask & (df['knots'] < 300), 'speed_anomaly'] = 1
        df.loc[off_apt_mask & med_com_mask & (df['knots'] > 480), 'speed_anomaly'] = 1
        
        small_com_mask = df['acft_make'].isin(['CESSNA', 'PIPER', 'BEECH', 'CIRRUS'])
        df.loc[off_apt_mask & small_com_mask & (df['knots'] < 80), 'speed_anomaly'] = 1
        df.loc[off_apt_mask & small_com_mask & (df['knots'] > 250), 'speed_anomaly'] = 1
        
        small_mask = df['acft_make'].isin(['DEHAVILLAND', 'DE HAVILLAND'])
        df.loc[off_apt_mask & small_mask & (df['knots'] < 60), 'speed_anomaly'] = 1
        df.loc[off_apt_mask & small_mask & (df['knots'] > 200), 'speed_anomaly'] = 1
        
        ####Risk scores
        ######Traverse through the rows and keep index 
        for index, row in df.iterrows():
            knots = row.get('knots', 0)
            acft_make = str(row.get('acft_make', '')).upper()
            phase = row.get(phase_col, '')
            # phase = row.get(phase_col, 'UNKNOWN')
            

            ####Skip empty speed/make
            if pd.isna(knots) or not acft_make:
                df.at[index, 'speed_risk_score'] = 0
                continue
            #####Tresholds for on airport    
            if phase in ['ONAP', 'ON AIRPORT', 'ON']:
                if knots > 100: 
                    df.at[index, 'speed_risk_score'] = 100
                elif knots > 80: 
                    df.at[index, 'speed_risk_score'] = 75
                elif knots > 60: 
                    df.at[index, 'speed_risk_score'] = 50
                elif knots > 40: 
                    df.at[index, 'speed_risk_score'] = 25
                else: 
                    df.at[index, 'speed_risk_score'] = 0
                    
            else:  
                ########Trshold for cruising 
                if 'BOEING' in acft_make or 'AIRBUS' in acft_make:
                    if knots < 350 or knots > 580: 
                        df.at[index, 'speed_risk_score'] = 100
                    elif knots < 400 or knots > 550: 
                        df.at[index, 'speed_risk_score'] = 50
                    else: 
                        df.at[index, 'speed_risk_score'] = 0
                elif any(x in acft_make for x in ['EMBRAER', 'BOMBARDIER', 'MCDONNELL']):
                    if knots < 300 or knots > 480: 
                        df.at[index, 'speed_risk_score'] = 100
                    elif knots < 350 or knots > 430: 
                        df.at[index, 'speed_risk_score'] = 50
                    else: 
                        df.at[index, 'speed_risk_score'] = 0
                else:
                    if knots < 80 or knots > 250: 
                        df.at[index, 'speed_risk_score'] = 100
                    elif knots < 100 or knots > 200: 
                        df.at[index, 'speed_risk_score'] = 50
                    else: 
                        df.at[index, 'speed_risk_score'] = 0
    else:
        ####airplanes not included
        df.loc[df['knots'] < 50, 'speed_anomaly'] = 1
        df.loc[df['knots'] > 300, 'speed_anomaly'] = 1
    df.loc[df['speed_risk_score'] >= 50, 'speed_anomaly'] = 1
   
    #######2.Maintenance Feature
    df['maintenance_anomaly'] = 0
    
    maintenance_words = ['maintenance', 'mechanical', 'short circuit', 'wear', 'tear', 'fatigue', 'failure', 'malfunction', 'fault']
    for col in Narratives:
        if col in df.columns:
            for keyword in maintenance_words:
                mask = df[col].str.contains(keyword, case=False, na=False)
                df.loc[mask, 'maintenance_anomaly'] = 1
    
    if 'afm_hrs_since' in df.columns:
        df['afm_hrs_since'] = pd.to_numeric(df['afm_hrs_since'], errors='coerce')
        df.loc[df['afm_hrs_since'] > 500, 'maintenance_anomaly'] = 1
    
    ######3.Weather feature
    df['weather_anomaly'] = 0
    
    if 'wind_vel_kts' in df.columns:
        df.loc[df['wind_vel_kts'] > 25, 'weather_anomaly'] = 1
    
    if 'vis_sm' in df.columns:
        df['vis_sm'] = pd.to_numeric(df['vis_sm'], errors='coerce')
        df.loc[df['vis_sm'] < 2, 'weather_anomaly'] = 1
    
    wweather_words = ['weather', 'bad visibility', 'visibility', 'blizzard', 'storm','lightning', 'turbulence', 'fog', 'ice', 'thunder']
    for col in Narratives:
        if col in df.columns:
            for keyword in wweather_words:
                mask = df[col].str.contains(keyword, case=False, na=False)
                df.loc[mask, 'weather_anomaly'] = 1
    
    #####4.Experience Feature
    df['experience_anomaly'] = 0
    
    if 'flight_hours' in df.columns:
        df['flight_hours'] = pd.to_numeric(df['flight_hours'], errors='coerce')
        df.loc[df['flight_hours'] < 100, 'experience_anomaly'] = 1
    
    exp_words = ['inexperienced','learning','failed test', 'student', 'first time', 'no experience', 'low time', 'new pilot']
    for col in Narratives:
        if col in df.columns:
            for keyword in exp_words:
                mask = df[col].str.contains(keyword, case=False, na=False)
                df.loc[mask, 'experience_anomaly'] = 1


#####5. Security Feature (including cybersecurity)
    df['security_anomaly'] = 0
    df['security_risk_score'] = 0


    df['acft_fire'] = df['acft_fire'].astype(str)
    df['acft_expl'] = df['acft_expl'].astype(str)

    ####Physical security anomalies
    if all(col in df.columns for col in ['acft_fire', 'acft_expl']):
        fire_mask = df['acft_fire'].str.contains('Y|TRUE|YES|1', case=False, na=False)
        explosion_mask = df['acft_expl'].str.contains('Y|TRUE|YES|1', case=False, na=False)
        df.loc[fire_mask, 'security_risk_score'] += 30
        df.loc[explosion_mask, 'security_risk_score'] += 30

    df['acars_sys'] = 'Normal'
    df['cpdlc_sys'] = 'Normal'

    #####Flight plan anomaly
    if 'flt_plan_filed' in df.columns:
        df['flt_plan_filed'] = df['flt_plan_filed'].astype(str)
        no_plan_mask = df['flt_plan_filed'].str.contains('N|FALSE|NO|0', case=False, na=False)
        df.loc[no_plan_mask, 'security_risk_score'] += 20

    #####narratives for security issues 
    security_keywords = ['unauthorized', 'illegal', 'violation', 'hijack', 'detour', 'attack', 'security']
    for col in Narratives:
        if col in df.columns:
            for keyword in security_keywords:
                mask = df[col].str.contains(keyword, case=False, na=False)
                df.loc[mask, 'security_risk_score'] += 20

    ########More than 3 missing col ind security issue
    important_cols = ['latitude', 'longitude', 'ev_time', 'acft_make']
    missing_data_no = df[important_cols].isnull().sum(axis=1)
    df.loc[missing_data_no >= 3, 'security_risk_score'] += 10

    ######Combined risk score used to set overall sec score
    # df['security_anomaly'] = (df['security_risk_score'] >= 25).astype(int)
    df.loc[df['security_risk_score'] >= 25, 'security_anomaly'] = 1

    return df
######Used limitations to gauage in general tresholds for each make not specific to model
###############http://www.b737.org.uk/limitations.htm;https://www.theairlinepilots.com/forumarchive/a320/a320-limitations.pdf

####Anomaly features being created
df = anomaly_detection(df)

##########Based on aomaly 0 or 1
df['overall_anomaly'] = (
    df['speed_anomaly'].astype(int) | 
    df['maintenance_anomaly'].astype(int) | 
    df['weather_anomaly'].astype(int) | 
    df['experience_anomaly'].astype(int) | 
    df['security_anomaly'].astype(int)
).astype(int)

#####Balancing classes, select features to focus on
key_features = [
    'wind_vel_kts', 'vis_sm', 'flight_hours', 'afm_hrs_since', 'ev_nr_apt_loc',
    'flt_plan_filed', 'acft_make',
    'knots','crew_age', 'cert_max_gr_wt', 'num_eng', 
    'acft_expl', 'acft_fire',
    'acars_sys', 'cpdlc_sys'
]

######Selecting features that are there in df 
available_features = [f for f in key_features if f in df.columns]
print("Available features:", available_features)
print(df[available_features].head())


X = df[available_features].copy()
y = df['overall_anomaly']

print("X shape:", X.shape)


#####Filling in empty/nulls
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        median_value = X[col].median()
        if pd.isna(median_value):
            median_value = 0
        X[col] = X[col].fillna(median_value)
    else:
        mode_value = X[col].mode()
        if len(mode_value) > 0:
            X[col] = X[col].fillna(mode_value[0])
        else:
            X[col] = X[col].fillna('Unknown')
###########Used deepseek last prompt: "How to loop through columns and fill numerics with median
####and categories with mode or 'Unknown"

#########Check for text columns and convert to numbers 
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        unique_vals = X[col].unique()
        if 'Unknown' not in unique_vals:
            X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
############https://www.geeksforgeeks.org/machine-learning/label-encoding-across-multiple-columns-in-scikit-learn/; 

normal = df[df['overall_anomaly'] == 0]
anomaly = df[df['overall_anomaly'] == 1]

print(f"Normal cases: {len(normal)}")
print(f"Anomaly cases: {len(anomaly)}")

########## Trying to normalize sample, too many anomaly cases 
normal_data = normal.sample(n=min(1000, len(normal)), random_state=42)
anomaly_sample = anomaly.sample(n=min(2000, len(anomaly)), random_state=42)
df_balanced = pd.concat([normal_data, anomaly_sample])

X_balanced = df_balanced[available_features].copy()
y_balanced = df_balanced['overall_anomaly']

####Handle missing values 
for col in X_balanced.columns:
    if X_balanced[col].dtype in ['int64', 'float64']:
        median_value = X_balanced[col].median()
        if pd.isna(median_value):
            median_value = 0
        X_balanced[col] = X_balanced[col].fillna(median_value)
    else:
        mode_val = X_balanced[col].mode()
        if len(mode_val) > 0:
            X_balanced[col] = X_balanced[col].fillna(mode_val[0])
        else:
            X_balanced[col] = X_balanced[col].fillna('Unknown')

#####Do encoding to convert text to numbers 
for col in X_balanced.columns:
    if col in label_encoders:
        X_balanced[col] = X_balanced[col].astype(str)
        valid_categories = label_encoders[col].classes_
        unseen_mask = ~X_balanced[col].isin(valid_categories)
        
        if 'Unknown' in valid_categories:
            X_balanced.loc[unseen_mask, col] = 'Unknown'
        else:
            X_balanced.loc[unseen_mask, col] = valid_categories[0]
        X_balanced[col] = label_encoders[col].transform(X_balanced[col])
#################deepseek: last prompt creating aviation anomlay detection system, as training the model I am using random forest 
#################There are mixed data types in cols.I want to use my trained label_encoders to map any new categories to 'Unknown' if that category exists, otherwise to the first valid category.

print("Balanced X shape:", X_balanced.shape)
print("Y balanced value counts:", y_balanced.value_counts())

######Select 30% to test and 70% to train 
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

########Model selection Random forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=6,
    min_samples_leaf=6,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

#####Model evaluation 
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

######Feature importance 
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

####Testing on entire dataset 
X_full = df[available_features].copy()

#####Ensure numeric columns are filled or median and text columns are filled or unknown 
for col in X_full.columns:
    if X_full[col].dtype in ['int64', 'float64']:
        median_val = X_full[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_full[col] = X_full[col].fillna(median_val)
    else:
        mode_val = X_full[col].mode()
        if len(mode_val) > 0:
            X_full[col] = X_full[col].fillna(mode_val[0])
        else:
            X_full[col] = X_full[col].fillna('Unknown')

for col in X_full.columns:
    if col in label_encoders:
        X_full[col] = label_encoders[col].transform(X_full[col].astype(str))

####Prediction on full dataset 
full_predictions = model.predict(X_full)
full_probabilities = model.predict_proba(X_full)[:, 1]

print(f"Anomalies detected: {full_predictions.sum()} out of {len(full_predictions)}")
print(f"Average anomaly probability: {full_probabilities.mean():.3f}")

####Save models
joblib.dump(model, 'anomaly_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(available_features, 'feature_names.pkl')

####Predictions and probabilities added to df
df['predicted_anomaly'] = full_predictions
df['anomaly_probability'] = full_probabilities

####Save dataset that has the predictions 
df.to_csv('NTSBMerged_processed_with_predictions.csv', index=False)
