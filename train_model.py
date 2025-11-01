import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score



#####Use merged and processed dataset and assign to df
dfmerged_new = pd.read_csv('ProcessedMerged.csv', low_memory=False)

#####View coloumns and content of df
print(dfmerged_new.columns.tolist())
print(dfmerged_new.head(20))

print(dfmerged_new['acft_make'].unique())
print(dfmerged_new['acft_make'].value_counts().head(30))

makes = ['AIRBUS','BEECH','BELLANCA','BOEING', 'BOMBARDIER', 'BOMBARDIER INC', 'CESSNA', 'CIRRUS DESIGN CORP',  
'DEHAVILLAND', 'DE HAVILLAND', 'EMBRAER','MCDONNELL DOUGLAS','MOONEY','PIPER']


######Selecting new df consisting of the makes specified above
df= dfmerged_new[dfmerged_new['acft_make'].isin(makes)].copy()

print(dfmerged_new.shape)
print(df.shape)
print(df.head())
print(df.columns.tolist())


class patterndetection:
    def __init__(self):
        self.largecom = ['AIRBUS','BOEING', 'MCDONNELL DOUGLAS']
        self.smallcom = ['BELLANCA', 'BOMBARDIER', 'BOMBARDIER INC', 'EMBRAER']
        self.private = ['BEECH', 'CIRRUS DESIGN CORP', 'DEHAVILLAND', 'DE HAVILLAND']
        self.small = ['PIPER', 'CESSNA','MOONEY']

    def complex_pattern_detection(self, df):
        numeric = ['afm_hrs_since', 'wind_vel_kts', 'wx_temp', 'altimeter', 'flight_hours', 
                       'vis_sm', 'crew_age', 'cert_max_gr_wt', 'knots']
        
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['acars_sys'] = 'Normal'
        df['cpdlc_sys'] = 'Normal'
        ###############Basic thresholds
        df['wind_anomaly'] = (df['wind_vel_kts'] > 25).astype(int)
        df['vis_anomaly'] = (df['vis_sm'] < 3).astype(int)
        df['exp_anomaly'] = (df['flight_hours'] < 500).astype(int)
        df['expl_anomaly'] = (df['acft_expl'] == 'YES')
        df['fire_anomaly'] = (df['acft_fire'] == 'YES')
        df['acars_anomaly'] = (df['acars_sys'] == 'FAILED')
        df['cpdlc_anomaly'] = (df['cpdlc_sys'] == 'FAILED')
        df['maint_anomaly'] = (df['afm_hrs_since'] > 500).astype(int)
        df['plan_anomaly'] = (df['flt_plan_filed'] == 'YES')
        
        
        risklevel = 0

        df['acars_sys'] = 'NORMAL'
        df['cpdlc_sys'] = 'NORMAL'

        if 'acars_sys' in df.columns:
            acars = df['acars_sys'].astype(str).str.contains('FAILED|SLOW', case=False, na=False)
            risklevel += acars.astype(int)
        if 'cpdlc_sys' in df.columns:
            cpdlc = df['cpdlc_sys'].astype(str).str.contains('FAILED|SLOW', case=False, na=False)
            risklevel += cpdlc.astype(int)
        if 'flt_plan_filed' in df.columns:
            no_plan = ~df['flt_plan_filed'].astype(str).str.contains('YES|Y|TRUE', case=False, na=False)
            risklevel += no_plan.astype(int)      
        if 'acft_fire' in df.columns:
            fire = df['acft_fire'].astype(str).str.contains('YES|Y|TRUE', case=False, na=False)
            risklevel += fire.astype(int) 
        if 'acft_expl' in df.columns:
            explosion = df['acft_expl'].astype(str).str.contains('YES|Y|TRUE', case=False, na=False)
            risklevel += explosion.astype(int)
        
        df['wx_cond_basic'] = 'VMC'
        if 'wx_cond_basic' == 'IMC':
            flighthrs_req = 1200
        else:
            flighthrs_req = 500

        df['speed_anomaly'] = self.speed_anomaly(df)
        
        ########flight hours vs low vis
        if 'flight_hours' in df.columns and 'vis_sm' in df.columns:
            df['vis_sm'] = pd.to_numeric(df['vis_sm'], errors='coerce').fillna(10)
            df['exp_visibility'] = ((flighthrs_req/ (df['flight_hours'] + 1)) *  (10 / (df['vis_sm'] + 1)))
        #####Maintenance
        if 'afm_hrs_since' in df.columns:
            df['afm_hrs_since'] = pd.to_numeric(df['afm_hrs_since'], errors='coerce').fillna(0)
            df['main_risk'] = df['afm_hrs_since'] / 500
           ########weather,experience and instrument type flight
        if 'flight_hours' in df.columns and 'wind_vel_kts' in df.columns:
            df['flight_hours'] = pd.to_numeric(df['flight_hours'], errors='coerce').fillna(1000)
            df['wind_vel_kts'] = pd.to_numeric(df['wind_vel_kts'], errors='coerce').fillna(0)
            df['exp_weather'] = ((flighthrs_req / (df['flight_hours'] + 1) * df['wind_vel_kts'] / 25))
        
        df['multirisks'] = risklevel
        narrative = ['narr_accp', 'narr_accf', 'narr_cause', 'narr_inc']
        df['narrative_totalrisks'] = self.narrative_risks(df, narrative)
        ##########Three complex patterns for now
        df['cp1'] = ((df['exp_weather'] > 2.0) & (df['exp_visibility'] > 2.0)).astype(int)
        df['cp2'] = ((df['main_risk'] > 1.0) & (df['flight_hours'] < 500)).astype(int)
        df['cp3'] = ((df['multirisks'] >= 2) & (df['narrative_totalrisks'] >= 2)).astype(int)
        
        return df

    def speed_anomaly(self, df):
        anomaly_speed = []
        ####Risk scores
        ######Traverse through the rows and keep index 
        for index, row in df.iterrows():
            phase = row.get('ev_nr_apt_loc', 'OFAP')
            acft_make = str(row.get('acft_make', '')).upper()
            knots = row.get('knots', 0)

            smallmixprivate = self.private + self.small

            if phase in ['ONAP', 'ON']:
                if any(make in acft_make for make in self.largecom):
                    speedanomaly = knots > 80
                elif any(make in acft_make for make in self.smallcom):
                    speedanomaly = knots > 70
                elif any(make in acft_make for make in self.private):
                    speedanomaly = knots > 60
                elif any(make in acft_make for make in self.small):
                    speedanomaly = knots > 50
                else:
                    speedanomaly = knots > 100
            else:
                if any(make in acft_make for make in self.largecom):
                    speedanomaly = knots < 350 or knots > 580
                elif any(make in acft_make for make in self.smallcom):
                    speedanomaly = knots < 300 or knots > 480
                elif any(make in acft_make for make in smallmixprivate):
                    speedanomaly = knots < 80 or knots > 250
                else:
                    speedanomaly = knots < 100 or knots > 400
            
            anomaly_speed.append(int(speedanomaly))
        return pd.Series(anomaly_speed, index=df.index)

    def narrative_risks(self, df, narrative):
        narratives_combined = ''
        for col in narrative:
            if col in df.columns:
                narratives_combined += ' ' + df[col].fillna('').astype(str)
        risk_keywords = ['inexperienced', 'student', 'bad', 'dive','stall', 'overspeed', 'failure', 'malfunction', 'broken', 
            'emergency', 'divert', 'weather', 'storm', 'turbulence','jam', 'freeze', 'training', 'maintenance', 'thunder',
            'fire', 'explosion', 'communication', 'fail', 'instrument','heavy','lost'
        ]
        totalrisks = pd.Series(0, index=df.index)
        for keyword in risk_keywords:
            totalrisks += narratives_combined.str.contains(keyword, case=False, na=False).astype(int)
        
        return totalrisks


def complex_anomalies(df):
    """Create labels using both thresholds AND complex patterns"""
    df['overall_anomaly'] = 0
    
    ########if two or more of these features have anomalies
    threshold_anomalies = (
        df['wind_anomaly'] + df['expl_anomaly']+ df['vis_anomaly'] + df['fire_anomaly'] + df['exp_anomaly'] + 
        df['maint_anomaly'] + df['acars_anomaly'] + df['speed_anomaly'] + df['cpdlc_anomaly'] + df['plan_anomaly']
    )
    comp1 = threshold_anomalies >= 2
    #####if any of the 3 prev patterns are true
    comp2 = (
        (df.get('cp1', 0) == 1) | (df.get('cp2', 0) == 1) | (df.get('cp3', 0) == 1)
    )
    ######Narratives has more than 3 flags (flagged words) and already 1 threshold anoamly or more found 
    comp3 = (
        (df.get('narrative_totalrisks', 0) >= 3) & (threshold_anomalies >= 1)
    )
    ####more than 3 risks
    comp4 = df.get('multirisks', 0) >= 3
   ######Extremely high, when it is alr ano and way above/below threshold
    comp5 = (
        (df['wind_anomaly'] == 1) & (df['wind_vel_kts'] > 40) | (df['vis_anomaly'] == 1) & (df['vis_sm'] < 1) |
        (df['exp_anomaly'] == 1) & (df['flight_hours'] < 50) | (df['maint_anomaly'] == 1) & (df['afm_hrs_since'] > 1000)
    ) 
    anomaly_mask = comp1 | comp2 | comp3 | comp4 | comp5   
    df.loc[anomaly_mask, 'overall_anomaly'] = 1

    return df

engineered_features = patterndetection()
df_with_all_features = engineered_features.complex_pattern_detection(df)

df_with_labels = complex_anomalies(df_with_all_features.copy())

df = df_with_labels
enginerring_features = [
    'wind_vel_kts', 'vis_sm', 'flight_hours', 'afm_hrs_since', 'ev_nr_apt_loc', 'flt_plan_filed', 'acft_make', 
    'crew_age', 'cert_max_gr_wt', 'num_eng', 'acft_expl', 'acft_fire', 'acars_sys', 'cpdlc_sys', 'knots', 'wx_cond_basic'
]

available_features = [f for f in enginerring_features if f in df_with_all_features.columns]
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
normal_data = normal.sample(n=min(5000, len(normal)), random_state=42)
anomaly_sample = anomaly.sample(n=min(5000, len(anomaly)), random_state=42)
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

print("Balanced X:", X_balanced.shape)

######Select 30% to test and 70% to train 
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


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

print(f"ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

######Feature importance 
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

###########Validation that prediction is as accurate as possible
predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.savefig('predictionacc.png')
plt.show()

###########Featureimportance ranking
plt.figure(figsize=(12, 10))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


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

####Save models
joblib.dump(model, 'anomaly_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(available_features, 'feature_names.pkl')

####Save dataset that has the predictions 
df.to_csv('ProccesssedMergedTrained.csv', index=False)

