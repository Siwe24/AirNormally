import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import StringIO, BytesIO
from flask import Flask, send_file, request, render_template, jsonify, session
import os
import base64
import random
import re
from datetime import datetime
import tempfile
from functools import wraps

app = Flask(__name__)
app.secret_key = 'air2124'

#####Authrorized users and their roles
####Analyst has full access: Operator cant download report but upl file: public can only analyze single flight and view results
USERS = {
    'gen_air': {'password': 'jessica38!@', 'role': 'general'},
    'operator_tom24': {'password': 'tomhollan32/!', 'role': 'operator'}, 
    'jakesmith_analyst20': {'password': 'Analyst@J!0921', 'role': 'analyst'},
    'admin': {'password': 'Administrator1999!y', 'role': 'analyst'}
}
##########Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'success': False, 'error': 'Authentication required. Please login first.'}), 401
        return f(*args, **kwargs)
    return decorated_function

def role_required(required_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return jsonify({'success': False,'error': 'Log in first'}), 401
            user_role = session.get('role')
            if user_role not in required_roles:
                role_names = " or ".join([r.title() for r in required_roles])
                return jsonify({'success': False,'error': f'Access denied. {role_names} role required.'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

######Python decorators for user/session management from https://dokumen.pub/flask-web-development-developing-web-applications-with-python-first-edition-9781449372620-1449372627.html
####Deepseek "How to use decorators in python for user and access management"

#######Model loading 
try:
    model = joblib.load('anomaly_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Models successfully loaded")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None


##############Categorizing narratives into different fields
class traverse_narratives:
    def __init__(self):

        self.weather= ['weather', 'storm','low visibility', 'bad weather', 'turbulence', 'wind', 'fog', 'rain', 'snow', 
        'thunder', 'lightning', 'blizzard', 'crosswind', 'visibility', 'severe weather','ice', 'icing'
        ]

        self.speed = {'stall': ['stall', 'stalled', 'loss of lift'],
            'overspeed': ['overspeed', 'excessive speed', 'too fast', 'high speed'],
            'underspeed': ['slow', 'low speed', 'below speed', 'minimum speed'],
            'uncontrolled': ['uncontrolled', 'loss of control','no control']
        }
        self.maintenance = ['maintenance', 'mechanical', 'failure', 'malfunction', 'fault', 'broken', 'wear', 'tear',
            'fatigue', 'short circuit', 'electrical fault', 'engine failure', 'system failure'
        ]
        
        self.experience = ['inexperienced', 'student', 'training', 'first time', 'new pilot',
            'low time', 'low hours', 'learning', 'failed test'
        ]
        self.security = ['unauthorized', 'illegal', 'hijack', 'jam','security', 'breach', 'attack', 
                            'explosion', 'fire', 'bomb', 'threat', 'emergency', 'mayday'
        ]

    def extract_narratives(self, narrative_text):
        narrative_lower = narrative_text.lower()
        
        highrisk_words = [
            'stall', 'overspeed', 'failure', 'emergency', 'divert', 'explosion', 'fire', 'hijack', 'attack',
                'crash', 'collision', 'mayday', 'engine failure', 'engine out', 'lost engine', 'declaring emergency'
        ]
        
        highrisk_narratives = 0
        for keyword in highrisk_words:
            if keyword in narrative_lower:
                highrisk_narratives = 1
                break
        
        return {
            'highrisk_narratives': highrisk_narratives
        }

    def incl_narrative(self, flight_data, narrative_text):
        incl_data = flight_data.copy()
        
        if narrative_text and narrative_text.strip():
            narrative_features = self.extract_narratives(narrative_text)
            incl_data.update(narrative_features)

            return incl_data

class Airnormally:
    def __init__(self):
        self.feature_extractor = traverse_narratives()
        self.anomaly_types = {
            'speed': 'Speed anomalies such as crosswinds',
            'maintenance': 'Maintenance related issues',
            'weather': 'Weather issues',
            'experience': 'Experience and Training issues',
            'security': 'Security related issues',
            'complex_patterns': 'Complex interaction patterns'
        }


        ######Classifying aircrafts based on make
        self.largecom = ['AIRBUS','BOEING', 'MCDONNELL DOUGLAS']
        self.smallcom = ['BELLANCA', 'BOMBARDIER', 'BOMBARDIER INC', 'EMBRAER']
        self.private = ['BEECH', 'CIRRUS DESIGN CORP', 'DEHAVILLAND', 'DE HAVILLAND']
        self.small = ['CESSNA','MOONEY','PIPER']


 ########Calculate conf score, for anomaly is as is but in percentage format, normal is 1-prob
    def confidence_score(self, probability, anomaly):
        if anomaly:
            return probability * 100
        else:
            return (1.0 - probability) * 100




    def complex_detection(self, flight_data, narrative_text=""):
        incl_data = flight_data.copy()
        
        #####ensure numeric fields are float/numeric
        numeric = ['wind_vel_kts', 'vis_sm', 'flight_hours', 'afm_hrs_since', 'knots', 'crew_age', 'cert_max_gr_wt','wind_vel_kts', 'wx_temp']
        for field in numeric:
            if field in incl_data:
                try:
                    incl_data[field] = float(incl_data[field])
                except:
                    incl_data[field] = 0
        
        wind_vel = incl_data.get('wind_vel_kts', 0)
        vis_sm = incl_data.get('vis_sm', 10)
        flight_hours = incl_data.get('flight_hours', 1000)
        afm_hrs_since = incl_data.get('afm_hrs_since', 0)
        knots = incl_data.get('knots', 200)
        acft_expl = incl_data.get('acft_expl', 'NO')
        acft_fire = incl_data.get('acft_fire', 'NO')
        acars_sys = incl_data.get('acars_sys', 'NORMAL')
        acpdlc_sys = incl_data.get('cpdlc_sys', 'NORMAL')
        flt_plan_filed = incl_data.get('flt_plan_filed', 'YES')
        wx_cond_basic = incl_data.get('wx_cond_basic', 'VMC')
        
        incl_data['critical_wind'] = 1 if wind_vel > 25 else 0
        incl_data['critical_visibility'] = 1 if vis_sm < 3 else 0
        incl_data['critical_experience'] = 1 if flight_hours < 500 else 0
        incl_data['critical_maintenance'] = 1 if afm_hrs_since > 500 else 0
    


        ###################Speed based calculation
        acft_make = str(incl_data.get('acft_make', '')).upper()
        phase = incl_data.get('ev_nr_apt_loc', 'OFAP')
        
        if phase in ['ONAP', 'ON']:
            if any(make in acft_make for make in self.largecom):
                incl_data['critical_speed'] = 1 if knots > 80 else 0
            elif any(make in acft_make for make in self.smallcom):
                incl_data['critical_speed'] = 1 if knots > 70 else 0
            else:
                incl_data['critical_speed'] = 1 if knots > 60 else 0
        else:
            if any(make in acft_make for make in self.largecom):
                incl_data['critical_speed'] = 1 if knots < 350 or knots > 580 else 0
            elif any(make in acft_make for make in self.smallcom):
                incl_data['critical_speed'] = 1 if knots < 300 or knots > 480 else 0
            else:
                incl_data['critical_speed'] = 1 if knots < 80 or knots > 250 else 0
        
 
        ##############Security anoamlies also based on thresholds
        incl_data['communication_failure'] = 0
        acars_status = str(incl_data.get('acars_sys', 'Normal')).upper()
        cpdlc_status = str(incl_data.get('cpdlc_sys', 'Normal')).upper()
        
        if acars_status in ['SLOW', 'FAILED'] or cpdlc_status in ['SLOW', 'FAILED']:
            incl_data['communication_failure'] = 1
            print(f"COMMUNICATION INTERFERENCE DETECTED: ACARS-{acars_status}, CPDLC-{cpdlc_status}")
        
        incl_data['flight_procedurev'] = 0
        if str(flight_data.get('flt_plan_filed', 'YES')).upper() in ['NO', 'N', 'FALSE']:
            incl_data['flight_procedurev'] = 1

        incl_data['security_incident'] = 0
        if str(flight_data.get('acft_fire', 'NO')).upper() in ['YES', 'Y']:
            incl_data['security_incident'] = 1
        if str(flight_data.get('acft_expl', 'NO')).upper() in ['YES', 'Y']:
            incl_data['security_incident'] = 1


        ##############Complex relationships and interactions between the features
        incl_data['weather_exp_interaction'] = 0
        if 'flight_hours' < '1200' and 'wx_cond_basic' == 'IMC':
            incl_data['weather_exp_interaction'] = 1
        if 'flight_hours' < '500' and 'wx_cond_basic' == 'VMC':
            incl_data['weather_exp_interaction'] = 1

        incl_data['vis_exp_interaction'] = 1 if (vis_sm < 3) and (flight_hours < 500) else 0
        incl_data['maint_exp_interaction'] = 1 if (afm_hrs_since > 300) and (flight_hours < 500) else 0


        #######################Risks within narratives in CVR,using keywords
        incl_data['highrisk_narratives'] = 0
        if narrative_text:
            narrative_lower = narrative_text.lower()
            strong_keywords = [
                'stall', 'overspeed', 'failure', 'emergency', 'divert', 'stuck', 'explosion', 
                'fire', 'hijack', 'terrorism', 'attack', 'low visibility', 'crash', 
                'collision', 'mayday', 'engine failure', 'bad', 'jam',
                'engine out', 'lost engine', 'declaring emergency', 'shaker'
            ]
            for keyword in strong_keywords:
                if keyword in narrative_lower:
                    incl_data['highrisk_narratives'] = 1
                    break
        
        ##################Analysis of combined risks
        incl_data['total_critical_anomalies'] = (
            incl_data['critical_wind'] + incl_data['critical_visibility'] + 
            incl_data['critical_experience'] + incl_data['critical_maintenance'] + 
            incl_data['critical_speed'] + incl_data['security_incident'] +
            incl_data['communication_failure'] + incl_data['flight_procedurev']
        )
        
        incl_data['total_interaction_risks'] = (
            incl_data['weather_exp_interaction'] + incl_data['vis_exp_interaction'] + incl_data['maint_exp_interaction']
        )
        return incl_data

    def anomaly_prediction(self, flight_data, narrative_text=""):
        if model is None:
            return False, 0.0, "No Model Found", flight_data, False
        
        try:
            incl_data = self.complex_detection(flight_data, narrative_text)
            is_narrative_enhanced = bool(narrative_text and narrative_text.strip())
            
            total_critical_anomalies = incl_data.get('total_critical_anomalies', 0)
            total_interaction_risks = incl_data.get('total_interaction_risks', 0)
            highrisk_narratives = incl_data.get('highrisk_narratives', 0)

            security_emergency = (
                str(flight_data.get('acft_fire', 'NO')).upper() in ['YES', 'Y'] or
                str(flight_data.get('acft_expl', 'NO')).upper() in ['YES', 'Y']
            )
            

            ######Emergency keywords
            has_emergency_narrative = False
            if narrative_text:
                narrative_lower = narrative_text.lower()
                emergency_keywords = [
                    'engine failure', 'mayday', 'emergency divert', 'declaring emergency',
                    'engine out', 'lost engine', 'emergency landing', 'flare out',
                    'crash', 'explosion', 'fire', 'hijack',  'attack', 'no controls'
                ]
                has_emergency_narrative = any(keyword in narrative_lower for keyword in emergency_keywords)

            df = pd.DataFrame([incl_data])
            
           #######Traverse to find feature set as unkn if in label otherwise leave as null/0
            for feature in feature_names:
                if feature not in df.columns:
                    if feature in label_encoders:
                        df[feature] = 'Unknown'
                    else:
                        df[feature] = 0
            
            ###########Label econdoing reference is in train_model.py
            for feature in feature_names:
                if feature in label_encoders:
                    current = str(df[feature].iloc[0])
                    if current not in label_encoders[feature].classes_:
                        valid = label_encoders[feature].classes_
                        if 'Unknown' in valid:
                            default = 'Unknown'
                        else:
                            default = valid[0] if len(valid) > 0 else 'Unknown'
                        df[feature] = default
                    else:
                        df[feature] = current
                    df[feature] = label_encoders[feature].transform(df[feature].astype(str))
            
            df = df[feature_names].fillna(0)
            ######https://youtu.be/naRQyRZrXCE?si=r5-4qNr_RQr1TDqk

            #####Using model to make predictions and set prob
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
        
            is_critical_emergency = security_emergency or has_emergency_narrative
            

            ########Security and emergency narratives can overide thresholds since they are more dire
            if is_critical_emergency:
                is_anomaly = True
                probability = max(probability, 0.95)
            
            ########Critical has to at least be at least 85%
            elif total_critical_anomalies > 0:
                is_anomaly = True
                probability = max(probability, 0.85)
            #######narrative risks that are high enough should at least be 80%
            elif highrisk_narratives == 1:
                is_anomaly = True
                probability = max(probability, 0.80)
            #####If no violations, use the value as is this is for noamal scenarios
            else:
                is_anomaly = bool(prediction)

            return is_anomaly, float(probability), incl_data, is_narrative_enhanced
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return False, 0.0, str(e), flight_data, False


######Assigning risk level based on the prob only for anomalies
    def risk_level(self, anomaly, probability):
        if not anomaly:
            return "NORMAL"
        elif probability >= 0.8:
            return "HIGH"
        elif probability >= 0.6:
            return "MEDIUM"
        elif probability >= 0.4:
            return "LOW"
        else:
            return "VERY LOW"


#######Analysis on both thresholds and patterns 
    def analyze_anomaly_types(self, flight_data, narrative_text=""):
        analysis = {}
        threshold_violations = {}
        complex_patterns = {}

        combination_data = self.complex_detection(flight_data, narrative_text)
        
        ############Normal Threshold violations datacted with reason
        if combination_data.get('critical_wind', 0) == 1:
            threshold_violations['wind'] = f"High crosswind: {flight_data.get('wind_vel_kts', 0)}kts"
        
        if combination_data.get('critical_visibility', 0) == 1:
            threshold_violations['visibility'] = f"Low visibility: {flight_data.get('vis_sm', 10)} miles"
        
        if combination_data.get('critical_experience', 0) == 1:
            threshold_violations['experience'] = f"Low pilot experience: {flight_data.get('flight_hours', 1000)} hours"
        
        if combination_data.get('critical_maintenance', 0) == 1:
            threshold_violations['maintenance'] = f"Extended maintenance: {flight_data.get('afm_hrs_since', 0)} hours since inspection"
        
        if combination_data.get('critical_speed', 0) == 1:
            threshold_violations['speed'] = f"Speed anomaly: {flight_data.get('knots', 0)}kts for {flight_data.get('acft_make', 'Unknown')}"
        
        if combination_data.get('security_incident', 0) == 1:
            threshold_violations['security'] = "Security incident detected (fire/explosion)"
        
        if combination_data.get('communication_failure', 0) == 1:
            threshold_violations['communication'] = "Communication system failure"
        
        if combination_data.get('flight_procedurev', 0) == 1:
            threshold_violations['procedural'] = "Flight plan not filed"
        
        ################Complex patterns detected with reason
        if combination_data.get('weather_exp_interaction', 0) == 1:
            complex_patterns['weather_experience'] = "High winds with inexperienced pilot"
        
        if combination_data.get('vis_exp_interaction', 0) == 1:
            complex_patterns['visibility_experience'] = "Low visibility with inexperienced pilot"
        
        if combination_data.get('maint_exp_interaction', 0) == 1:
            complex_patterns['maintenance_experience'] = "Overdue maintenance with inexperienced pilot"
        
        if combination_data.get('highrisk_narratives', 0) == 1 and narrative_text:
            complex_patterns['narrative'] = "High risk issues detected from CVR"
        
        total_critical = combination_data.get('total_critical_anomalies', 0)
        if total_critical >= 3:
            complex_patterns['multiple_critical'] = f"Multiple critical anomalies: {total_critical}"
        
        total_interactions = combination_data.get('total_interaction_risks', 0)
        if total_interactions >= 2:
            complex_patterns['multiple_interactions'] = f"Multiple risk relationships: {total_interactions}"
        
        return {
            'threshold_violations': threshold_violations,
            'complex_patterns': complex_patterns,
            'risk_scores': {
                'total_critical_anomalies': total_critical,
                'total_interaction_risks': total_interactions,
                'narrative_risk': combination_data.get('highrisk_narratives', 0)
            }
        }

    def suggest_recommendations(self, anomaly, probability, flight_data, narrative_text=""):
        recommendations = []
        
        if not anomaly:
            recommendations.append("Continue normal operations")
            recommendations.append("Maintain current safety protocols")
            return recommendations
        
        #####High priority rec
        if probability > 0.7:
            recommendations.append("IMMEDIATE ACTION: Conduct emergency safety briefing")
        elif probability > 0.5:
            recommendations.append("Schedule immediate maintenance inspection")
            recommendations.append("Review pilot training and qualifications")

        analysis = self.analyze_anomaly_types(flight_data, narrative_text)

        for violation_type, violation_desc in analysis['threshold_violations'].items():
            if 'wind' in violation_type:
                recommendations.append("Implement crosswind landing procedures")
            elif 'visibility' in violation_type:
                recommendations.append("Review/Implement low visibility procedures")
            elif 'experience' in violation_type:
                recommendations.append("Assign experienced Pilot or First Officer")
            elif 'maintenance' in violation_type:
                recommendations.append("Schedule immediate maintenance inspection")
            elif 'speed' in violation_type:
                recommendations.append("Review speed thresholds")
            elif 'security' in violation_type:
                recommendations.append("Conduct immediate security investigation")
            elif 'communication' in violation_type:
                recommendations.append("Ensure enough security techniques are applied in communications systems")
            elif 'procedural' in violation_type:
                recommendations.append("Review flight planning procedures and checklists")
        
        for pattern_type, pattern_desc in analysis['complex_patterns'].items():
            if 'weather_experience' in pattern_type:
                recommendations.append("Enhanced weather training required for inexperienced pilots")
            elif 'visibility_experience' in pattern_type:
                recommendations.append("IMC/Low visibility training required for inexperienced pilots")
            elif 'maintenance_experience' in pattern_type:
                recommendations.append("Supervised operations during maintenance periods")
            elif 'narrative' in pattern_type:
                recommendations.append("Conduct detailed narrative investigation")
            elif 'multiple_critical' in pattern_type:
                recommendations.append("Comprehensive safety and systems review required")
            elif 'multiple_interactions' in pattern_type:
                recommendations.append("Review operational procedures for complex scenarios")
        
        ######Narrative based recommendation, depending on the words detected
        if narrative_text:
            narrative_lower = narrative_text.lower()
            if any(keyword in narrative_lower for keyword in ['stall', 'dive', 'nose down', 'overspeed']):
                recommendations.append("Review all stall recovery techniques")
            if any(keyword in narrative_lower for keyword in ['fire', 'explosion']):
                recommendations.append("Review all emergency procedures for fires and explosions")
            if any(keyword in narrative_lower for keyword in ['communication', 'radio','silent']):
                recommendations.append("Test all communications systems for backdoors")
            if any(keyword in narrative_lower for keyword in ['engine failure', 'engine out']):
                recommendations.append("IMMEDIATE: Execute engine failure emergency procedures and abort take-off/prepare for emergency landing")
        
        #######Give any recommendation just for safety assurance 
        if not recommendations:
            recommendations.append("Review flight operations and procedures")
            recommendations.append("Conduct safety assessment")
        
        return recommendations


############Batch prediction for csv files
    def batch_prediction(self, df):
        if model is None:
            return None      
        try:
            df_processed = df.copy()

            for feature in feature_names:
                if feature not in df_processed.columns:
                    if feature in label_encoders:
                        df_processed[feature] = 'Unknown'
                    else:
                        df_processed[feature] = 0
            
            for feature in feature_names:
                if feature in label_encoders:
                    valid_categories = label_encoders[feature].classes_
                    df_processed[feature] = df_processed[feature].astype(str)
                    mask = ~df_processed[feature].isin(valid_categories)
                    if 'Unknown' in valid_categories:
                        df_processed.loc[mask, feature] = 'Unknown'
                    else:
                        df_processed.loc[mask, feature] = valid_categories[0] if len(valid_categories) > 0 else 'Unknown'
                    df_processed[feature] = label_encoders[feature].transform(df_processed[feature])
            
            df_processed = df_processed[feature_names].fillna(0) 
            predictions = model.predict(df_processed)
            probabilities = model.predict_proba(df_processed)[:, 1]
            
            real_anomalies = []
            anomaly_reasons = []
            operation_status = []
            risk_levels = []
            anomaly_narratives = []  
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                flight_data = df.iloc[i].to_dict()
                narrative_text = flight_data.get('narrative_text', '')
                is_anomaly, enhanced_prob, incl_data, _ = self.anomaly_prediction(flight_data, narrative_text)
                real_anomalies.append(is_anomaly)
                
                if is_anomaly:
                    anomaly_analysis = self.analyze_anomaly_types(flight_data, narrative_text)
                    anomaly_reasons.append(anomaly_analysis['threshold_violations'])
                    operation_status.append("Real Anomaly")
                    risk_levels.append(self.risk_level(True, enhanced_prob))
                else:
                    anomaly_reasons.append({})
                    operation_status.append("Normal Operation")
                    risk_levels.append("NORMAL")
            ######https://stackoverflow.com/questions/47117136/predict-for-each-sample-check-the-value-in-pandas-dataframe-and-append-to-a-new
            #####new list, get pred and prob,
            return {
                'model_predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'real_anomalies': real_anomalies,
                'anomaly_reasons': anomaly_reasons,
                'normal_operations': [not anomaly for anomaly in real_anomalies],
                'operation_status': operation_status,
                'risk_levels': risk_levels
            }
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return None
    
    def forensics_findings(self, flight_data, narrative_text=""):
        findings = []     
        combination_data = self.complex_detection(flight_data, narrative_text)
        
        if combination_data.get('critical_wind', 0) == 1:
            findings.append("High wind speeds detected")
        
        if combination_data.get('critical_visibility', 0) == 1:
            findings.append("Reduced visibility conditions")
        
        if combination_data.get('critical_experience', 0) == 1:
            findings.append("Pilot experience is significantly low")
        
        if combination_data.get('critical_maintenance', 0) == 1:
            findings.append("Maintenance inspection overdue")
        
        if combination_data.get('critical_speed', 0) == 1:
            findings.append("Aircraft speed outside normal operating range")
        
        if combination_data.get('security_incident', 0) == 1:
            findings.append("Security incident detected")
        
        if combination_data.get('communication_failure', 0) == 1:
            findings.append("Communication system failure (check for spoofing, eavesdropping or jamming)")
        
        if combination_data.get('flight_procedurev', 0) == 1:
            findings.append("Flight plan not filed")
        
        if combination_data.get('highrisk_narratives', 0) == 1 and narrative_text:
            findings.append("High risk CVR Data detected")
        
        if not findings:
            findings.append("No significant forensic anomalies detected")
        
        return "\n".join(findings)
    
    def safety_recommendations(self, anomaly, probability, flight_data, narrative_text=""):
        recommendations = []
        
        if anomaly:
            recommendations.append("1. Enhance monitoring procedures")
            recommendations.append("2. Analyze flight plans and conduct pilot briefing")
            recommendations.append("3. Review maintenance records")
            
            phase_of_flight = flight_data.get('ev_nr_apt_loc', 'OFAP')
            if phase_of_flight in ['ONAP', 'ON']:
                recommendations.append("4. Implement ground speed monitoring system")
            else:
                recommendations.append("4. Review cruise speed procedures")
        
        if flight_data.get('flight_hours', 1000) < 100:
            recommendations.append("5. Provide experienced supervision")

        if str(flight_data.get('acars_sys', 'Normal')).upper() in ['SLOW', 'FAILED']:
            recommendations.append("Ensure ACARS communications are protected")

        if str(flight_data.get('cpdlc_sys', 'Normal')).upper() in ['SLOW', 'FAILED']:
            recommendations.append("Ensure CPDLC communications are protected")
      
        if not recommendations:
            recommendations.append("No specific safety recommendations - Normal operation")
        
        return "\n".join(recommendations)


    def compliance_check(self, flight_data):
        compliance = {
            "flight_planning": "Compliant" if str(flight_data.get('flt_plan_filed', 'YES')).upper() in ['YES', 'Y'] else "Non-compliant",
            "maintenance": "Compliant" if flight_data.get('afm_hrs_since', 0) <= 500 else "Review needed",
            "pilot_qualifications": "Compliant" if flight_data.get('flight_hours', 1000) >= 50 else "Review needed",
            "speed_compliance": "Compliant" if self.speed_compliance(flight_data) else "Non-compliant"
        }
        return compliance
    
    def speed_compliance(self, flight_data):
        knots = flight_data.get('knots', 0)
        acft_make = str(flight_data.get('acft_make', '')).upper()
        phase_of_flight = flight_data.get('ev_nr_apt_loc', 'OFAP')     
        if phase_of_flight in ['ONAP', 'ON']:
            return knots <= 100 
        else:
            if acft_make in self.largecom:
                return 350 <= knots <= 580
            elif acft_make in self.smallcom:
                return 300 <= knots <= 480
            elif acft_make in (self.private + self.small):
                return 80 <= knots <= 250
            else:
                return 100 <= knots <= 400


    def forensics_report_generation(self, flight_data, prediction_result, narrative_text=""):
        anomaly, probability, incl_data, is_narrative_enhanced = prediction_result    
        random_no = ''.join(random.choices('0123456789', k=3))
        current_user = session.get('user', 'Unknown Analyst')
        confidence_display = self.confidence_score(probability, anomaly)
        
        anomaly_analysis = self.analyze_anomaly_types(flight_data, narrative_text)
        
        report_data = {
            "report_id": f"DFIR-{datetime.now().strftime('%d%m%Y-%H%M%S')}-{random_no}",
            "analyst_username": current_user,
            "timestamp": datetime.now().strftime('%d %B %Y %H:%M:%S'),
            "narrative_enhanced": is_narrative_enhanced,
            "original_narrative": narrative_text if narrative_text else "N/A",
            "flight_data": flight_data,
            "analysis": {
                "anomaly_detected": anomaly,
                "anomaly_probability": probability,
                "confidence_score": confidence_display,
                "risk_level": self.risk_level(anomaly, probability)
            },
            "anomaly_breakdown": anomaly_analysis['threshold_violations'],
            "complex_patterns": anomaly_analysis['complex_patterns'],
            "risk_scores": anomaly_analysis['risk_scores'],
            "recommendations": self.suggest_recommendations(anomaly, probability, flight_data, narrative_text),
            "compliance_check": self.compliance_check(flight_data)
        }
        return report_data

    def ntsb_report_generation(self, flight_data, prediction_result, narrative_text=""):
        anomaly, probability, incl_data, is_narrative_enhanced = prediction_result   
        current_user = session.get('user', 'Unknown Analyst')    
        random_no = ''.join(random.choices('0123456789', k=14))
        confidence_display = self.confidence_score(probability, anomaly)
        narrative_section = f"NARRATIVE: {narrative_text}\n\n" if narrative_text else "NARRATIVE: No narrative provided\n\n"
        
        anomaly_analysis = self.analyze_anomaly_types(flight_data, narrative_text)
        findings = []
        for violation in anomaly_analysis['threshold_violations'].values():
            findings.append(f"- {violation}")
        for pattern in anomaly_analysis['complex_patterns'].values():
            findings.append(f"- {pattern}")


        findings_section = "\n".join(findings) if findings else "No specific anomalies identified"
        
        report = f"""
NATIONAL TRANSPORTATION SAFETY BOARD REPORT

Report Number: NTSB-DFIR-{datetime.now().strftime('%d%m%Y')}
Date of Analysis: {datetime.now().strftime('%B %d, %Y')}
Analyst Username: {current_user}
Narrative Enhanced Analysis: {'Yes' if is_narrative_enhanced else 'No'}

{narrative_section}
EXECUTIVE SUMMARY:
{'ANOMALY DETECTED - Immediate action required' if anomaly else 'NORMAL OPERATION - No anomalies detected'}
Anomaly Probability: {probability:.1%}
Confidence Level: {confidence_display:.1f}%
Risk Assessment: {self.risk_level(anomaly, probability).upper()}

FINDINGS:
{findings_section}

ANALYSIS PARAMETERS:
- Aircraft Make: {flight_data.get('acft_make', 'Unknown')}
- Phase of Flight: {flight_data.get('ev_nr_apt_loc', 'Unknown')}
- Weather Conditions: Wind {flight_data.get('wind_vel_kts', 'Unknown')}kts, Visibility {flight_data.get('vis_sm', 'Unknown')} miles
- Pilot Experience: {flight_data.get('flight_hours', 'Unknown')} hours
- Speed: {flight_data.get('knots', 'Unknown')} knots
- Flight Plan: {flight_data.get('flt_plan_filed', 'Unknown')}

DIGITAL FORENSICS FINDINGS:
{self.forensics_findings(flight_data, narrative_text) if anomaly else "No anomalies detected - Normal flight operation"}

SAFETY RECOMMENDATIONS:
{self.safety_recommendations(anomaly, probability, flight_data, narrative_text)}

CONCLUSION:
This analysis indicates {'Safety issues found requiring further investigation' if anomaly else 'Normal operation - No safety concerns'}.
        """     
        return report

safety_system = Airnormally()

##########For all routing same source as flask web dev ######Python decorators for user/session management from https://dokumen.pub/flask-web-development-developing-web-applications-with-python-first-edition-9781449372620-1449372627.html
####Deepseek "How to use decorators in python for user and access management"
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid'})        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        role = data.get('role', '')
        
        if not username or not password or not role:
            return jsonify({'success': False, 'error': 'Username, password and role are required'})
        
        if (username in USERS and 
            USERS[username]['password'] == password and 
            USERS[username]['role'] == role):
            
            session['user'] = username
            session['role'] = role
            session['logged_in'] = True
            
            return jsonify({
                'success': True,
                'user': username,
                'role': role,
                'message': f'Welcome {username}!'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid credentials or role mismatch'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/check_auth', methods=['GET'])
def check_auth():
    if 'user' in session and session.get('logged_in'):
        return jsonify({
            'success': True,
            'user': session['user'],
            'role': session['role']
        })
    else:
        return jsonify({'success': False, 'error': 'Not authenticated'})

@app.route('/user_info', methods=['GET'])
@login_required
def user_info():
    return jsonify({
        'success': True,
        'user': session['user'],
        'role': session['role']
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_flight():
    try:
        flight_data = {
            'wind_vel_kts': float(request.form.get('wind_vel_kts', 0)),
            'vis_sm': float(request.form.get('vis_sm', 10)),
            'flight_hours': float(request.form.get('flight_hours', 1000)),
            'flt_plan_filed': request.form.get('flt_plan_filed', 'YES'),
            'acft_make': request.form.get('acft_make', 'BOEING'),
            'crew_age': float(request.form.get('crew_age', 40)),
            'cert_max_gr_wt': float(request.form.get('cert_max_gr_wt', 30000)),
            'num_eng': int(request.form.get('num_eng', 2)),
            'afm_hrs_since': float(request.form.get('afm_hrs_since', 10)),
            'knots': float(request.form.get('knots', 170)),
            'acft_expl': request.form.get('acft_expl', 'NO'),
            'acft_fire': request.form.get('acft_fire', 'NO'),
            'ev_nr_apt_loc': request.form.get('ev_nr_apt_loc', 'OFAP'),
            'acars_sys': request.form.get('acars_sys', 'Normal'),
            'cpdlc_sys': request.form.get('cpdlc_sys', 'Normal')
        }
        
        narrative_text = request.form.get('narrative_text', '').strip()
        
        prediction_result = safety_system.anomaly_prediction(flight_data, narrative_text)
        anomaly, probability, incl_data, is_narrative_enhanced = prediction_result
        confidence_display = safety_system.confidence_score(probability, anomaly)
        detailed_analysis = safety_system.analyze_anomaly_types(flight_data, narrative_text)
        recommendations = safety_system.suggest_recommendations(anomaly, probability, flight_data, narrative_text)
        
        forensics_report = None
        ntsb_report = None
        
        if anomaly:
            forensics_report = safety_system.forensics_report_generation(flight_data, prediction_result, narrative_text)
            ntsb_report = safety_system.ntsb_report_generation(flight_data, prediction_result, narrative_text)
        
        response_data = {
            'success': True,
            'prediction': {
                'anomaly': anomaly,
                'probability': probability,
                'confidence': confidence_display,
                'risk_level': safety_system.risk_level(anomaly, probability)
            },
            'narrative_enhanced': is_narrative_enhanced,
            'narrative_provided': bool(narrative_text),
            'threshold_violations': detailed_analysis['threshold_violations'],
            'complex_patterns': detailed_analysis['complex_patterns'],
            'risk_scores': detailed_analysis['risk_scores'],
            'recommendations': recommendations,
            'reports_generated': anomaly, 
        }
        
        if anomaly:
            response_data['forensics_report'] = forensics_report
            response_data['ntsb_report'] = ntsb_report
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/batch_analyze', methods=['POST'])
@login_required
@role_required(['operator', 'analyst'])
def batch_analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}) 
        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}) 
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format. Use CSV file.'})
        batch_results = safety_system.batch_prediction(df)
        
        if batch_results is None:
            return jsonify({'success': False, 'error': 'Batch analysis failed'})
        result_df = df.copy()
        result_df['model_prediction'] = batch_results['model_predictions']
        result_df['probability'] = batch_results['probabilities']
        result_df['real_anomaly'] = batch_results['real_anomalies']
        result_df['normal_operation'] = batch_results['normal_operations']
        result_df['operation_status'] = batch_results['operation_status']
        result_df['risk_level'] = batch_results['risk_levels']
        result_df['anomaly_reasons'] = batch_results['anomaly_reasons']
        total_records = len(result_df)
        model_anomalies = sum(batch_results['model_predictions'])
        real_anomalies = sum(batch_results['real_anomalies'])
        normal_operations = total_records - real_anomalies
        
        summary = {
            'total_records': total_records,
            'model_anomalies': model_anomalies,
            'real_anomalies': real_anomalies,
            'normal_operations': normal_operations,
            'false_positives': model_anomalies - real_anomalies,
            'anomaly_rate': real_anomalies / total_records if total_records > 0 else 0
        } 
        response_data = {
            'success': True,
            'summary': summary,
            'results': result_df.to_dict('records'),
            'records_analyzed': total_records
        }
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_batch_results', methods=['POST'])
@login_required
@role_required(['operator', 'analyst'])
def download_batch_results():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON data'})        
        results = data.get('results', [])
        
        if not results:
            return jsonify({'success': False, 'error': 'No results available'}) 
        df = pd.DataFrame(results)
        
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)     
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'batch_analysis_results_{timestamp}.csv'
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )     
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_report', methods=['POST'])
@login_required
@role_required(['analyst'])
def generate_report():
    try:
        flight_data = {
            'wind_vel_kts': float(request.form.get('wind_vel_kts', 0)),
            'vis_sm': float(request.form.get('vis_sm', 10)),
            'flight_hours': float(request.form.get('flight_hours', 1000)),
            'flt_plan_filed': request.form.get('flt_plan_filed', 'YES'),
            'acft_make': request.form.get('acft_make', 'BOEING'),
            'crew_age': float(request.form.get('crew_age', 40)),
            'cert_max_gr_wt': float(request.form.get('cert_max_gr_wt', 30000)),
            'num_eng': int(request.form.get('num_eng', 2)),
            'afm_hrs_since': float(request.form.get('afm_hrs_since', 10)),
            'knots': float(request.form.get('knots', 170)),
            'acft_expl': request.form.get('acft_expl', 'NO'),
            'acft_fire': request.form.get('acft_fire', 'NO'),
            'ev_nr_apt_loc': request.form.get('ev_nr_apt_loc', 'OFAP'),
            'acars_sys': request.form.get('acars_sys', 'Normal'),
            'cpdlc_sys': request.form.get('cpdlc_sys', 'Normal')
        } 
        narrative_text = request.form.get('narrative_text', '').strip()
        prediction_result = safety_system.anomaly_prediction(flight_data, narrative_text)
        anomaly, probability, incl_data, is_narrative_enhanced = prediction_result
        
        if not anomaly:
            return jsonify({
                'success': False,
                'error': 'No anomalies detected. Report generation is not available for normal operations.'
            })  

        forensics_report = safety_system.forensics_report_generation(flight_data, prediction_result, narrative_text)
        ntsb_report = safety_system.ntsb_report_generation(flight_data, prediction_result, narrative_text)
        
        return jsonify({
            'success': True,
            'forensics_report': forensics_report,
            'ntsb_report': ntsb_report
        })  
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)