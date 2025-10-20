import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import tempfile
from datetime import datetime
from io import StringIO, BytesIO
from flask import Flask, send_file, request, render_template, jsonify, session
import base64
import random
from functools import wraps

app = Flask(__name__)
app.secret_key = 'airnormally-secure-key-2024'  # Change this in production

########## User Database ##########
USERS = {
    'public_user': {'password': 'public123', 'role': 'public'},
    'flight_operator': {'password': 'operator456', 'role': 'operator'}, 
    'safety_analyst': {'password': 'analyst789', 'role': 'analyst'},
    'admin': {'password': 'adminSecure2024', 'role': 'analyst'},
    'ntsb_inspector': {'password': 'ntsbSafe123', 'role': 'analyst'}
}

########## Authentication Decorators ##########
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
                return jsonify({'success': False, 'error': 'Authentication required'}), 401
            
            user_role = session.get('role')
            if user_role not in required_roles:
                role_names = " or ".join([r.title() for r in required_roles])
                return jsonify({'success': False, 'error': f'Access denied. {role_names} role required.'}), 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

########## Load Models ##########
try:
    model = joblib.load('anomaly_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None

class Airnormally:
    def __init__(self):
        self.anomaly_types = {
            'speed': 'Speed anomalies such as crosswinds and stalls',
            'maintenance': 'Maintenance related issues',
            'weather': 'Weather issues',
            'experience': 'Experience and Training issues',
            'security': 'Security related issues'
        }
        # Aircraft type classifications for phase-aware analysis
        self.large_com_plane = ['BOEING', 'AIRBUS', 'MCDONNELL DOUGLAS']
        self.small_com_plane = ['EMBRAER', 'BOMBARDIER', 'BOMBARDIER INC']
        self.private_plane = ['BEECH', 'CIRRUS', 'DEHAVILLAND', 'DE HAVILLAND']
        self.small_plane = ['PIPER', 'CESSNA']

    def prediction(self, sample_data):
        if model is None:
            return False, 0.0, "No Model Found"
        try:
            df = pd.DataFrame([sample_data])      
            for feature in feature_names:
                if feature not in df.columns:
                    if feature in label_encoders:
                        df[feature] = 'Unknown'
                    else:
                        df[feature] = 0
            
            for feature in feature_names:
                if feature in label_encoders:
                    current_value = str(df[feature].iloc[0])
                    
                    if current_value not in label_encoders[feature].classes_:
                        valid_classes = label_encoders[feature].classes_
                        if 'Unknown' in valid_classes:
                            default_value = 'Unknown'
                        else:
                            default_value = valid_classes[0] if len(valid_classes) > 0 else 'Unknown'
                        df[feature] = default_value
                    else:
                        df[feature] = current_value
                    
                    df[feature] = label_encoders[feature].transform(df[feature].astype(str))
            
            df = df[feature_names].fillna(0)
            
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
            
            is_actually_anomaly = self.validate_anomaly(prediction, probability, sample_data)
            
            return bool(is_actually_anomaly), float(probability), "Success"
            
        except Exception as e:
            return False, 0.0, str(e)

    def validate_anomaly(self, model_prediction, probability, flight_data):
        if not model_prediction:
            return False
        
        if self.is_normal_operation(flight_data):
            return False
        return True

    def is_normal_operation(self, flight_data):
        phase_of_flight = flight_data.get('ev_nr_apt_loc', 'OFAP')
        
        knots = flight_data.get('knots', 200)
        acft_make = str(flight_data.get('acft_make', '')).upper()
        
        if phase_of_flight in ['ONAP', 'ON']:
            if knots > 100:
                return False
            elif acft_make in self.large_com_plane and knots > 80:
                return False
            elif acft_make in self.small_com_plane and knots > 70:
                return False
            elif acft_make in self.private_plane and knots > 60:
                return False
            elif acft_make in self.small_plane and knots > 50:
                return False
        else:
            if acft_make in self.large_com_plane:
                if knots < 350 or knots > 580:
                    return False
            elif acft_make in self.small_com_plane:
                if knots < 300 or knots > 480:
                    return False
            elif acft_make in (self.private_plane + self.small_plane):
                if knots < 80 or knots > 250:
                    return False
            else:
                if knots < 100 or knots > 400:
                    return False
        
        wind_vel = flight_data.get('wind_vel_kts', 0)
        if wind_vel > 25:
            return False
        
        visibility = flight_data.get('vis_sm', 5)
        if visibility < 3:
            return False
        
        flight_hours = flight_data.get('flight_hours', 200)
        if flight_hours < 100:
            return False
        
        hours_since_inspection = flight_data.get('afm_hrs_since', 10)
        if hours_since_inspection > 500:
            return False
        
        flight_plan = str(flight_data.get('flt_plan_filed', 'YES')).upper()
        if flight_plan in ['NO', 'N', 'FALSE']:
            return False

        ACARS = str(flight_data.get('acars_sys', 'Normal')).upper().strip()
        if ACARS in ['SLOW', 'FAILED']:
            return False

        CPDLC = str(flight_data.get('cpdlc_sys', 'Normal')).upper().strip()
        if CPDLC in ['SLOW', 'FAILED']:
            return False
       
        explosion = str(flight_data.get('acft_expl', 'NO')).upper()
        fire = str(flight_data.get('acft_fire', 'NO')).upper()
        if explosion in ['YES', 'Y', 'TRUE'] or fire in ['YES', 'Y', 'TRUE']:
            return False
        
        return True

    def batch_prediction(self, df):
        if model is None:
            return None      
        try:
            processed_df = df.copy()

            for feature in feature_names:
                if feature not in processed_df.columns:
                    if feature in label_encoders:
                        processed_df[feature] = 'Unknown'
                    else:
                        processed_df[feature] = 0
            
            for feature in feature_names:
                if feature in label_encoders:
                    valid_categories = label_encoders[feature].classes_
                    processed_df[feature] = processed_df[feature].astype(str)
                    mask = ~processed_df[feature].isin(valid_categories)
                    if 'Unknown' in valid_categories:
                        processed_df.loc[mask, feature] = 'Unknown'
                    else:
                        processed_df.loc[mask, feature] = valid_categories[0] if len(valid_categories) > 0 else 'Unknown'
                    
                    processed_df[feature] = label_encoders[feature].transform(processed_df[feature])
            
            processed_df = processed_df[feature_names].fillna(0)
            
            predictions = model.predict(processed_df)
            probabilities = model.predict_proba(processed_df)[:, 1]
            
            real_anomalies = []
            anomaly_reasons = []
            operation_status = []
            risk_levels = []
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                flight_data = df.iloc[i].to_dict()
                
                if pred and not self.is_normal_operation(flight_data):
                    real_anomalies.append(True)
                    anomaly_reasons.append(self.analyze_anomaly_types(flight_data))
                    operation_status.append("Real Anomaly")
                    risk_levels.append(self.risk_level(True, prob))
                else:
                    real_anomalies.append(False)
                    anomaly_reasons.append({})
                    operation_status.append("Normal Operation")
                    risk_levels.append("Normal")
            
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

    def forensics_report_generation(self, flight_data, prediction_result):
        anomaly, probability, status = prediction_result    
        random_suffix = ''.join(random.choices('0123456789', k=3))

        
        
        report_data = {
            "report_id": f"DFR-30092025-182945-{random_suffix}",
            "Analyst": "FTKS24",
            "timestamp": "30 September 2025 18:29:45",
            "flight_data": flight_data,
            "analysis": {
                "anomaly_detected": anomaly,
                "confidence_score": probability,
                "risk_level": self.risk_level(anomaly, probability),
                "status": status
            },
            "anomaly_breakdown": self.analyze_anomaly_types(flight_data) if anomaly else {},
            "recommendations": self.suggest_recommendations(anomaly, probability, flight_data),
            "compliance_check": self.compliance_check(flight_data)
        }
        
        return report_data

    def ntsb_report_generation(self, flight_data, prediction_result):
        anomaly, probability, status = prediction_result       
        report = f"""
NATIONAL TRANSPORTATION SAFETY BOARD Report

Report Number: NTSB-AIR-{datetime.now().strftime('%d%m%Y')}
Date of Analysis: {datetime.now().strftime('%B %d, %Y')}
Analyst Username: FTKS24
Case ID: {flight_data.get('ev_id', 'SYSTEM-ANALYSIS')}

EXECUTIVE SUMMARY:
{'ANOMALY DETECTED - Immediate action required' if anomaly else 'NORMAL OPERATION - No anomalies detected'}
Confidence Level: {probability:.1%}
Risk Assessment: {self.risk_level(anomaly, probability).upper()}

ANALYSIS PARAMETERS:
- Aircraft Make: {flight_data.get('acft_make', 'Unknown')}
- Phase of Flight: {flight_data.get('ev_nr_apt_loc', 'Unknown')}
- Weather Conditions: {flight_data.get('wx_cond_basic', 'Unknown')}
- Pilot Experience: {flight_data.get('flight_hours', 'Unknown')} hours
- Speed: {flight_data.get('knots', 'Unknown')} knots
- Flight Plan: {flight_data.get('flt_plan_filed', 'Unknown')}
- Security Threat (Explosion): {flight_data.get('acft_expl', 'Unknown')}
- Security Threat (Fire): {flight_data.get('acft_fire', 'Unknown')}
- Cybersecurity Threat (ACARS): {flight_data.get('acars_sys', 'Unknown')}
- Cybersecurity Threat (CPDLC): {flight_data.get('cpdlc_sys', 'Unknown')}

DIGITAL FORENSICS FINDINGS:
{self.forensics_findings(flight_data) if anomaly else "No anomalies detected - Normal flight operation"}

SAFETY RECOMMENDATIONS:
{self.safety_recommendations(anomaly, probability, flight_data)}

CONCLUSION:
This analysis indicates {'Safety issues found requiring further investigation' if anomaly else 'Normal operation - No safety concerns'}.
        """
        
        return report

    def risk_level(self, anomaly, probability):
        if not anomaly:
            return "normal"
        elif probability >= 0.8:
            return "high"
        elif probability >= 0.6:
            return "medium"
        elif probability >= 0.4:
            return "low"
        else:
            return "very low"

    def analyze_anomaly_types(self, flight_data):
        analysis = {}
        
        knots = flight_data.get('knots')
        acft_make = str(flight_data.get('acft_make', '')).upper()
        phase_of_flight = flight_data.get('ev_nr_apt_loc', 'OFAP')

        if knots and acft_make:
            if phase_of_flight in ['ONAP', 'ON']:
                if knots > 100:
                    analysis['speed'] = f"CRITICAL: Excessive ground speed: {knots}kts"
                elif acft_make in self.large_com_plane and knots > 80:
                    analysis['speed'] = f"High airport speed: {knots}kts for {acft_make}"
                elif acft_make in self.small_com_plane and knots > 70:
                    analysis['speed'] = f"High airport speed: {knots}kts for {acft_make}"
                elif acft_make in self.private_plane and knots > 60:
                    analysis['speed'] = f"High airport speed: {knots}kts for {acft_make}"
                elif acft_make in self.small_plane and knots > 50:
                    analysis['speed'] = f"High airport speed: {knots}kts for {acft_make}"
            else:
                if acft_make in self.large_com_plane:
                    if knots < 350:
                        analysis['speed'] = f"Slow cruise: {knots}kts for {acft_make}"
                    elif knots > 580:
                        analysis['speed'] = f"Overspeed: {knots}kts for {acft_make}"
                elif acft_make in self.small_com_plane:
                    if knots < 300:
                        analysis['speed'] = f"Slow cruise: {knots}kts for {acft_make}"
                    elif knots > 480:
                        analysis['speed'] = f"Overspeed: {knots}kts for {acft_make}"
                elif acft_make in (self.private_plane + self.small_plane):
                    if knots < 80:
                        analysis['speed'] = f"Slow flight: {knots}kts for {acft_make}"
                    elif knots > 250:
                        analysis['speed'] = f"Overspeed: {knots}kts for {acft_make}"
        
        wind_vel = flight_data.get('wind_vel_kts', 0)
        if wind_vel > 25:
            analysis['crosswind'] = f"High crosswind: {wind_vel}kts"

        if flight_data.get('afm_hrs_since', 0) > 500:
            analysis['maintenance'] = "Extended time since last inspection"
        
        if flight_data.get('vis_sm', 10) < 3:
            analysis['weather'] = "Low visibility conditions"
        
        if flight_data.get('flight_hours', 1000) < 100:
            analysis['experience'] = "Low pilot flight hours"

        ACARS = str(flight_data.get('acars_sys', 'Normal')).upper().strip()
        if ACARS in ['SLOW', 'FAILED']:
            if 'security' in analysis:
                analysis['security'] += f" | ACARS System- Possible communication jamming"
            else:
                analysis['security'] = f"ACARS System - Possible communication jamming"

        CPDLC = str(flight_data.get('cpdlc_sys', 'Normal')).upper().strip()
        if CPDLC in ['SLOW', 'FAILED']:
            if 'security' in analysis:
                analysis['security'] += f" | CPDLC System- Possible communication jamming"
            else:
                analysis['security'] = f"CPDLC System - Possible communication jamming"

        explosion = str(flight_data.get('acft_expl', 'NO')).upper()
        fire = str(flight_data.get('acft_fire', 'NO')).upper()
        if explosion in ['YES', 'Y', 'TRUE'] or fire in ['YES', 'Y', 'TRUE']:
            if 'security' in analysis:
                analysis['security'] += f" | Explosion or Fire Detected"
            else:
                analysis['security'] = f"Explosion or Fire Detected"
        
        if str(flight_data.get('flt_plan_filed', 'YES')).upper() in ['NO', 'N', 'FALSE']:
            analysis['flightplan'] = "Flight plan not filed"
    
        return analysis

    def suggest_recommendations(self, anomaly, probability, flight_data):
        recommendations = []
        
        if anomaly:
            if probability > 0.7:
                recommendations.append("Immediate Action Required: Conduct safety analysis before next flight")
                recommendations.append("Suggestion: Review Pilot Training on particular aircraft")
            else:
                recommendations.append("Continue monitoring flight parameters")
                recommendations.append("Review standard operating procedures")
        else:
            recommendations.append("Continue normal operations")
            recommendations.append("Maintain current safety protocols")
        
        if anomaly:
            phase_of_flight = flight_data.get('ev_nr_apt_loc', 'OFAP')
            knots = flight_data.get('knots', 170)
            acft_make = str(flight_data.get('acft_make', '')).upper()
            
            if phase_of_flight in ['ONAP', 'ON']:
                if knots > 80:
                    recommendations.append("SPEED CRITICAL: Reduce ground speed immediately")
                elif knots > 60:
                    recommendations.append("Speed: Monitor ground operations carefully")
            else:
                if acft_make in self.large_com_plane and (knots < 350 or knots > 580):
                    recommendations.append("Speed: Adjust to normal cruise range (350-580 kts)")
                elif acft_make in self.small_com_plane and (knots < 300 or knots > 480):
                    recommendations.append("Speed: Adjust to normal cruise range (300-480 kts)")

            if flight_data.get('vis_sm', 10) < 3:
                recommendations.append("Weather: Consider alternative routing")

            if flight_data.get('flight_hours', 1000) < 100:
                recommendations.append("Training: Consider additional supervision")
            
            if flight_data.get('afm_hrs_since', 0) > 500:
                recommendations.append("Maintenance: Schedule inspection")

            ACARS = str(flight_data.get('acars_sys', 'Normal')).upper()
            if ACARS in ['SLOW', 'FAILED']:
                recommendations.append("Security: Encrypt ACARS communication, implement firewall or IDS")

            CPDLC = str(flight_data.get('cpdlc_sys', 'Normal')).upper()
            if CPDLC in ['SLOW', 'FAILED']:
                recommendations.append("Security: Encrypt CPDLC communication, implement firewall or IDS")

            explosion = str(flight_data.get('acft_expl', 'NO')).upper()
            fire = str(flight_data.get('acft_fire', 'NO')).upper()
            if explosion in ['YES', 'Y', 'TRUE'] or fire in ['YES', 'Y', 'TRUE']:
                recommendations.append("Security: Explosion or Fire investigate immediately")
        
        return recommendations
    
    def compliance_check(self, flight_data):
        compliance = {
            "flight_planning": "Compliant" if str(flight_data.get('flt_plan_filed', 'YES')).upper() in ['YES', 'Y', 'TRUE'] else "Non-compliant",
            "maintenance": "Compliant" if flight_data.get('afm_hrs_since', 0) <= 500 else "Review needed",
            "pilot_qualifications": "Compliant" if flight_data.get('flight_hours', 1000) >= 50 else "Review needed",
            "speed_compliance": "Compliant" if self._check_speed_compliance(flight_data) else "Non-compliant"
        }
        
        return compliance
    
    def _check_speed_compliance(self, flight_data):
        knots = flight_data.get('knots', 0)
        acft_make = str(flight_data.get('acft_make', '')).upper()
        phase_of_flight = flight_data.get('ev_nr_apt_loc', 'OFAP')
        
        if phase_of_flight in ['ONAP', 'ON']:
            return knots <= 100 
        else:
            if acft_make in self.large_com_plane:
                return 350 <= knots <= 580
            elif acft_make in self.small_com_plane:
                return 300 <= knots <= 480
            elif acft_make in (self.private_plane + self.small_plane):
                return 80 <= knots <= 250
            else:
                return 100 <= knots <= 400 
    
    def forensics_findings(self, flight_data):
        findings = []
        
        phase_of_flight = flight_data.get('ev_nr_apt_loc', 'OFAP')
        knots = flight_data.get('knots', 0)
        acft_make = str(flight_data.get('acft_make', '')).upper()
        
        if phase_of_flight in ['ONAP', 'ON AIRPORT', 'ON']:
            if knots > 100:
                findings.append("CRITICAL: Excessive ground speed detected")
            elif knots > 80:
                findings.append("High-speed ground operation")
        else:
            if acft_make in self.large_com_plane and (knots < 350 or knots > 580):
                findings.append("Aircraft outside normal cruise speed")
        
        if flight_data.get('wind_vel_kts', 0) > 25:
            findings.append("High wind speeds")
        
        if flight_data.get('vis_sm', 10) < 3:
            findings.append("Reduced visibility conditions")
        
        if flight_data.get('flight_hours', 1000) < 100:
            findings.append("Pilot experience is significantly low")

        if str(flight_data.get('flt_plan_filed', 'YES')).upper() in ['NO', 'N', 'FALSE']:
            findings.append("Flight plan not filed")

        ACARS = str(flight_data.get('acars_sys', 'Normal')).upper()
        if ACARS in ['SLOW', 'FAILED']:
            findings.append("ACARS Failure check for spoofing, eavesdropping or jamming")

        CPDLC = str(flight_data.get('cpdlc_sys', 'Normal')).upper()
        if CPDLC in ['SLOW', 'FAILED']:
            findings.append("CPDLC Failure check for spoofing, eavesdropping or jamming")

        explosion = str(flight_data.get('acft_expl', 'NO')).upper()
        fire = str(flight_data.get('acft_fire', 'NO')).upper()
        if explosion in ['YES', 'Y', 'TRUE'] or fire in ['YES', 'Y', 'TRUE']:
            findings.append("Fire or Explosion detected")

        if flight_data.get('afm_hrs_since', 0) > 500:
            findings.append("Maintenance inspection overdue")

        if not findings:
            findings.append("No significant forensic anomalies detected")
        
        return "\n".join(findings)
    
    def safety_recommendations(self, anomaly, probability, flight_data):
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

        ACARS = str(flight_data.get('acars_sys', 'Normal')).upper()
        if ACARS in ['SLOW', 'FAILED']:
            recommendations.append("Ensure ACARS communications are protected")

        CPDLC = str(flight_data.get('cpdlc_sys', 'Normal')).upper()
        if CPDLC in ['SLOW', 'FAILED']:
            recommendations.append("Ensure CPDLC communications are protected")
      
        if not recommendations:
            recommendations.append("No specific safety recommendations - Normal operation")
        
        return "\n".join(recommendations)

safety_system = Airnormally()

########## Authentication Routes ##########
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON data'})
            
        username = data.get('username', '').strip()
        password = data.get('password', '')
        role = data.get('role', '')
        
        if not username or not password or not role:
            return jsonify({'success': False, 'error': 'Username, password and role are required'})
        
        # Check credentials
        if (username in USERS and 
            USERS[username]['password'] == password and 
            USERS[username]['role'] == role):
            
            # Store user in session
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

########## Protected Application Routes ##########
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
        
        prediction_result = safety_system.prediction(flight_data)
        anomaly, probability, status = prediction_result
        
        forensics_report = None
        ntsb_report = None
        
        if anomaly:
            forensics_report = safety_system.forensics_report_generation(flight_data, prediction_result)
            ntsb_report = safety_system.ntsb_report_generation(flight_data, prediction_result)
        
        response_data = {
            'success': True,
            'prediction': {
                'anomaly': anomaly,
                'probability': probability,
                'status': status
            },
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
        
        prediction_result = safety_system.prediction(flight_data)
        anomaly, probability, status = prediction_result
        
        if not anomaly:
            return jsonify({
                'success': False,
                'error': 'No anomalies detected. Report generation is not available for normal operations.'
            })
        
        forensics_report = safety_system.forensics_report_generation(flight_data, prediction_result)
        ntsb_report = safety_system.ntsb_report_generation(flight_data, prediction_result)
        
        return jsonify({
            'success': True,
            'forensics_report': forensics_report,
            'ntsb_report': ntsb_report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)