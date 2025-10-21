AirNormally Documentation

This system is developed to detect anomalies in aircrafts during any phase of the flight and generate reports on the anomaly found with some suggested actions. The system detects according to 5 features 
(Speed, Maintenance, Weather, Experience, and Security) and is trained on the RandomForest Classifier Model, which boasts above 90% F1, precision, and recall. There are 4 components tasked with implementing
the system, namely, preprocess.py, train_model.py, app.py, and index.html. 

1. preprocess.py
This takes in datasets collected from NTSB under https://data.ntsb.gov/avdata, particularly the avall.zip (Note each month it is updated with more data of recent events). From avall.zip 5 datasets are picked,
aircraft(NTSBaircraft.csv)),events(NTSBevents.csv), flight_time(NTSBflight_time.csv), Flight_Crew(NTSBFlight_Crew.csv) and narratives(NTSB_Narratives). The 5 datasets are preprocessed, selected columns are kept
that will be used for feature engineering, and the 5 datasets are then left joined by ev_id and aircraftkey into one dataset(NTSBMerged_processed.csv) that will be used from henceforth.

3. train_model.py
This component consists of the feature engineering and the model training. The dataset is loaded and is filtered down to the makes that will be used, and the model will be trained on. A function is defined,
def anomaly_detection(df): that performs feature engineering based on the 5 features and returns a df with the engineered features.

First is the speed feature, first ensuring that categorical columns are strings and numeric columns are float/int. Since it is important to use the phase of flight as a base in defining whether thresholds 
are normal, it is ensured that if there is no phase of flight, one is estimated. For speed,anomalies are detected based on crosswind, stalls, and airspeed. 

Second is the maintenance feature, where narratives that include keywords such as 'fault' indicate a maintenance issue, and afm_hrs_since greater than 500 is a maintenance anomaly.

Third is the weather feature that looks at the wind speed and sets anomaly if greater than 25 knots. It also looks at the visibility, a visibility of 2 or lower is an anomaly. Also makes use of narrative keywords
such as storm etc.

Fourth is the experience feature, it looks at the flight hours a pilot has clocked in that aircraft. Flight hours less than 100 shows an inexperience when it comes to that particular aircraft and is an anomaly. 
Also narratives are looked at where words like ‘student’ are also an inexperience sign and are an anomaly. 

Lastly, the security feature. This looks at physical security anomalies such as explosions and fires onboard. Flight plans not being filed are also a sign of security breaches. Some communication systems also 
are breached when they are slow or have failed (usually MIM/jamming attacks) then some narratives are also looked at incl words like hijack.

Then these features are stored in overall_anomaly. Then we select the key features(coloumns)that contribute to the feature anomalies/normalcy. Then it is also ensured that numeric fields are numeric, 
and if there is no value, a median is assigned. The same is done for categorical. Then label encoders are used to convert text cols into num in order to be able to train the model.  Due to class imbalance, most 
of our data is for anomalies, since this NTSB dataset is for accidents and incidents, there was a scaling down in the balancing where at least 67% can be anomalies and 33% can be normal. 
The RandomForest classifier is used, which is great in being able to work in such imbalances; the decision trees eventually come up with the best outcome due to the votes. It independently views cases by case,
which allows even normal cases to be picked up (In this case, some accidents really do happen, the system did not consider instances where accidents happened due to miscommunication? Head-on collisions? 
Cause generally the plane is still doing what it is supposed to do.) There are three resulting models generated anomaly_model.pkl, label_encoders.pkl and feature_names.pkl.



