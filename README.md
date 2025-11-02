AirNormally Documentation

This system is developed to detect anomalies in aircrafts during any phase of the flight and generate reports on the anomaly found with some suggested actions. The system detects according to 5 features 
(Speed, Maintenance, Weather, Experience, and Security) and is trained on the RandomForest Classifier Model, which boasts above 90% F1, precision, and recall. There are 4 components tasked with implementing
the system, namely, preprocess.py, train_model.py, app.py, and index.html. 

1. Download the zip, create a virtual menvironment on vs code. And activate the venv and install all imports as seen on the python files. 
2. Unzip the files, and also unzip narratives. The file was too huge to upload on git. 
3. Add the files in a folder on vs code. There should be two subfolders static and templates, 3 python files and 4 csv files.
4. Run preprocess.py via terminal (python preprocess.py), that results in 5 additional datasets being created.
5. Run train_model.py which results in the 3 pkl files being crated, one is for model, one feature and one labels.
6. Run app.py, and click on the address displayed on the terminal. The UI will pop up in chrome/edge depending on your settings. Then test and preffered.
