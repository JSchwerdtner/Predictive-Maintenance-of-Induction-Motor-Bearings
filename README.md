# Predictive-Maintenance-of-Induction-Motor-Bearings
An end-to-end Machine Learning web application built with Streamlit and scikit-learn for industrial motor bearing predictive maintenance, featuring Remaining Useful Life (RUL) estimation and vibration health degradation analytics.


bearing_predictive_maintenance/
├── data/
│   └── ims_bearing_features.csv    # Telemetry feature dataset
├── src/
│   ├── app.py                      # Interactive Streamlit dashboard
│   └── train.py                    # Standalone CLI training script
├── tests/
│   └── test_model.py               # Unit tests for the data pipeline
├── requirements.txt                # Python package dependencies



Train the model: Save this file inside the src/ folder. It supports both Remaining Useful Life (RUL) Regression and Bearing Health Classification, and exports the trained model as a .pkl file.

Start the app with the command: streamlit run src/app.py
