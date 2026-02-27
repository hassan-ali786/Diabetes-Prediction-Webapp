🏥 Healthcare Disease Prediction Web App

An end-to-end Machine Learning powered web application that predicts the likelihood of Diabetes using clinical health parameters. This system leverages supervised learning techniques and provides real-time predictions through a professional web interface.

Built with a focus on real-world healthcare AI deployment, model reliability, and user-friendly interaction.

🚀 Project Overview

This application analyzes patient health metrics and predicts whether a patient is at risk of diabetes. It uses a trained Machine Learning model and provides instant predictions through a clean and premium user interface.

The system follows a complete ML lifecycle:

Data preprocessing

Feature scaling

Model training

Model evaluation using Precision, Recall, and F1-score

Model deployment using Flask

Professional frontend integration

🧠 Machine Learning Model

Algorithm Used: Random Forest Classifier

Random Forest was chosen because of its:

High performance on tabular healthcare data

Resistance to overfitting

Strong generalization capability

Stability on small datasets

📊 Dataset Information

Dataset used: National Institute of Diabetes and Digestive and Kidney Diseases Pima Indian Diabetes Dataset

Features:

Pregnancies

Glucose Level

Blood Pressure

Skin Thickness

Insulin Level

BMI

Diabetes Pedigree Function

Age

Target:

Outcome (0 = No Diabetes, 1 = Diabetes)

📈 Model Performance Metrics

Because healthcare datasets are imbalanced, multiple evaluation metrics are used.

Example performance:

Metric	Value
Accuracy	73%
Precision	0.71
Recall	0.68
F1-Score	0.69
Metric Explanation

Accuracy
Overall correct predictions.

Precision
How many predicted diabetic patients were actually diabetic.

Recall
How many actual diabetic patients were correctly identified.

F1-Score
Balanced score combining Precision and Recall.

This is very important in healthcare applications.

🏗️ Project Structure
Healthcare-Disease-Prediction/
│
├── model/
│   ├── diabetes_model.pkl
│   ├── scaler.pkl
│   └── metrics.pkl
│
├── static/
│   ├── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
⚙️ Installation Guide
Step 1: Clone repository
git clone https://github.com/yourusername/Healthcare-Disease-Prediction.git
cd Healthcare-Disease-Prediction
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Train model
python train_model.py
Step 4: Run application
python app.py
🌐 Usage

Open browser

Go to:

http://127.0.0.1:5000

Enter patient details

Click Predict

Get instant result

💻 Technology Stack

Backend:

Python

Flask

Scikit-learn

Pandas

NumPy

Frontend:

HTML

CSS

JavaScript

Machine Learning:

Random Forest Classifier

StandardScaler

Classification Metrics

🔍 Key Features

Real-time disease prediction

Professional healthcare UI

Machine Learning integration

Precision, Recall, F1 evaluation

Clean and modular architecture

Production-ready structure

🎯 Real-World Applications

This system can be used in:

Hospitals

Clinics

Telemedicine platforms

Healthcare AI products

Medical research tools

🔮 Future Improvements

Planned upgrades include:

XGBoost and LightGBM integration

Model accuracy optimization

Prediction probability visualization

Feature importance visualization

User authentication system

Cloud deployment (AWS / Render / Docker)

REST API integration

Database integration for patient records

Deep learning model integration

Multi-disease prediction system

📊 Machine Learning Pipeline
Dataset → Data Cleaning → Feature Scaling → Model Training →
Model Evaluation → Model Saving → Flask Integration → Web Prediction
👨‍💻 Author

Hassan Ali

Computer Science Student
Machine Learning Enthusiast
Aspiring Data Scientist

⭐ Why This Project Matters

This project demonstrates:

End-to-end Machine Learning deployment

Real-world healthcare application

Professional software engineering practices

Production-ready AI system design

This makes it highly valuable for:

Data Science Portfolio

Machine Learning Portfolio

Job Applications

Recruiter Evaluation
