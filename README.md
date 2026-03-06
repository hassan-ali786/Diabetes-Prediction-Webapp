Healthcare Disease Prediction Web Application

An end-to-end Machine Learning web application that predicts the likelihood of diabetes using clinical health parameters. The system leverages supervised learning techniques and delivers real-time predictions through a professional web interface.

Built with a focus on real-world healthcare AI deployment, model reliability, and user-friendly interaction.

Project Overview

This application analyzes patient health metrics and predicts diabetes risk. It implements a complete ML lifecycle:

Data preprocessing and feature scaling

Model training using Random Forest Classifier

Model evaluation with Precision, Recall, F1-score, and Accuracy

Deployment using Flask

Professional frontend integration for real-time user interaction

Machine Learning Model

Algorithm: Random Forest Classifier

Reasons for choosing Random Forest:

High performance on tabular healthcare datasets

Resistant to overfitting

Strong generalization capability

Stable performance on small datasets

Dataset Information

Dataset: Pima Indian Diabetes Dataset (National Institute of Diabetes and Digestive and Kidney Diseases)

Features:

Pregnancies, Glucose, Blood Pressure, Skin Thickness

Insulin, BMI, Diabetes Pedigree Function, Age

Target: Outcome (0 = No Diabetes, 1 = Diabetes)

Model Performance

Healthcare datasets are often imbalanced, so multiple evaluation metrics are used:

Metric	Value
Accuracy	73%
Precision	0.71
Recall	0.68
F1-Score	0.69

Metric Explanation:

Accuracy: Overall correct predictions

Precision: Proportion of predicted positive cases that are actually positive

Recall: Proportion of actual positive cases correctly identified

F1-Score: Balance between Precision and Recall; critical in healthcare applications

Project Structure
Healthcare-Disease-Prediction/
│
├── model/
│   ├── diabetes_model.pkl
│   ├── scaler.pkl
│   └── metrics.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
Installation Guide

Clone repository:

git clone https://github.com/hassan-ali786/Healthcare-Disease-Prediction.git
cd Healthcare-Disease-Prediction

Install dependencies:

pip install -r requirements.txt

Train model:

python train_model.py

Run application:

python app.py

Open browser at http://127.0.0.1:5000, enter patient details, and click Predict

Technology Stack

Backend: Python, Flask, Scikit-learn, Pandas, NumPy

Frontend: HTML, CSS, JavaScript

Machine Learning: Random Forest Classifier, StandardScaler, Classification Metrics

Key Features

Real-time diabetes prediction

Professional healthcare user interface

Full machine learning integration

Evaluation using multiple performance metrics

Modular, production-ready architecture

Real-World Applications

Hospitals and clinics

Telemedicine platforms

Healthcare AI products

Medical research tools

Future Improvements

Integration with XGBoost and LightGBM

Model accuracy optimization

Prediction probability visualization

Feature importance visualization

User authentication system

Cloud deployment (AWS, Render, Docker)

REST API integration

Database integration for patient records

Deep learning and multi-disease prediction system

Machine Learning Pipeline

Dataset → Data Cleaning → Feature Scaling → Model Training → Model Evaluation → Model Saving → Flask Integration → Web Prediction

Author

Hassan Ali
Aspiring Data Scientist and Machine Learning Engineer

Why This Project Matters

Demonstrates end-to-end ML deployment

Shows experience with real-world healthcare applications

Highlights professional software engineering practices

Adds strong value to Data Science and Machine Learning portfolios

Enhances credibility for job applications and recruiter evaluation
