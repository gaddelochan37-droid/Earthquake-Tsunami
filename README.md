# Earthquake-Tsunami

## Problem Statement
Predict whether an earthquake will generate a tsunami based on seismic data using 6 different classification models.

## Dataset Description
- **Source**: Kaggle - Global Earthquake-Tsunami Risk Assessment Dataset
- **Instances**: 782
- **Features**: 12
  - magnitude: Earthquake magnitude (Richter scale)
  - cdi: Community Decimal Intensity (0-9)
  - mmi: Modified Mercalli Intensity (1-9)
  - sig: Significance score (0-2000)
  - nst: Number of seismic stations used
  - dmin: Distance to nearest station (degrees)
  - gap: Azimuthal gap (degrees)
  - depth: Earthquake depth (km)
  - latitude: Latitude coordinate
  - longitude: Longitude coordinate
  - Year: Year of occurrence (2001-2022)
  - Month: Month of occurrence (1-12)
- **Target**: tsunami (0=No Tsunami, 1=Tsunami)
- **Type**: Binary Classification

## Repository Structure
```
Earthquake-Tsunami/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── test_data.csv               # Earthquake-Tsunami test dataset           
└── models/                     # Saved trained models (.pkl files)
     └── train_models.py        # Script to train all 6 models
```

## Models Used

| ML Model            |Accuracy|  AUC   |Precision| Recall | F1     | MCC    |
|---------------------|--------|--------|---------|--------|--------|--------|
| Logistic Regression | 0.8599 | 0.9319 | 0.7746  | 0.9016 | 0.8333 | 0.7198 |
| Decision Tree       | 0.8917 | 0.8829 | 0.8793  | 0.8361 | 0.8571 | 0.7707 |
| kNN                 | 0.8854 | 0.9258 | 0.8209  | 0.9016 | 0.8594 | 0.7654 |
| Naive Bayes         | 0.8280 | 0.8613 | 0.7237  | 0.9016 | 0.8029 | 0.6660 |
| Random Forest       | 0.9299 | 0.9640 | 0.8676  | 0.9672 | 0.9147 | 0.8592 |
| XGBoost             | 0.9236 | 0.9679 | 0.8889  | 0.9180 | 0.9032 | 0.8404 |

## Observations

| ML Model            | Observation about model performance                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Good performance with high recall (90.16%). Linear model works well for this dataset. AUC of 0.93 indicates good class separation.        |
| Decision Tree       | Achieved good accuracy (89.17%) but lower AUC (0.88) compared to ensemble methods. Prone to overfitting.                                  |
| kNN                 | Strong performance with balanced metrics. K=5 provides good balance between bias and variance. Scaling helped improve performance.        |
| Naive Bayes         | Lowest accuracy (82.80%) but still acceptable. Assumes feature independence which may not hold for seismic data. Good recall though       |
| Random Forest       | Best overall performance with highest accuracy (92.99%) and MCC (0.86).                                                                   |
| XGBoost             | Excellent AUC (0.97) and strong precision (0.89). Gradient boosting handles complex patterns well. Slightly lower recall than RandomForest|
