import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Earthquake-Tsunami Prediction", layout="wide")

st.title("Earthquake-Tsunami Risk Prediction App")
st.markdown("---")

# Sidebar for model selection and upload
st.sidebar.header("Model Selection")
model_names = ['Logistic Regression', 'Decision Tree', 'kNN', 
               'Naive Bayes', 'Random Forest', 'XGBoost']
selected_model = st.sidebar.selectbox('Choose a model', model_names)

st.sidebar.markdown("---")
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.info("Upload a CSV file with earthquake data to get predictions")

# Main content
if uploaded_file is None:
    st.header("Welcome to Earthquake-Tsunami Prediction App")
    st.write("👈 Please upload a CSV file using the sidebar to start the prediction")

if uploaded_file is not None:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)
    
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    
    # Check if target column exists
    has_target = 'tsunami' in df.columns
    
    if has_target:
        X = df.drop('tsunami', axis=1)
        y_true = df['tsunami']
    else:
        X = df
        y_true = None
        st.warning("No 'tsunami' column found. Only predictions will be shown (no metrics).")
    
    # Load the selected model
    model_file = f"models/{selected_model.replace(' ', '_').lower()}.pkl"
    
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Apply scaling if needed
        if selected_model in ['Logistic Regression', 'kNN']:
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            X_processed = scaler.transform(X)
        else:
            X_processed = X
        
        # Make predictions
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)[:, 1]
        
        st.markdown("---")
        st.header("Prediction Results")
        
        # Show predictions
        results_df = X.copy()
        results_df['Predicted_Tsunami'] = y_pred
        results_df['Probability'] = y_prob
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Summary")
            st.write(f"Total samples: {len(y_pred)}")
            st.write(f"Predicted Tsunami (1): {sum(y_pred)}")
            st.write(f"Predicted No Tsunami (0): {len(y_pred) - sum(y_pred)}")
        
        with col2:
            st.subheader("Prediction Distribution")
            pred_counts = pd.Series(y_pred).value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            pred_counts.plot(kind='bar', ax=ax, color=['lightblue', 'salmon'])
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Count')
            ax.set_title('Predicted Class Distribution')
            ax.set_xticklabels(['No Tsunami (0)', 'Tsunami (1)'], rotation=0)
            st.pyplot(fig)
        
        # Show detailed results
        st.subheader("Detailed Predictions")
        st.dataframe(results_df)
        
        # If ground truth is available, show metrics
        if has_target:
            st.markdown("---")
            st.header("Evaluation Metrics")
            
            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("AUC Score", f"{auc:.4f}")
            
            with col2:
                st.metric("Precision", f"{prec:.4f}")
                st.metric("Recall", f"{rec:.4f}")
            
            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.markdown("---")
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Tsunami', 'Tsunami'],
                       yticklabels=['No Tsunami', 'Tsunami'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
            
            # Classification Report - Table format
            st.markdown("---")
            st.subheader("Classification Report")
            
            # Get report as dictionary
            report_dict = classification_report(y_true, y_pred, 
                                               target_names=['No Tsunami', 'Tsunami'],
                                               output_dict=True)
            
            # Display accuracy separately
            accuracy_val = report_dict['accuracy']
            st.metric("Overall Accuracy", f"{accuracy_val:.4f}")
            
            st.markdown("**Per-Class and Average Metrics**")
            
            # Create table with precision, recall, f1-score, support columns
            report_data = []
            
            # Per-class rows
            for class_name in ['No Tsunami', 'Tsunami']:
                report_data.append({
                    'Class': class_name,
                    'Precision': f"{report_dict[class_name]['precision']:.2f}",
                    'Recall': f"{report_dict[class_name]['recall']:.2f}",
                    'F1-Score': f"{report_dict[class_name]['f1-score']:.2f}",
                    'Support': f"{int(report_dict[class_name]['support'])}"
                })
            
            # Average rows
            report_data.append({
                'Class': 'macro avg',
                'Precision': f"{report_dict['macro avg']['precision']:.2f}",
                'Recall': f"{report_dict['macro avg']['recall']:.2f}",
                'F1-Score': f"{report_dict['macro avg']['f1-score']:.2f}",
                'Support': f"{int(report_dict['macro avg']['support'])}"
            })
            
            report_data.append({
                'Class': 'weighted avg',
                'Precision': f"{report_dict['weighted avg']['precision']:.2f}",
                'Recall': f"{report_dict['weighted avg']['recall']:.2f}",
                'F1-Score': f"{report_dict['weighted avg']['f1-score']:.2f}",
                'Support': f"{int(report_dict['weighted avg']['support'])}"
            })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df.set_index('Class'), use_container_width=True)
    
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {model_file}")
        st.error(f"Error details: {str(e)}")
        st.info("Please train the models first by running: python train_models.py")
    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
