import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set page config
st.set_page_config(
    page_title="DDoS Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        # Load all models
        rf_model = joblib.load('rf_model.joblib')
        nb_model = joblib.load('nb_model.joblib')
        lr_model = joblib.load('lr_model.joblib')
        
        # Load the scaler
        scaler = joblib.load('scaler.joblib')
        
        return {
            'Random Forest': rf_model,
            'Naive Bayes': nb_model,
            'Logistic Regression': lr_model
        }, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Generate synthetic data for demonstration
@st.cache_data
def generate_sample_data(n_samples=1000):
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(n_samples)]
    
    # Network features
    data = {
        'timestamp': timestamps,
        'src_bytes': np.random.exponential(scale=1000, size=n_samples),
        'dst_bytes': np.random.exponential(scale=800, size=n_samples),
        'count': np.random.poisson(lam=5, size=n_samples),
        'srv_count': np.random.poisson(lam=3, size=n_samples),
        'same_srv_rate': np.random.beta(2, 5, size=n_samples),
        'diff_srv_rate': np.random.beta(1, 10, size=n_samples),
        'dst_host_count': np.random.poisson(lam=15, size=n_samples),
        'dst_host_srv_count': np.random.poisson(lam=10, size=n_samples),
        'dst_host_same_srv_rate': np.random.beta(8, 2, size=n_samples),
        'dst_host_diff_srv_rate': np.random.beta(1, 8, size=n_samples),
        'duration': np.random.exponential(scale=30, size=n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], size=n_samples),
        'service': np.random.choice(['http', 'smtp', 'ftp', 'ssh', 'dns'], size=n_samples),
        'flag': np.random.choice(['SF', 'REJ', 'S0', 'RSTO'], size=n_samples)
    }
    
    # Generate some patterns for DDoS attacks
    for i in range(n_samples):
        if i % 50 < 10:  # Create bursts of attacks
            data['src_bytes'][i] *= 5
            data['count'][i] *= 3
            data['same_srv_rate'][i] = np.random.beta(8, 2)
            data['dst_host_count'][i] *= 2
    
    df = pd.DataFrame(data)
    
    return df

# Function to preprocess data for model input
def preprocess_data(df, scaler):
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Save timestamp separately
    timestamps = processed_df['timestamp'] if 'timestamp' in processed_df.columns else None
    
    # Convert categorical features to one-hot encoding
    processed_df = pd.get_dummies(processed_df, columns=['protocol_type', 'service', 'flag'])
    
    # Drop timestamp for model input if it exists
    if 'timestamp' in processed_df.columns:
        processed_df = processed_df.drop('timestamp', axis=1)
    
    # Scale numerical features
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_cols] = scaler.transform(processed_df[numeric_cols])
    
    return processed_df, timestamps

# Make predictions with multiple models
def predict_ddos(models, selected_model, data, scaler):
    try:
        # Preprocess data
        X, timestamps = preprocess_data(data, scaler)
        
        # Make predictions with selected model
        model = models[selected_model]
        
        # Add timestamps back to results
        results_df = data.copy()
        
        # Get prediction probabilities if available
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            results_df['ddos_probability'] = y_proba[:, 1]
        else:
            # If no predict_proba method, use a placeholder
            results_df['ddos_probability'] = 0.5
            
        # Get predictions
        y_pred = model.predict(X)
        results_df['is_ddos'] = y_pred
        
        return results_df
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        # Add placeholder columns
        data['ddos_probability'] = np.random.beta(1, 10, size=len(data))
        data['is_ddos'] = (data['ddos_probability'] > 0.7).astype(int)
        return data

# Function to evaluate all models on test data
def evaluate_models(models, test_data, scaler, actual_labels=None):
    results = {}
    
    # If we don't have actual labels, we'll use synthetic ones
    if actual_labels is None:
        # For demo purposes, use one model to generate "ground truth"
        X, _ = preprocess_data(test_data, scaler)
        ref_model = list(models.values())[0]
        actual_labels = ref_model.predict(X)
    
    # Process data
    X, _ = preprocess_data(test_data, scaler)
    
    # Evaluate each model
    for name, model in models.items():
        predictions = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions, zero_division=0)
        recall = recall_score(actual_labels, predictions, zero_division=0)
        f1 = f1_score(actual_labels, predictions, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(actual_labels, predictions)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    return results

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices[:15]],
            'Importance': importances[indices[:15]]
        })
        
        # Plot
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                    orientation='h', title='Feature Importance')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        return fig
    else:
        return None

# Main dashboard function
def main():
    st.title("🛡️ DDoS Detection Dashboard")
    
    # Load all models and scaler
    models, scaler = load_models()
    
    if not models or not scaler:
        st.error("Failed to load models. Please check if model files exist.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Detection",
            options=list(models.keys()),
            index=0
        )
        
        # Data source
        data_source = st.radio(
            "Data Source",
            options=["Simulated Data", "Upload Data"],
            index=0
        )
        
        if data_source == "Upload Data":
            uploaded_file = st.file_uploader("Upload network traffic data (CSV)", type="csv")
        
        # Data amount slider for simulated data
        if data_source == "Simulated Data":
            data_amount = st.slider(
                "Amount of Data",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100
            )
            
        # Detection threshold
        detection_threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
        
        # Show model comparison option
        show_comparison = st.checkbox("Show Model Comparison", value=True)
    
    # Load or generate data based on selection
    if data_source == "Upload Data" and uploaded_file is not None:
        with st.spinner("Loading uploaded data..."):
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded data with {len(df)} records")
    else:
        with st.spinner("Generating simulated data..."):
            df = generate_sample_data(n_samples=data_amount if data_source == "Simulated Data" else 1000)
    
    # Make predictions with selected model
    df = predict_ddos(models, selected_model, df, scaler)
    
    # Apply custom threshold
    df['is_ddos_threshold'] = (df['ddos_probability'] > detection_threshold).astype(int)
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3 = st.tabs(["Detection Overview", "Model Performance", "Traffic Analysis"])
    
    # Tab 1: Detection Overview
    with tab1:
        # Key metrics
        st.header("DDoS Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_traffic = len(df)
        ddos_count = df['is_ddos_threshold'].sum()
        ddos_percent = (ddos_count / total_traffic) * 100 if total_traffic > 0 else 0
        alert_level = "High" if ddos_percent > 15 else "Medium" if ddos_percent > 5 else "Low"
        
        col1.metric("Total Traffic", f"{total_traffic:,}")
        col2.metric("DDoS Attacks", f"{ddos_count:,}", f"{ddos_percent:.2f}%")
        col3.metric("Alert Level", alert_level)
        col4.metric("Active Model", selected_model)
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Network traffic visualization with attacks highlighted
        st.subheader("Network Traffic with DDoS Attacks")
        
        # Resample traffic data
        traffic_volume = df_indexed['src_bytes'].resample('5T').mean().reset_index()
        
        # Plot traffic with attack highlights
        fig = go.Figure()
        
        # Add traffic line
        fig.add_trace(go.Scatter(
            x=traffic_volume['timestamp'],
            y=traffic_volume['src_bytes'],
            mode='lines',
            name='Network Traffic',
            line=dict(color='blue')
        ))
        
        # Add attack points
        attack_data = df[df['is_ddos_threshold'] == 1]
        if len(attack_data) > 0:
            fig.add_trace(go.Scatter(
                x=attack_data['timestamp'],
                y=attack_data['src_bytes'],
                mode='markers',
                name='DDoS Attacks',
                marker=dict(color='red', size=10)
            ))
        
        fig.update_layout(
            title='Network Traffic with DDoS Attack Detection',
            xaxis_title='Time',
            yaxis_title='Traffic Volume',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Attack distribution by protocol
        st.subheader("Attack Distribution by Protocol")
        
        # Get protocol columns
        protocol_cols = [col for col in df.columns if col.startswith('protocol_type_')]
        
        if protocol_cols:
            # Create protocol data
            protocol_data = []
            for col in protocol_cols:
                protocol_name = col.replace('protocol_type_', '')
                attack_count = df[(df['is_ddos_threshold'] == 1) & (df[col] == 1)].shape[0]
                normal_count = df[(df['is_ddos_threshold'] == 0) & (df[col] == 1)].shape[0]
                
                protocol_data.append({
                    'Protocol': protocol_name,
                    'Attack': attack_count,
                    'Normal': normal_count
                })
            
            protocol_df = pd.DataFrame(protocol_data)
            
            if not protocol_df.empty:
                # Plot protocol distribution
                fig = px.bar(protocol_df, x='Protocol', y=['Attack', 'Normal'],
                            title='Traffic by Protocol',
                            barmode='group')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent DDoS attacks table
        st.subheader("Recent DDoS Attacks")
        
        # Filter and sort attacks
        recent_attacks = df[df['is_ddos_threshold'] == 1].sort_values('timestamp', ascending=False).head(10)
        
        if len(recent_attacks) > 0:
            # Select columns for display
            display_cols = ['timestamp', 'src_bytes', 'dst_bytes', 'count', 'ddos_probability']
            st.dataframe(recent_attacks[display_cols], use_container_width=True)
        else:
            st.info("No DDoS attacks detected in the current data.")
    
    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance")
        
        if show_comparison:
            # Evaluate all models
            with st.spinner("Evaluating model performance..."):
                # Generate test data for evaluation
                test_data = generate_sample_data(n_samples=500)
                model_results = evaluate_models(models, test_data, scaler)
            
            # Display performance metrics
            st.subheader("Performance Comparison")
            
            # Create metrics dataframe
            metrics_data = []
            for model_name, results in model_results.items():
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1 Score': results['f1']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Plot metrics comparison
            fig = px.bar(metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                        x='Model', y='Score', color='Metric', barmode='group',
                        title='Model Performance Comparison')
            fig.update_layout(height=500, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            
            cols = st.columns(len(models))
            
            for i, (model_name, results) in enumerate(model_results.items()):
                with cols[i]:
                    cm = results['confusion_matrix']
                    
                    # Create heatmap
                    fig = px.imshow(cm,
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=['Normal', 'DDoS'],
                                   y=['Normal', 'DDoS'],
                                   text_auto=True,
                                   color_continuous_scale='Blues',
                                   title=f"{model_name} Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for Random Forest
        if selected_model == "Random Forest":
            st.subheader("Feature Importance")
            
            # Get feature names
            X, _ = preprocess_data(df, scaler)
            feature_names = X.columns.tolist()
            
            # Plot feature importance
            importance_fig = plot_feature_importance(models[selected_model], feature_names)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            else:
                st.info("Feature importance visualization not available for this model")
    
    # Tab 3: Traffic Analysis
    with tab3:
        st.header("Network Traffic Analysis")
        
        # Distribution of attack probabilities
        st.subheader("Attack Probability Distribution")
        
        fig = px.histogram(df, x='ddos_probability', 
                          color='is_ddos_threshold', nbins=30,
                          labels={'ddos_probability': 'DDoS Probability', 'is_ddos_threshold': 'Is DDoS'},
                          title='Distribution of DDoS Probabilities',
                          color_discrete_map={0: 'blue', 1: 'red'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Key features to analyze
        key_features = ['src_bytes', 'dst_bytes', 'count', 'srv_count']
        
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(key_features):
            with col1 if i % 2 == 0 else col2:
                # Create histogram by attack class
                fig = px.histogram(df, x=feature, color='is_ddos_threshold', 
                                 marginal="box", 
                                 labels={'is_ddos_threshold': 'DDoS Attack'},
                                 color_discrete_map={0: 'blue', 1: 'red'},
                                 title=f'Distribution of {feature}')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
        # Attack characteristics comparison
        st.subheader("Normal vs Attack Traffic Characteristics")
        
        # Calculate average values for normal vs attack traffic
        attack_stats = df.groupby('is_ddos_threshold').mean()[
            ['src_bytes', 'dst_bytes', 'count', 'srv_count', 'ddos_probability']
        ].reset_index()
        
        # Reshape for plotting
        attack_stats_melted = attack_stats.melt(id_vars=['is_ddos_threshold'], 
                                              var_name='Feature', 
                                              value_name='Average Value')
        attack_stats_melted['Traffic Type'] = attack_stats_melted['is_ddos_threshold'].map({
            0: 'Normal Traffic', 1: 'DDoS Attack'
        })
        
        # Plot comparison
        fig = px.bar(attack_stats_melted, x='Feature', y='Average Value', color='Traffic Type',
                    barmode='group', title='Comparison of Normal vs Attack Traffic')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()




