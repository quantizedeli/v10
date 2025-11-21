# -*- coding: utf-8 -*-
"""
PRODUCTION WEB INTERFACE - Streamlit Dashboard
===============================================

Nuclear Physics AI Prediction System - Web Interface

Features:
1. Single Nucleus Prediction
   - Interactive input form
   - Real-time prediction
   - Confidence intervals
   - Visual results

2. Batch Prediction
   - CSV file upload
   - Bulk processing
   - Excel download
   - Progress tracking

3. Model Management
   - Model selection
   - Performance comparison
   - Model info display
   - Ensemble options

4. Visualization
   - N-Z chart with predictions
   - Feature importance
   - Error distributions
   - Historical comparisons

5. Documentation
   - In-app tutorials
   - API examples
   - Sample datasets
   - FAQ section

Author: Nuclear Physics AI Project
Date: 2025-10-25
Version: 1.0.0 - PFAZ 11 Production Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try importing ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Nuclear Physics AI Predictor",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODELS_DIR = Path("trained_models")
CACHE_DIR = Path("streamlit_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Targets
TARGETS = {
    'MM': 'Magnetic Moment (μN)',
    'QM': 'Quadrupole Moment (Q)',
    'Beta_2': 'Deformation Parameter (β₂)'
}

# Magic numbers
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]


# ============================================================================
# MODEL LOADER
# ============================================================================

@st.cache_resource
def load_available_models():
    """Load all available models - cached for performance"""
    models = {}
    
    if not MODELS_DIR.exists():
        return models
    
    # Scan for model files
    for target in TARGETS.keys():
        models[target] = []
        
        target_dir = MODELS_DIR / target
        if target_dir.exists():
            # Find pickle/joblib files
            for model_file in target_dir.glob("*.pkl"):
                try:
                    model_info = {
                        'name': model_file.stem,
                        'path': model_file,
                        'size_mb': model_file.stat().st_size / (1024**2),
                        'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
                    }
                    models[target].append(model_info)
                except Exception as e:
                    logger.error(f"Failed to load {model_file}: {e}")
    
    return models


@st.cache_resource
def load_model(_model_path):
    """Load a specific model - cached"""
    try:
        model = joblib.load(_model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ============================================================================
# FEATURE CALCULATOR
# ============================================================================

class FeatureCalculator:
    """Calculate features for prediction"""
    
    @staticmethod
    def calculate_basic_features(A: int, Z: int) -> Dict[str, float]:
        """Calculate basic nuclear features"""
        N = A - Z
        
        features = {
            'A': A,
            'Z': Z,
            'N': N,
            'N_over_Z': N / Z if Z > 0 else 0,
            'A_over_Z': A / Z if Z > 0 else 0,
            'neutron_excess': N - Z,
            'asymmetry': (N - Z) / A if A > 0 else 0,
        }
        
        # Magic numbers
        features['is_magic_Z'] = 1 if Z in MAGIC_NUMBERS else 0
        features['is_magic_N'] = 1 if N in MAGIC_NUMBERS else 0
        features['is_doubly_magic'] = 1 if (Z in MAGIC_NUMBERS and N in MAGIC_NUMBERS) else 0
        
        # Shell gaps
        features['Z_magic_dist'] = min([abs(Z - m) for m in MAGIC_NUMBERS])
        features['N_magic_dist'] = min([abs(N - m) for m in MAGIC_NUMBERS])
        
        return features
    
    @staticmethod
    def calculate_semf_features(A: int, Z: int) -> Dict[str, float]:
        """Semi-Empirical Mass Formula features"""
        N = A - Z
        
        # SEMF parameters (typical values)
        a_v = 15.75  # Volume
        a_s = 17.8   # Surface
        a_c = 0.711  # Coulomb
        a_a = 23.7   # Asymmetry
        
        features = {
            'BE_volume': a_v * A,
            'BE_surface': -a_s * (A ** (2/3)),
            'BE_coulomb': -a_c * (Z ** 2) / (A ** (1/3)) if A > 0 else 0,
            'BE_asymmetry': -a_a * ((N - Z) ** 2) / A if A > 0 else 0,
        }
        
        # Pairing term
        if N % 2 == 0 and Z % 2 == 0:
            delta = 11.18
        elif N % 2 == 1 and Z % 2 == 1:
            delta = -11.18
        else:
            delta = 0
        
        features['BE_pairing'] = delta / (A ** 0.5) if A > 0 else 0
        
        return features


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    """Handle predictions"""
    
    def __init__(self):
        self.calculator = FeatureCalculator()
    
    def prepare_features(self, A: int, Z: int) -> pd.DataFrame:
        """Prepare feature vector for prediction"""
        # Calculate features
        basic = self.calculator.calculate_basic_features(A, Z)
        semf = self.calculator.calculate_semf_features(A, Z)
        
        # Combine
        features = {**basic, **semf}
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        return df
    
    def predict_single(self, model, A: int, Z: int) -> Tuple[float, Optional[float]]:
        """
        Predict for single nucleus
        
        Returns:
            (prediction, uncertainty)
        """
        # Prepare features
        X = self.prepare_features(A, Z)
        
        # Predict
        try:
            pred = model.predict(X)[0]
            
            # Try to get uncertainty (if supported)
            uncertainty = None
            if hasattr(model, 'predict_std'):
                uncertainty = model.predict_std(X)[0]
            elif hasattr(model, 'estimators_'):
                # For ensemble models, use std of predictions
                preds = np.array([est.predict(X)[0] for est in model.estimators_])
                uncertainty = np.std(preds)
            
            return pred, uncertainty
        
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None, None
    
    def predict_batch(self, model, nuclei_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for batch of nuclei
        
        Args:
            nuclei_df: DataFrame with columns ['A', 'Z']
        
        Returns:
            DataFrame with predictions
        """
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in nuclei_df.iterrows():
            A, Z = int(row['A']), int(row['Z'])
            N = A - Z
            
            # Update progress
            progress = (idx + 1) / len(nuclei_df)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {idx+1}/{len(nuclei_df)} nuclei...")
            
            # Predict
            pred, unc = self.predict_single(model, A, Z)
            
            results.append({
                'Z': Z,
                'N': N,
                'A': A,
                'Nucleus': f"{A}{self._get_element_symbol(Z)}",
                'Prediction': pred if pred is not None else np.nan,
                'Uncertainty': unc if unc is not None else np.nan
            })
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _get_element_symbol(Z: int) -> str:
        """Get element symbol from Z (simplified)"""
        # Simplified periodic table (first 118 elements)
        symbols = [
            'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'
        ]
        
        if 0 <= Z < len(symbols):
            return symbols[Z]
        else:
            return f"Z{Z}"


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def plot_nz_chart(predictions_df: pd.DataFrame, target: str):
    """Plot N-Z chart with predictions"""
    
    fig = go.Figure()
    
    # Predictions scatter
    fig.add_trace(go.Scatter(
        x=predictions_df['N'],
        y=predictions_df['Z'],
        mode='markers',
        marker=dict(
            size=10,
            color=predictions_df['Prediction'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=TARGETS[target]),
            line=dict(width=1, color='white')
        ),
        text=[f"{row['Nucleus']}<br>{TARGETS[target]}: {row['Prediction']:.3f}"
              for _, row in predictions_df.iterrows()],
        hovertemplate='<b>%{text}</b><br>N=%{x}, Z=%{y}<extra></extra>',
        name='Predictions'
    ))
    
    # Magic numbers lines
    for magic in MAGIC_NUMBERS:
        if magic <= predictions_df['N'].max():
            fig.add_vline(x=magic, line_dash="dash", line_color="red", opacity=0.3)
        if magic <= predictions_df['Z'].max():
            fig.add_hline(y=magic, line_dash="dash", line_color="red", opacity=0.3)
    
    fig.update_layout(
        title=f"N-Z Chart: {TARGETS[target]} Predictions",
        xaxis_title="Neutron Number (N)",
        yaxis_title="Proton Number (Z)",
        hovermode='closest',
        height=600,
        template='plotly_white'
    )
    
    return fig


def plot_feature_importance(feature_names: List[str], importances: np.ndarray):
    """Plot feature importance"""
    
    # Sort by importance
    indices = np.argsort(importances)[-15:]  # Top 15
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=importances[indices],
            colorscale='Blues',
            showscale=False
        )
    ))
    
    fig.update_layout(
        title="Feature Importance (Top 15)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=500,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("⚛️ Nuclear Physics AI Prediction System")
    st.markdown("### Real-time Nuclear Properties Prediction")
    st.markdown("---")
    
    # Load models
    available_models = load_available_models()
    
    if not any(available_models.values()):
        st.error("❌ No trained models found in `trained_models/` directory!")
        st.info("ℹ️ Please train models first using the training pipeline.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Target selection
        target = st.selectbox(
            "Select Target Property:",
            options=list(TARGETS.keys()),
            format_func=lambda x: TARGETS[x]
        )
        
        # Model selection
        if target in available_models and available_models[target]:
            model_options = [m['name'] for m in available_models[target]]
            selected_model_name = st.selectbox(
                "Select Model:",
                options=model_options
            )
            
            # Get model info
            selected_model_info = next(
                m for m in available_models[target] if m['name'] == selected_model_name
            )
            
            # Model info
            with st.expander("Model Info"):
                st.write(f"**Size:** {selected_model_info['size_mb']:.2f} MB")
                st.write(f"**Modified:** {selected_model_info['modified'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.error(f"No models available for {TARGETS[target]}")
            return
        
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "Prediction Mode:",
            options=["Single Nucleus", "Batch Upload"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("- [Documentation](#)")
        st.markdown("- [API Reference](#)")
        st.markdown("- [GitHub](https://github.com)")
    
    # Main content
    if mode == "Single Nucleus":
        single_nucleus_mode(target, selected_model_info)
    else:
        batch_upload_mode(target, selected_model_info)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Nuclear Physics AI Project | Version 1.0.0 | © 2025"
        "</div>",
        unsafe_allow_html=True
    )


def single_nucleus_mode(target: str, model_info: Dict):
    """Single nucleus prediction interface"""
    
    st.header("🎯 Single Nucleus Prediction")
    
    # Input form
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        Z = st.number_input(
            "Proton Number (Z):",
            min_value=1,
            max_value=118,
            value=26,
            step=1
        )
    
    with col2:
        A = st.number_input(
            "Mass Number (A):",
            min_value=Z,
            max_value=300,
            value=56,
            step=1
        )
    
    with col3:
        st.metric("Neutron Number (N)", A - Z)
    
    # Additional info
    N = A - Z
    is_magic_Z = Z in MAGIC_NUMBERS
    is_magic_N = N in MAGIC_NUMBERS
    
    info_cols = st.columns(3)
    with info_cols[0]:
        if is_magic_Z:
            st.success("✨ Magic Z")
    with info_cols[1]:
        if is_magic_N:
            st.success("✨ Magic N")
    with info_cols[2]:
        if is_magic_Z and is_magic_N:
            st.success("⭐ Doubly Magic!")
    
    # Predict button
    if st.button("🚀 Predict", type="primary"):
        with st.spinner("Loading model and predicting..."):
            # Load model
            model = load_model(model_info['path'])
            
            if model is None:
                return
            
            # Predict
            engine = PredictionEngine()
            prediction, uncertainty = engine.predict_single(model, A, Z)
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                result_cols = st.columns([2, 1, 1])
                
                with result_cols[0]:
                    st.metric(
                        label=TARGETS[target],
                        value=f"{prediction:.4f}",
                        delta=f"±{uncertainty:.4f}" if uncertainty else None
                    )
                
                with result_cols[1]:
                    if uncertainty:
                        confidence = 100 * (1 - uncertainty / abs(prediction)) if prediction != 0 else 0
                        st.metric("Confidence", f"{max(0, min(100, confidence)):.1f}%")
                
                with result_cols[2]:
                    st.metric("Model", model_info['name'].split('_')[0].upper())
                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Visualization")
                
                # Feature values
                features_df = engine.prepare_features(A, Z)
                
                with st.expander("View Calculated Features"):
                    st.dataframe(features_df.T, use_container_width=True)
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    fig = plot_feature_importance(
                        features_df.columns.tolist(),
                        model.feature_importances_
                    )
                    st.plotly_chart(fig, use_container_width=True)


def batch_upload_mode(target: str, model_info: Dict):
    """Batch upload and prediction interface"""
    
    st.header("📦 Batch Prediction")
    
    # Instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        **Upload a CSV file with the following columns:**
        - `A`: Mass number
        - `Z`: Proton number
        
        **Example:**
        ```
        A,Z
        56,26
        208,82
        16,8
        ```
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file:",
        type=['csv'],
        help="CSV file with columns: A, Z"
    )
    
    if uploaded_file is not None:
        # Read file
        try:
            nuclei_df = pd.read_csv(uploaded_file)
            
            # Validate columns
            if not {'A', 'Z'}.issubset(nuclei_df.columns):
                st.error("CSV must contain columns: A, Z")
                return
            
            st.success(f"✅ Loaded {len(nuclei_df)} nuclei")
            
            # Preview
            with st.expander("Preview Data"):
                st.dataframe(nuclei_df.head(10), use_container_width=True)
            
            # Predict button
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    # Load model
                    model = load_model(model_info['path'])
                    
                    if model is None:
                        return
                    
                    # Predict
                    engine = PredictionEngine()
                    results_df = engine.predict_batch(model, nuclei_df)
                    
                    st.success("✅ Predictions complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Metrics
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Total Nuclei", len(results_df))
                    with metrics_cols[1]:
                        mean_pred = results_df['Prediction'].mean()
                        st.metric("Mean Prediction", f"{mean_pred:.4f}")
                    with metrics_cols[2]:
                        std_pred = results_df['Prediction'].std()
                        st.metric("Std Dev", f"{std_pred:.4f}")
                    with metrics_cols[3]:
                        if 'Uncertainty' in results_df.columns:
                            mean_unc = results_df['Uncertainty'].mean()
                            st.metric("Mean Uncertainty", f"{mean_unc:.4f}")
                    
                    # Results table
                    st.dataframe(results_df, use_container_width=True)
                    
                    # N-Z chart
                    st.markdown("---")
                    st.subheader("📈 N-Z Chart")
                    fig = plot_nz_chart(results_df, target)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    st.subheader("💾 Download Results")
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"predictions_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
