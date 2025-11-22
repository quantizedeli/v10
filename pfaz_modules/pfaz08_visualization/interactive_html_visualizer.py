"""
Interactive HTML Visualizations
Inspired by ooo.py's comprehensive visualization system
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveHTMLVisualizer:
    """
    Creates interactive HTML visualizations using Plotly
    5 main visualizations from ooo.py
    """
    
    def __init__(self, output_dir: str = 'outputs/interactive_html'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"HTML visualizer initialized: {self.output_dir}")
    
    def create_all_visualizations(self, results_df: pd.DataFrame):
        """
        Create all 5 interactive visualizations
        
        Args:
            results_df: DataFrame with columns:
                - dataset_name, target, config_name
                - ai_r2, final_r2, improvement
                - training_time, num_rules, etc.
        """
        logger.info("\n" + "="*60)
        logger.info("CREATING INTERACTIVE HTML VISUALIZATIONS")
        logger.info("="*60)
        
        try:
            # 1. 3D R² Comparison
            logger.info("1/5: Creating 3D R² comparison...")
            self.create_3d_r2_comparison(results_df)
            
            # 2. Config × Target Heatmap
            logger.info("2/5: Creating config heatmap...")
            self.create_config_heatmap(results_df)
            
            # 3. Feature Importance Analysis
            logger.info("3/5: Creating feature analysis...")
            self.create_feature_analysis(results_df)
            
            # 4. Training Timeline
            logger.info("4/5: Creating training timeline...")
            self.create_training_timeline(results_df)
            
            # 5. Clustering Analysis
            logger.info("5/5: Creating clustering visualization...")
            self.create_clustering_analysis(results_df)
            
            # 6. BONUS: Error Distribution
            logger.info("BONUS: Creating error distribution...")
            self.create_error_distribution(results_df)
            
            logger.info(f"\n[SUCCESS] All visualizations saved to: {self.output_dir}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
    
    def create_3d_r2_comparison(self, df: pd.DataFrame):
        """3D scatter: AI R² vs ANFIS R² vs Improvement"""
        
        fig = px.scatter_3d(
            df,
            x='ai_r2',
            y='final_r2',
            z='improvement',
            color='target',
            size='training_time',
            hover_data=['dataset_name', 'config_name', 'num_rules'],
            title='3D Performance Analysis: AI vs ANFIS',
            labels={
                'ai_r2': 'AI R²',
                'final_r2': 'ANFIS R²',
                'improvement': 'Improvement (Δ)'
            },
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Add diagonal plane (AI R² = ANFIS R²)
        mesh_x = np.linspace(df['ai_r2'].min(), df['ai_r2'].max(), 10)
        mesh_y = mesh_x
        mesh_z = np.zeros_like(mesh_x)
        
        fig.add_trace(go.Surface(
            x=np.outer(mesh_x, np.ones(10)),
            y=np.outer(np.ones(10), mesh_y),
            z=np.outer(np.ones(10), mesh_z),
            opacity=0.3,
            colorscale='Greys',
            showscale=False,
            name='AI = ANFIS'
        ))
        
        fig.update_layout(
            height=700,
            scene=dict(
                xaxis_title='AI R²',
                yaxis_title='ANFIS R²',
                zaxis_title='Improvement',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            )
        )
        
        output_file = self.output_dir / '01_3d_r2_comparison.html'
        fig.write_html(output_file)
        logger.info(f"   Saved: {output_file.name}")
    
    def create_config_heatmap(self, df: pd.DataFrame):
        """Config × Target performance heatmap"""
        
        # Pivot table
        pivot = df.pivot_table(
            values='final_r2',
            index='config_name',
            columns='target',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=np.round(pivot.values, 3),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="Average R²"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Configuration × Target Performance Heatmap',
            xaxis_title='Target Variable',
            yaxis_title='FIS Configuration',
            height=600,
            width=800
        )
        
        output_file = self.output_dir / '02_config_heatmap.html'
        fig.write_html(output_file)
        logger.info(f"   Saved: {output_file.name}")
    
    def create_feature_analysis(self, df: pd.DataFrame):
        """Feature set usage and performance analysis"""
        
        # Parse feature sets from dataset names
        feature_counts = defaultdict(lambda: {'count': 0, 'avg_r2': []})
        
        for _, row in df.iterrows():
            name = row['dataset_name']
            
            # Determine feature set
            if 'AZNSP_beta_p' in name or '7_feat' in name.lower():
                feat = '7 Features (A,Z,N,S,P,beta,p)'
            elif 'AZNSP' in name or '5_feat' in name.lower():
                feat = '5 Features (A,Z,N,S,P)'
            elif 'AZNP_beta' in name:
                feat = '5 Features (A,Z,N,P,beta)'
            elif 'AZNP' in name or '4_feat' in name.lower():
                feat = '4 Features (A,Z,N,P)'
            elif 'AZN_beta' in name:
                feat = '4 Features (A,Z,N,beta)'
            else:
                feat = '3 Features (A,Z,N)'
            
            feature_counts[feat]['count'] += 1
            feature_counts[feat]['avg_r2'].append(row['final_r2'])
        
        # Create DataFrame
        feat_data = []
        for feat, data in feature_counts.items():
            feat_data.append({
                'Feature Set': feat,
                'Usage Count': data['count'],
                'Average R²': np.mean(data['avg_r2']),
                'Std R²': np.std(data['avg_r2'])
            })
        
        feat_df = pd.DataFrame(feat_data).sort_values('Average R²', ascending=False)
        
        # Dual-axis plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart: Usage count
        fig.add_trace(
            go.Bar(
                x=feat_df['Feature Set'],
                y=feat_df['Usage Count'],
                name='Usage Count',
                marker_color='lightblue',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Line chart: Average R²
        fig.add_trace(
            go.Scatter(
                x=feat_df['Feature Set'],
                y=feat_df['Average R²'],
                name='Average R²',
                mode='lines+markers',
                marker=dict(size=12, color='red'),
                line=dict(width=3, color='red')
            ),
            secondary_y=True
        )
        
        # Add error bars
        fig.add_trace(
            go.Scatter(
                x=feat_df['Feature Set'],
                y=feat_df['Average R²'],
                error_y=dict(
                    type='data',
                    array=feat_df['Std R²'],
                    visible=True
                ),
                mode='markers',
                marker=dict(size=0.1, color='red'),
                showlegend=False
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Feature Set Performance Analysis',
            height=600,
            width=1000
        )
        
        fig.update_xaxes(title_text="Feature Set", tickangle=-45)
        fig.update_yaxes(title_text="Usage Count", secondary_y=False)
        fig.update_yaxes(title_text="Average R² ± Std", secondary_y=True)
        
        output_file = self.output_dir / '03_feature_importance.html'
        fig.write_html(output_file)
        logger.info(f"   Saved: {output_file.name}")
    
    def create_training_timeline(self, df: pd.DataFrame):
        """Training progress over time"""
        
        df_sorted = df.sort_index().copy()
        df_sorted['test_index'] = range(len(df_sorted))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('R² Values Over Time', 'Training Duration'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Top plot: R² over time
        for target in df['target'].unique():
            target_df = df_sorted[df_sorted['target'] == target]
            
            fig.add_trace(
                go.Scatter(
                    x=target_df['test_index'],
                    y=target_df['final_r2'],
                    mode='lines+markers',
                    name=f'{target} R²',
                    legendgroup=target,
                    hovertemplate='<b>%{text}</b><br>R²: %{y:.4f}<extra></extra>',
                    text=target_df['dataset_name']
                ),
                row=1, col=1
            )
        
        # Bottom plot: Training time
        for target in df['target'].unique():
            target_df = df_sorted[df_sorted['target'] == target]
            
            fig.add_trace(
                go.Scatter(
                    x=target_df['test_index'],
                    y=target_df['training_time'],
                    mode='markers',
                    name=f'{target} Time',
                    legendgroup=target,
                    showlegend=False,
                    marker=dict(size=6),
                    hovertemplate='Time: %{y:.2f}s<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Test Sequence", row=2, col=1)
        fig.update_yaxes(title_text="R²", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
        
        fig.update_layout(
            height=800,
            title_text="Training Progress Analysis",
            showlegend=True
        )
        
        output_file = self.output_dir / '04_training_timeline.html'
        fig.write_html(output_file)
        logger.info(f"   Saved: {output_file.name}")
    
    def create_clustering_analysis(self, df: pd.DataFrame):
        """K-means clustering with PCA visualization"""
        
        # Select features for clustering
        features = ['ai_r2', 'final_r2', 'improvement', 'training_time', 'num_rules']
        
        # Handle missing values
        X = df[features].fillna(df[features].median())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Prepare plot data
        df_plot = df.copy()
        df_plot['Cluster'] = clusters
        df_plot['PC1'] = X_pca[:, 0]
        df_plot['PC2'] = X_pca[:, 1]
        
        # Cluster labels
        df_plot['Cluster_Label'] = df_plot['Cluster'].apply(lambda x: f'Cluster {x+1}')
        
        # Create scatter plot
        fig = px.scatter(
            df_plot,
            x='PC1',
            y='PC2',
            color='Cluster_Label',
            hover_data=['dataset_name', 'config_name', 'final_r2', 'improvement'],
            title=f'Dataset Clustering Analysis (PCA - {pca.explained_variance_ratio_.sum()*100:.1f}% variance explained)',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
            },
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Add cluster centroids
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        
        fig.add_trace(go.Scatter(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            mode='markers',
            marker=dict(
                size=20,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Centroids',
            showlegend=True
        ))
        
        fig.update_layout(height=700, width=1000)
        
        output_file = self.output_dir / '05_clustering_analysis.html'
        fig.write_html(output_file)
        logger.info(f"   Saved: {output_file.name}")
        
        # Print cluster statistics
        logger.info("\n   Cluster Statistics:")
        for i in range(n_clusters):
            cluster_df = df_plot[df_plot['Cluster'] == i]
            logger.info(f"   Cluster {i+1}: {len(cluster_df)} samples, "
                       f"Avg R²={cluster_df['final_r2'].mean():.4f}")
    
    def create_error_distribution(self, df: pd.DataFrame):
        """BONUS: Error distribution across models"""
        
        # Calculate errors if available
        if 'final_rmse' in df.columns:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('RMSE Distribution', 'R² Distribution',
                              'Improvement Distribution', 'Training Time Distribution'),
                specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                      [{'type': 'histogram'}, {'type': 'histogram'}]]
            )
            
            # RMSE distribution
            fig.add_trace(
                go.Histogram(x=df['final_rmse'], name='RMSE', nbinsx=30,
                           marker_color='indianred'),
                row=1, col=1
            )
            
            # R² distribution
            fig.add_trace(
                go.Histogram(x=df['final_r2'], name='R²', nbinsx=30,
                           marker_color='lightseagreen'),
                row=1, col=2
            )
            
            # Improvement distribution
            fig.add_trace(
                go.Histogram(x=df['improvement'], name='Improvement', nbinsx=30,
                           marker_color='gold'),
                row=2, col=1
            )
            
            # Training time distribution
            fig.add_trace(
                go.Histogram(x=df['training_time'], name='Time', nbinsx=30,
                           marker_color='mediumpurple'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Error and Performance Distributions",
                showlegend=False
            )
            
            output_file = self.output_dir / '06_error_distribution.html'
            fig.write_html(output_file)
            logger.info(f"   Saved: {output_file.name}")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    n_samples = 200
    test_data = {
        'dataset_name': [f'Dataset_{i}' for i in range(n_samples)],
        'target': np.random.choice(['MM', 'QM', 'MM-QM'], n_samples),
        'config_name': np.random.choice(['CFG001', 'CFG002', 'CFG003'], n_samples),
        'ai_r2': np.random.uniform(0.7, 0.95, n_samples),
        'final_r2': np.random.uniform(0.75, 0.98, n_samples),
        'improvement': np.random.uniform(-0.05, 0.15, n_samples),
        'training_time': np.random.uniform(30, 300, n_samples),
        'num_rules': np.random.randint(10, 100, n_samples),
        'final_rmse': np.random.uniform(0.01, 0.1, n_samples)
    }
    
    df = pd.DataFrame(test_data)
    
    # Create visualizations
    visualizer = InteractiveHTMLVisualizer('test_output/html')
    visualizer.create_all_visualizations(df)
    
    print("\n[SUCCESS] Test visualizations created successfully!")
