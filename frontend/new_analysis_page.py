import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64
import scipy.stats as stats

# Import analysis modules
try:
    from analysis.lr_mc_analysis import IonicLiquidAnalysis, FragmentAnalyzer, MonteCarloAnalyzer, PCAAnalyzer, ClusterAnalyzer
except ImportError:
    # Use importlib for dynamic import if the module name contains hyphens
    import importlib.util
    spec = importlib.util.spec_from_file_location("lr_mc_analysis", 
                                                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                            "analysis", "lr-mc-analysis.py"))
    lr_mc_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lr_mc_analysis)
    IonicLiquidAnalysis = lr_mc_analysis.IonicLiquidAnalysis
    FragmentAnalyzer = lr_mc_analysis.FragmentAnalyzer
    MonteCarloAnalyzer = lr_mc_analysis.MonteCarloAnalyzer
    PCAAnalyzer = lr_mc_analysis.PCAAnalyzer
    ClusterAnalyzer = lr_mc_analysis.ClusterAnalyzer

def run_analysis():
    """Main function to run the analysis page"""
    st.set_page_config(page_title="Ionic Liquid Analysis", layout="wide")
    
    st.title("Ionic Liquid Property Analysis")
    
    # Upload data
    st.write("Upload the exported data from the Ionic Liquid Optimizer")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Option to use sample data
    use_sample_data = st.checkbox("Use sample data from data folder", value=False)
    
    # Load data
    data_loaded = False
    combinations = None
    
    # Load from sample data if selected
    if use_sample_data:
        sample_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "thermal_ILs_long-list.csv")
        if os.path.exists(sample_data_path):
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        combinations = pd.read_csv(sample_data_path, encoding=encoding)
                        data_loaded = True
                        st.success(f"Successfully loaded sample data using {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        st.error(f"Error loading sample data with {encoding} encoding: {e}")
                        break
                
                if not data_loaded:
                    st.error("Could not load the sample data with any of the attempted encodings")
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
        else:
            st.error(f"Sample data file not found at {sample_data_path}")
    # Only load from uploaded file if sample data is not used
    elif uploaded_file is not None:
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    combinations = pd.read_csv(uploaded_file, encoding=encoding)
                    data_loaded = True
                    st.success(f"Successfully loaded data using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    # Reset file pointer for next attempt
                    uploaded_file.seek(0)
                    continue
                except Exception as e:
                    st.error(f"Error loading with {encoding} encoding: {e}")
                    break
            
            if not data_loaded:
                st.error("Could not load the file with any of the attempted encodings")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    
    if data_loaded:
        # Convert to list of dictionaries for compatibility with analysis code
        
        # Debug information
        st.subheader("Data Preview")
        st.write(f"Loaded data with {len(combinations)} rows and {len(combinations.columns)} columns")
        st.write("Column names:", combinations.columns.tolist())
        
        # Check for required fragment columns with proper casing
        fragment_cols = ['cation', 'anion', 'alkyl_chain', 'functional_group']
        fragment_cols_capitalized = ['Cation', 'Anion', 'Alkyl Chain', 'Functional Group']
        
        # Create a mapping of capitalized column names to lowercase with underscores
        column_mapping = {}
        for i, col in enumerate(fragment_cols_capitalized):
            if col in combinations.columns:
                column_mapping[col] = fragment_cols[i]
        
        # Rename columns to match expected format
        if column_mapping:
            combinations = combinations.rename(columns=column_mapping)
            st.success(f"Renamed columns to match expected format: {column_mapping}")
        
        # Check if any required columns are still missing
        missing_cols = [col for col in fragment_cols if col not in combinations.columns]
        if missing_cols:
            st.warning(f"Missing expected fragment columns: {missing_cols}")
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Heat Capacity (J/mol·K)', 'Density (kg/m³)', 'Toxicity (IC50 in mM)', 
                       'Solubility (g/L)', 'Hydrophobicity']
        
        for col in numeric_cols:
            if col in combinations.columns:
                try:
                    combinations[col] = pd.to_numeric(combinations[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Could not convert {col} to numeric: {e}")
        
        # Show a preview of the data
        st.dataframe(combinations.head())
        
        # Convert to list of dictionaries for compatibility with analysis code
        combinations_list = combinations.to_dict('records')
        
        # Store data in session state for access across tabs
        st.session_state.combinations = combinations_list
        
        # Create tabs for different analyses
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
            "Fragment Influence Analysis", 
            "Monte Carlo Uncertainty Analysis",
            "Property Correlation Analysis",
            "Principal Component Analysis",
            "Cluster Analysis"
        ])
        
        with analysis_tab1:
            st.header("Fragment Influence Analysis")
            st.write("Analyze how different fragments influence ionic liquid properties using linear regression.")
            
            # Select property for analysis
            property_options = [col for col in combinations.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score'] and pd.api.types.is_numeric_dtype(combinations[col])]
            selected_property = st.selectbox("Select property to analyze:", property_options)
            
            if st.button("Run Fragment Analysis"):
                with st.spinner("Running fragment analysis..."):
                    # Create fragment analyzer
                    fragment_analyzer = FragmentAnalyzer(combinations_list)
                    
                    # Run analysis
                    results = fragment_analyzer.analyze_property(selected_property)
                    
                    # Display results
                    st.subheader(f"Fragment Influence on {selected_property}")
                    st.dataframe(results)
                    
                    # Plot top influential fragments
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_n = min(15, len(results))
                    
                    # Plot positive and negative influences separately
                    positive_influences = results[results['Influence'] > 0].head(top_n)
                    negative_influences = results[results['Influence'] < 0].tail(top_n)
                    
                    # Combine for plotting
                    plot_data = pd.concat([positive_influences.head(top_n//2), 
                                          negative_influences.head(top_n//2)])
                    
                    # Create plot
                    sns.barplot(data=plot_data, x='Influence', y='Fragment', ax=ax)
                    ax.set_title(f"Top Fragment Influences on {selected_property}")
                    ax.set_xlabel("Influence Coefficient")
                    ax.set_ylabel("Fragment")
                    
                    # Display plot
                    st.pyplot(fig)
                    
                    # Provide download link for results
                    csv = results.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="fragment_analysis_{selected_property}.csv">Download Fragment Analysis Results</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        with analysis_tab2:
            st.header("Monte Carlo Uncertainty Analysis")
            st.write("Analyze uncertainty in property predictions using Monte Carlo simulation.")
            
            # Get only numeric columns for Monte Carlo analysis
            numeric_cols = combinations.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                st.error("Monte Carlo analysis requires numeric properties. Your data doesn't have any numeric columns.")
            else:
                # Select property for analysis
                selected_property = st.selectbox("Select property for Monte Carlo analysis:", 
                                               numeric_cols, 
                                               key="mc_property")
                
                # Set simulation parameters
                col1, col2 = st.columns(2)
                with col1:
                    num_simulations = st.slider("Number of simulations:", min_value=100, max_value=2000, value=500, step=100)
                with col2:
                    uncertainty_percentage = st.slider("Uncertainty percentage:", min_value=1, max_value=30, value=10, step=1)
                
                if st.button("Run Monte Carlo Analysis"):
                    with st.spinner("Running Monte Carlo simulation..."):
                        try:
                            # Create Monte Carlo analyzer
                            mc_analyzer = MonteCarloAnalyzer(combinations_list)
                            
                            # Run analysis
                            simulation_results = mc_analyzer.run_analysis(
                                selected_property, 
                                uncertainty_percentage=uncertainty_percentage,
                                n_simulations=num_simulations
                            )
                            
                            # Check if there was an error
                            if isinstance(simulation_results, dict) and 'error' in simulation_results:
                                st.error(f"Error in Monte Carlo analysis: {simulation_results['error']}")
                            else:
                                # Display results
                                st.subheader(f"Monte Carlo Simulation Results for {selected_property}")
                                
                                # Calculate statistics
                                mean_value = np.mean(simulation_results)
                                std_dev = np.std(simulation_results)
                                confidence_interval_95 = (
                                    mean_value - 1.96 * std_dev,
                                    mean_value + 1.96 * std_dev
                                )
                                
                                # Display statistics
                                st.write(f"Mean value: {mean_value:.4f}")
                                st.write(f"Standard deviation: {std_dev:.4f}")
                                st.write(f"95% Confidence interval: ({confidence_interval_95[0]:.4f}, {confidence_interval_95[1]:.4f})")
                                
                                # Plot histogram of simulation results
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(simulation_results, kde=True, ax=ax)
                                ax.axvline(mean_value, color='red', linestyle='--', label=f"Mean: {mean_value:.4f}")
                                ax.axvline(confidence_interval_95[0], color='green', linestyle=':', label=f"95% CI Lower: {confidence_interval_95[0]:.4f}")
                                ax.axvline(confidence_interval_95[1], color='green', linestyle=':', label=f"95% CI Upper: {confidence_interval_95[1]:.4f}")
                                ax.set_title(f"Monte Carlo Simulation Results for {selected_property}")
                                ax.set_xlabel(f"{selected_property} Value")
                                ax.set_ylabel("Frequency")
                                ax.legend()
                                
                                # Display plot
                                st.pyplot(fig)
                                
                                # Provide download link for results
                                results_df = pd.DataFrame(simulation_results, columns=[selected_property])
                                csv = results_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="monte_carlo_{selected_property}.csv">Download Monte Carlo Simulation Results</a>'
                                st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error in Monte Carlo analysis: {str(e)}")
                        finally:
                            pass
        
        with analysis_tab3:
            st.header("Property Correlation Analysis")
            st.write("Analyze correlations between different ionic liquid properties.")
            
            # Get numeric columns for correlation analysis
            numeric_cols = combinations.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.error("No numeric columns found for correlation analysis.")
            else:
                # Select properties for correlation analysis
                selected_properties = st.multiselect(
                    "Select properties for correlation analysis:",
                    options=numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if selected_properties:
                    if len(selected_properties) < 2:
                        st.warning("Please select at least 2 properties for correlation analysis.")
                    else:
                        # Calculate correlation matrix
                        try:
                            correlation_matrix = combinations[selected_properties].corr()
                            
                            # Plot correlation heatmap
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                            ax.set_title("Property Correlation Matrix")
                            st.pyplot(fig)
                            
                            # Plot pairplot for selected properties
                            st.subheader("Pairwise Relationships")
                            pairplot_fig = sns.pairplot(combinations[selected_properties])
                            st.pyplot(pairplot_fig.fig)
                        except Exception as e:
                            st.error(f"Error calculating correlations: {e}")
                else:
                    st.info("Please select properties for correlation analysis.")
        
        with analysis_tab4:
            st.header("Principal Component Analysis")
            st.write("Analyze the relationships between properties using Principal Component Analysis (PCA).")
            
            # Get only numeric columns for PCA
            numeric_cols = combinations.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("PCA requires at least 2 numeric properties. Your data doesn't have enough numeric columns.")
            else:
                # Select properties for PCA
                selected_properties = st.multiselect(
                    "Select properties for PCA:",
                    options=numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                # Set number of components
                n_components = st.slider("Number of principal components:", 2, min(5, len(numeric_cols)), 2)
                
                # Option to scale data
                scale_data = st.checkbox("Scale data before PCA", value=True)
                
                # Select coloring option
                fragment_cols = ['cation', 'anion', 'alkyl_chain', 'functional_group']
                available_fragment_cols = [col for col in fragment_cols if col in combinations.columns]
                color_options = ["None"] + available_fragment_cols
                color_by = st.selectbox("Color points by:", color_options)
                
                if len(selected_properties) < 2:
                    st.error("Please select at least two properties for PCA.")
                else:
                    with st.spinner("Running PCA..."):
                        try:
                            # Create PCA analyzer
                            pca_analyzer = PCAAnalyzer(combinations_list)
                            
                            # Run analysis
                            pca_results = pca_analyzer.run_analysis(n_components=n_components, 
                                                                  property_names=selected_properties, 
                                                                  fragment_types=None if color_by == "None" else [color_by],
                                                                  scale=scale_data)
                            
                            # Check if PCA was successful
                            if pca_results['pca_model'] is None:
                                st.error("PCA analysis failed. Please check your data and selected properties.")
                            else:
                                # Display results
                                st.subheader("PCA Results")
                                
                                # Display explained variance
                                explained_variance = pca_results['explained_variance']
                                cumulative_variance = np.cumsum(explained_variance)
                                
                                # Create a DataFrame for the variance information
                                variance_df = pd.DataFrame({
                                    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                                    'Explained Variance': explained_variance,
                                    'Cumulative Variance': cumulative_variance
                                })
                                
                                st.write("Explained Variance by Component:")
                                st.dataframe(variance_df)
                                
                                # Plot explained variance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
                                ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
                                ax.set_xlabel('Principal Components')
                                ax.set_ylabel('Explained Variance Ratio')
                                ax.set_title('Scree Plot')
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                
                                # Display loading vectors
                                if 'components' in pca_results:
                                    st.subheader("Loading Vectors")
                                    components = pca_results['components']
                                    
                                    # Create a DataFrame for the loading vectors
                                    loading_df = pd.DataFrame(components, 
                                                    columns=selected_properties,
                                                    index=[f'PC{i+1}' for i in range(components.shape[0])])
                                    
                                    st.dataframe(loading_df)
                                    
                                    # Plot loading vectors
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    
                                    # For 2D loading plot
                                    if components.shape[0] >= 2:
                                        for i, feature in enumerate(selected_properties):
                                            ax.arrow(0, 0, components[0, i], components[1, i], head_width=0.05, head_length=0.05)
                                            ax.text(components[0, i] * 1.15, components[1, i] * 1.15, feature, fontsize=12)
                                        
                                        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                                        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                                        ax.set_title('PCA Loading Vectors')
                                        ax.grid(True)
                                        
                                        # Set equal scaling for both axes
                                        ax.set_aspect('equal')
                                        
                                        # Add a circle to represent correlation of 1
                                        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray', linestyle='--')
                                        ax.add_patch(circle)
                                        
                                        # Set limits slightly larger than the circle
                                        ax.set_xlim(-1.1, 1.1)
                                        ax.set_ylim(-1.1, 1.1)
                                        
                                        st.pyplot(fig)
                                
                                # Display scatter plot of samples in PC space
                                if 'transformed_data' in pca_results:
                                    st.subheader("PCA Scatter Plot")
                                    
                                    # Get transformed data
                                    transformed_data = pca_results['transformed_data']
                                    
                                    # Create a DataFrame with the transformed data
                                    transformed_df = pd.DataFrame(transformed_data, 
                                                         columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])])
                                    
                                    # Add original data for reference
                                    for col in combinations.columns:
                                        if col in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group']:
                                            transformed_df[col] = combinations[col].values
                                    
                                    # 2D scatter plot
                                    if transformed_data.shape[1] >= 2:
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        
                                        if color_by == "None":
                                            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.7)
                                        else:
                                            # Get unique values for the selected fragment type
                                            unique_fragments = combinations[color_by].unique()
                                            
                                            # Create a color map
                                            cmap = plt.cm.get_cmap('tab10', len(unique_fragments))
                                            
                                            # Plot each fragment type with a different color
                                            for i, fragment in enumerate(unique_fragments):
                                                mask = combinations[color_by] == fragment
                                                ax.scatter(transformed_data[mask, 0], transformed_data[mask, 1], 
                                                  alpha=0.7, label=fragment, color=cmap(i))
                                            
                                            ax.legend(title=color_by.replace('_', ' ').title())
                                        
                                        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                                        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                                        ax.set_title('PCA Scatter Plot')
                                        ax.grid(True)
                                        
                                        st.pyplot(fig)
                                    
                                    # 3D scatter plot if we have at least 3 components
                                    if transformed_data.shape[1] >= 3:
                                        st.subheader("3D PCA Scatter Plot")
                                        
                                        fig = plt.figure(figsize=(10, 8))
                                        ax = fig.add_subplot(111, projection='3d')
                                        
                                        if color_by == "None":
                                            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], alpha=0.7)
                                        else:
                                            # Get unique values for the selected fragment type
                                            unique_fragments = combinations[color_by].unique()
                                            
                                            # Create a color map
                                            cmap = plt.cm.get_cmap('tab10', len(unique_fragments))
                                            
                                            # Plot each fragment type with a different color
                                            for i, fragment in enumerate(unique_fragments):
                                                mask = combinations[color_by] == fragment
                                                ax.scatter(transformed_data[mask, 0], transformed_data[mask, 1], transformed_data[mask, 2],
                                                  alpha=0.7, label=fragment, color=cmap(i))
                                            
                                            ax.legend(title=color_by.replace('_', ' ').title())
                                        
                                        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                                        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                                        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
                                        ax.set_title('3D PCA Scatter Plot')
                                        
                                        st.pyplot(fig)
                                
                                # Provide download link for transformed data
                                csv = transformed_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="pca_results.csv">Download PCA Results</a>'
                                st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error running PCA analysis: {e}")
                        finally:
                            pass
        
        with analysis_tab5:
            st.header("Cluster Analysis")
            st.write("Analyze ionic liquids by clustering them based on fragment similarity.")
            
            # Check if data is loaded
            if 'combinations' not in st.session_state or not st.session_state.combinations:
                st.warning("Please load data first.")
                return
            
            # Get available fragment types
            fragment_types = [col for col in ['cation', 'anion', 'alkyl_chain', 'functional_group'] 
                             if col in pd.DataFrame(st.session_state.combinations).columns]
            
            if not fragment_types:
                st.error("No fragment type columns found in the data. Cannot perform cluster analysis.")
                return
            
            # Get available numeric properties
            df = pd.DataFrame(st.session_state.combinations)
            numeric_properties = [col for col in df.columns 
                                 if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score']
                                 and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_properties:
                st.error("No numeric properties found in the data. Cannot perform cluster analysis.")
                return
            
            # User inputs
            col1, col2 = st.columns(2)
            
            with col1:
                selected_fragments = st.multiselect(
                    "Select fragment types to include in analysis",
                    options=fragment_types,
                    default=fragment_types
                )
                
                if not selected_fragments:
                    st.error("Please select at least one fragment type.")
                    return
                
                cluster_method = st.selectbox(
                    "Clustering method",
                    options=["K-means", "Hierarchical", "DBSCAN", "Find optimal k"]
                )
                
                use_pca = st.checkbox("Use PCA for dimensionality reduction", value=True)
                
            with col2:
                selected_properties = st.multiselect(
                    "Select properties to include in analysis (optional)",
                    options=numeric_properties,
                    default=None
                )
                
                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=3,
                    disabled=(cluster_method == "DBSCAN" or cluster_method == "Find optimal k")
                )
                
                if use_pca:
                    n_components = st.slider(
                        "Number of PCA components",
                        min_value=2,
                        max_value=min(10, len(selected_fragments) * 5),
                        value=2
                    )
            
            # Additional parameters for specific methods
            if cluster_method == "Hierarchical":
                linkage = st.selectbox(
                    "Linkage method",
                    options=["ward", "complete", "average", "single"],
                    index=0
                )
            
            if cluster_method == "DBSCAN":
                col1, col2 = st.columns(2)
                with col1:
                    eps = st.slider(
                        "Epsilon (maximum distance between points)",
                        min_value=0.1,
                        max_value=2.0,
                        value=0.5,
                        step=0.1
                    )
                with col2:
                    min_samples = st.slider(
                        "Minimum samples in neighborhood",
                        min_value=2,
                        max_value=20,
                        value=5
                    )
            
            if cluster_method == "Find optimal k":
                max_k = st.slider(
                    "Maximum number of clusters to try",
                    min_value=2,
                    max_value=15,
                    value=10
                )
            
            # Run button
            if st.button("Run Cluster Analysis"):
                with st.spinner("Running cluster analysis..."):
                    try:
                        # Initialize cluster analyzer
                        cluster_analyzer = ClusterAnalyzer(st.session_state.combinations)
                        
                        # Run selected clustering method
                        if cluster_method == "K-means":
                            results = cluster_analyzer.run_kmeans_analysis(
                                n_clusters=n_clusters,
                                property_names=selected_properties if selected_properties else None,
                                fragment_types=selected_fragments,
                                use_pca=use_pca,
                                n_components=n_components if use_pca else 2
                            )
                        elif cluster_method == "Hierarchical":
                            results = cluster_analyzer.run_hierarchical_analysis(
                                n_clusters=n_clusters,
                                property_names=selected_properties if selected_properties else None,
                                fragment_types=selected_fragments,
                                linkage=linkage,
                                use_pca=use_pca,
                                n_components=n_components if use_pca else 2
                            )
                        elif cluster_method == "DBSCAN":
                            results = cluster_analyzer.run_dbscan_analysis(
                                eps=eps,
                                min_samples=min_samples,
                                property_names=selected_properties if selected_properties else None,
                                fragment_types=selected_fragments,
                                use_pca=use_pca,
                                n_components=n_components if use_pca else 2
                            )
                        elif cluster_method == "Find optimal k":
                            results = cluster_analyzer.determine_optimal_k(
                                max_k=max_k,
                                property_names=selected_properties if selected_properties else None,
                                fragment_types=selected_fragments,
                                use_pca=use_pca,
                                n_components=n_components if use_pca else 2
                            )
                        
                        # Check if there was an error
                        if isinstance(results, dict) and 'error' in results:
                            st.error(f"Error during cluster analysis: {results['error']}")
                            return
                        
                        # Display results
                        if cluster_method == "Find optimal k":
                            # Plot elbow and silhouette curves
                            fig = plt.figure(figsize=(12, 6))
                            
                            # Plot inertia (elbow method)
                            ax1 = fig.add_subplot(121)
                            ax1.plot(results['k_values'], results['inertia'], 'o-', color='blue')
                            ax1.set_xlabel('Number of Clusters (k)')
                            ax1.set_ylabel('Inertia')
                            ax1.set_title('Elbow Method')
                            ax1.grid(True)
                            
                            # Plot silhouette scores
                            ax2 = fig.add_subplot(122)
                            ax2.plot(results['k_values'][1:], results['silhouette'][1:], 'o-', color='green')
                            ax2.set_xlabel('Number of Clusters (k)')
                            ax2.set_ylabel('Silhouette Score')
                            ax2.set_title('Silhouette Method')
                            ax2.grid(True)
                            
                            st.pyplot(fig)
                            
                            st.info(f"Optimal number of clusters based on elbow method: {results['optimal_k']}")
                        else:
                            # Get cluster labels and reduced data
                            cluster_labels = results['cluster_labels']
                            reduced_data = results['reduced_data']
                            
                            # Display silhouette score if available
                            if results['silhouette_score'] is not None:
                                st.info(f"Silhouette score: {results['silhouette_score']:.3f}")
                            
                            # Plot clusters
                            if use_pca and n_components >= 2:
                                # 2D or 3D plot based on number of components
                                if n_components >= 3:
                                    fig = plt.figure(figsize=(10, 8))
                                    ax = fig.add_subplot(111, projection='3d')
                                    
                                    # Get unique labels
                                    unique_labels = np.unique(cluster_labels)
                                    
                                    # Create a colormap
                                    cmap = plt.cm.get_cmap('viridis', len(unique_labels))
                                    
                                    # Plot each cluster
                                    for i, label in enumerate(unique_labels):
                                        mask = cluster_labels == label
                                        if label == -1:
                                            # Noise points in DBSCAN
                                            ax.scatter(
                                                reduced_data[mask, 0],
                                                reduced_data[mask, 1],
                                                reduced_data[mask, 2],
                                                c='black',
                                                marker='x',
                                                label='Noise'
                                            )
                                        else:
                                            ax.scatter(
                                                reduced_data[mask, 0],
                                                reduced_data[mask, 1],
                                                reduced_data[mask, 2],
                                                c=[cmap(i)],
                                                marker='o',
                                                label=f'Cluster {label}'
                                            )
                                    
                                    ax.set_xlabel('PC1')
                                    ax.set_ylabel('PC2')
                                    ax.set_zlabel('PC3')
                                    ax.set_title(f'Clusters from {cluster_method}')
                                    ax.legend()
                                    
                                    st.pyplot(fig)
                                else:
                                    # 2D plot
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    # Get unique labels
                                    unique_labels = np.unique(cluster_labels)
                                    
                                    # Create a colormap
                                    cmap = plt.cm.get_cmap('viridis', len(unique_labels))
                                    
                                    # Plot each cluster
                                    for i, label in enumerate(unique_labels):
                                        mask = cluster_labels == label
                                        if label == -1:
                                            # Noise points in DBSCAN
                                            ax.scatter(
                                                reduced_data[mask, 0],
                                                reduced_data[mask, 1],
                                                c='black',
                                                marker='x',
                                                label='Noise'
                                            )
                                        else:
                                            ax.scatter(
                                                reduced_data[mask, 0],
                                                reduced_data[mask, 1],
                                                c=[cmap(i)],
                                                marker='o',
                                                label=f'Cluster {label}'
                                            )
                                    
                                    ax.set_xlabel('PC1')
                                    ax.set_ylabel('PC2')
                                    ax.set_title(f'Clusters from {cluster_method}')
                                    ax.legend()
                                    
                                    st.pyplot(fig)
                            
                            # Show cluster statistics
                            st.subheader("Cluster Statistics")
                            
                            # Add cluster labels to the data
                            df = pd.DataFrame(st.session_state.combinations)
                            df['cluster'] = cluster_labels
                            
                            # Count number of ILs in each cluster
                            cluster_counts = df['cluster'].value_counts().reset_index()
                            cluster_counts.columns = ['Cluster', 'Count']
                            
                            # Show cluster counts
                            st.write("Number of ionic liquids in each cluster:")
                            st.dataframe(cluster_counts)
                            
                            # Analyze fragment composition in each cluster
                            st.write("Fragment composition in each cluster:")
                            
                            for fragment_type in selected_fragments:
                                if fragment_type in df.columns:
                                    st.write(f"**{fragment_type.capitalize()} distribution:**")
                                    
                                    # Get fragment counts per cluster
                                    fragment_counts = df.groupby(['cluster', fragment_type]).size().unstack().fillna(0)
                                    
                                    # Convert to percentages
                                    fragment_pcts = fragment_counts.div(fragment_counts.sum(axis=1), axis=0) * 100
                                    
                                    # Display as a heatmap
                                    fig, ax = plt.subplots(figsize=(12, max(6, len(fragment_pcts.columns) * 0.4)))
                                    sns.heatmap(fragment_pcts, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
                                    ax.set_title(f'{fragment_type.capitalize()} Distribution (%) by Cluster')
                                    ax.set_ylabel('Cluster')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            
                            # Analyze property distributions in each cluster
                            if selected_properties:
                                st.write("**Property distributions by cluster:**")
                                
                                # Create a long-format dataframe for property analysis
                                property_data = df[['cluster'] + selected_properties].melt(
                                    id_vars=['cluster'],
                                    value_vars=selected_properties,
                                    var_name='Property',
                                    value_name='Value'
                                )
                                
                                # Plot boxplots for each property by cluster
                                fig = plt.figure(figsize=(12, max(6, len(selected_properties) * 1.5)))
                                
                                for i, prop in enumerate(selected_properties):
                                    ax = fig.add_subplot(len(selected_properties), 1, i+1)
                                    sns.boxplot(x='cluster', y=prop, data=df, ax=ax)
                                    ax.set_title(f'{prop} by Cluster')
                                    ax.set_xlabel('Cluster')
                                    ax.set_ylabel(prop)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error during cluster analysis: {str(e)}")
                        st.exception(e)
            
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    run_analysis()
