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

# Import analysis modules
try:
    from analysis.lr_mc_analysis import IonicLiquidAnalysis, FragmentAnalyzer, MonteCarloAnalyzer, PCAAnalyzer, ClusterAnalyzer, StatisticsAnalyzer
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
    StatisticsAnalyzer = lr_mc_analysis.StatisticsAnalyzer

def run_advanced_analysis():
    """Main function to run the analysis page"""
    st.title("Ionic Liquid Property Analysis")
    
    # Check if combinations data exists in session state
    if 'combinations' not in st.session_state or not st.session_state.combinations:
        st.warning("No ionic liquid data available for analysis. Please run property calculations first or upload a csv file.")
        return
    
    # Get combinations data from session state
    combinations = st.session_state.combinations
    
    # Convert combinations to DataFrame for analysis
    df = pd.DataFrame(combinations)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Fragment Influence Analysis", 
        "Monte Carlo Uncertainty Analysis", 
        "Property Correlation Analysis",
        "Principal Component Analysis",
        "Cluster Analysis",
        "Property Statistics"
    ])
    
    with tab1:
        st.header("Fragment Influence Analysis")
        st.write("Analyze how different fragments influence ionic liquid properties using linear regression.")
        
        # Select property for analysis
        property_options = [col for col in df.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score']]
        selected_property = st.selectbox("Select property to analyze:", property_options)
        
        if st.button("Run Fragment Analysis"):
            with st.spinner("Running fragment analysis..."):
                # Create fragment analyzer
                fragment_analyzer = FragmentAnalyzer(combinations)
                
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
    
    with tab2:
        st.header("Monte Carlo Uncertainty Analysis")
        st.write("Analyze uncertainty in property predictions using Monte Carlo simulation.")
        
        # Select property for analysis
        property_options = [col for col in df.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score']]
        selected_property = st.selectbox("Select property for Monte Carlo analysis:", property_options, key="mc_property")
        
        # Set simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            num_simulations = st.slider("Number of simulations:", min_value=100, max_value=2000, value=500, step=100)
        with col2:
            uncertainty_percentage = st.slider("Uncertainty percentage:", min_value=1, max_value=30, value=10, step=1)
        
        if st.button("Run Monte Carlo Analysis"):
            with st.spinner("Running Monte Carlo simulation..."):
                # Create Monte Carlo analyzer
                mc_analyzer = MonteCarloAnalyzer(combinations)
                
                # Run analysis
                simulation_results = mc_analyzer.run_analysis(
                    selected_property, 
                    num_simulations=num_simulations,
                    uncertainty_percentage=uncertainty_percentage
                )
                
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
    
    with tab3:
        st.header("Property Correlation Analysis")
        st.write("Analyze correlations between different ionic liquid properties.")
        
        # Select properties for correlation analysis
        st.subheader("Select Properties")
        property_options = [col for col in df.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score'] and pd.api.types.is_numeric_dtype(df[col])]
        
        col1, col2 = st.columns(2)
        with col1:
            selected_properties = st.multiselect(
                "Select properties for correlation analysis:",
                property_options,
                default=property_options[:min(4, len(property_options))]
            )
        
        with col2:
            correlation_method = st.selectbox(
                "Correlation method:",
                ["pearson", "spearman", "kendall"],
                index=0,
                help="Pearson: linear correlation, Spearman: rank correlation, Kendall: ordinal correlation"
            )
        
        if st.button("Run Correlation Analysis"):
            if len(selected_properties) < 2:
                st.warning("Please select at least two properties for correlation analysis.")
            else:
                with st.spinner("Calculating correlations..."):
                    # Calculate correlation matrix
                    correlation_df = df[selected_properties].corr(method=correlation_method)
                    
                    # Display correlation matrix
                    st.subheader("Correlation Matrix")
                    st.dataframe(correlation_df.style.background_gradient(cmap='coolwarm', axis=None))
                    
                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    mask = np.triu(np.ones_like(correlation_df, dtype=bool))
                    sns.heatmap(
                        correlation_df, 
                        mask=mask, 
                        annot=True, 
                        fmt=".2f", 
                        cmap="coolwarm", 
                        square=True, 
                        linewidths=0.5, 
                        ax=ax
                    )
                    ax.set_title(f"{correlation_method.capitalize()} Correlation Matrix")
                    
                    # Display heatmap
                    st.pyplot(fig)
                    
                    # Create pairplot for selected properties
                    if len(selected_properties) <= 6:  # Limit to avoid excessive computation
                        st.subheader("Pairwise Relationships")
                        fig = sns.pairplot(df[selected_properties], diag_kind="kde", height=2.5)
                        fig.fig.suptitle("Pairwise Relationships Between Properties", y=1.02)
                        st.pyplot(fig.fig)
                    
                    # Provide download link for correlation matrix
                    csv = correlation_df.to_csv()
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="correlation_matrix.csv">Download Correlation Matrix</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    with tab4:
        st.header("Principal Component Analysis (PCA)")
        st.write("Analyze the structure and patterns in ionic liquid data using dimensionality reduction.")
        
        # Select fragment types for analysis
        st.subheader("Select Fragments to Include")
        fragment_cols = [col for col in ['cation', 'anion', 'alkyl_chain', 'functional_group'] if col in df.columns]
        
        # Create checkboxes for fragment selection
        selected_fragments = []
        col1, col2 = st.columns(2)
        
        with col1:
            for i in range(0, len(fragment_cols), 2):
                if i < len(fragment_cols):
                    if st.checkbox(f"Include {fragment_cols[i]}", value=True, key=f"frag_{fragment_cols[i]}"):
                        selected_fragments.append(fragment_cols[i])
        
        with col2:
            for i in range(1, len(fragment_cols), 2):
                if i < len(fragment_cols):
                    if st.checkbox(f"Include {fragment_cols[i]}", value=True, key=f"frag_{fragment_cols[i]}"):
                        selected_fragments.append(fragment_cols[i])
        
        # PCA parameters
        st.subheader("PCA Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Number of components:", min_value=2, max_value=10, value=3, step=1)
        with col2:
            scale_data = st.checkbox("Standardize data", value=True)
        
        # Property for coloring points
        st.subheader("Visualization Options")
        property_options = ["None"] + [col for col in df.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score'] and pd.api.types.is_numeric_dtype(df[col])]
        color_by = st.selectbox("Color points by property:", property_options)
        
        # Convert "None" to None for the PCA function
        color_by = None if color_by == "None" else color_by
        
        if st.button("Run PCA Analysis"):
            if not selected_fragments:
                st.warning("Please select at least one fragment type for analysis.")
            else:
                with st.spinner("Running PCA analysis..."):
                    # Create PCA analyzer
                    pca_analyzer = PCAAnalyzer(combinations)
                    
                    # Run analysis
                    results = pca_analyzer.run_analysis(
                        n_components=n_components,
                        fragment_types=selected_fragments,
                        scale=scale_data
                    )
                    
                    # Display results
                    st.subheader("PCA Results")
                    
                    # Display explained variance
                    st.write("### Explained Variance")
                    
                    # Create explained variance table
                    explained_var_df = pd.DataFrame({
                        'Principal Component': [f"PC{i+1}" for i in range(n_components)],
                        'Explained Variance (%)': [f"{var:.2%}" for var in results['explained_variance']],
                        'Cumulative Variance (%)': [f"{var:.2%}" for var in results['cumulative_variance']]
                    })
                    st.dataframe(explained_var_df)
                    
                    # Plot explained variance
                    st.write("### Explained Variance Plot")
                    fig_var, _ = pca_analyzer.plot_explained_variance()
                    st.pyplot(fig_var)
                    
                    # Plot 2D scatter
                    st.write("### PCA Scatter Plot (2D)")
                    fig_scatter, _ = pca_analyzer.plot_pca_scatter(color_by=color_by)
                    st.pyplot(fig_scatter)
                    
                    # Plot loading vectors
                    st.write("### Feature Contribution Plot")
                    fig_loadings, _ = pca_analyzer.plot_loading_vectors(n_features=10)
                    st.pyplot(fig_loadings)
                    
                    # If we have at least 3 components, show 3D plot
                    if n_components >= 3:
                        st.write("### PCA Scatter Plot (3D)")
                        fig_3d, _ = pca_analyzer.plot_3d_scatter(color_by=color_by)
                        st.pyplot(fig_3d)
                    
                    # Display top features for each component
                    st.write("### Top Features by Component")
                    top_features = pca_analyzer.get_top_features(n_features=10)
                    
                    for i in range(min(3, n_components)):  # Show top 3 components at most
                        st.write(f"#### Principal Component {i+1} (Explains {results['explained_variance'][i]:.2%} of variance)")
                        
                        # Create dataframe of top features
                        features_df = pd.DataFrame(
                            top_features[i], 
                            columns=['Feature', 'Loading']
                        )
                        
                        # Sort by absolute loading value
                        features_df['Abs Loading'] = features_df['Loading'].abs()
                        features_df = features_df.sort_values('Abs Loading', ascending=False).drop('Abs Loading', axis=1)
                        
                        st.dataframe(features_df)
                    
                    # Provide download link for transformed data
                    transformed_df = pd.DataFrame(
                        results['transformed_data'],
                        columns=[f"PC{i+1}" for i in range(n_components)]
                    )
                    
                    # Add original data for reference
                    for col in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group']:
                        if col in df.columns:
                            transformed_df[col] = df[col].values
                    
                    csv = transformed_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="pca_results.csv">Download PCA Results</a>'
                    st.markdown(href, unsafe_allow_html=True)

    with tab5:
        st.header("Cluster Analysis")
        st.write("Group ionic liquids based on similarity of their features using clustering algorithms.")
        
        # Select fragment types for clustering
        st.subheader("Select Fragments to Include")
        fragment_cols = [col for col in ['cation', 'anion', 'alkyl_chain', 'functional_group'] if col in df.columns]
        
        # Create checkboxes for fragment selection
        selected_fragments = []
        col1, col2 = st.columns(2)
        
        with col1:
            for i in range(0, len(fragment_cols), 2):
                if i < len(fragment_cols):
                    if st.checkbox(f"Include {fragment_cols[i]}", value=True, key=f"cluster_frag_{fragment_cols[i]}"):
                        selected_fragments.append(fragment_cols[i])
        
        with col2:
            for i in range(1, len(fragment_cols), 2):
                if i < len(fragment_cols):
                    if st.checkbox(f"Include {fragment_cols[i]}", value=True, key=f"cluster_frag_{fragment_cols[i]}"):
                        selected_fragments.append(fragment_cols[i])
        
        # Clustering parameters
        st.subheader("Clustering Parameters")
        
        # Select clustering algorithm
        clustering_algorithm = st.selectbox(
            "Clustering Algorithm:",
            ["K-means", "Hierarchical", "DBSCAN"],
            index=0,
            key="clustering_algorithm"
        )
        
        # Parameters based on selected algorithm
        if clustering_algorithm == "K-means":
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3, step=1, key="kmeans_n_clusters")
            with col2:
                use_pca = st.checkbox("Use PCA for dimensionality reduction", value=True, key="kmeans_use_pca")
                
            if use_pca:
                n_components = st.slider("Number of PCA components:", min_value=2, max_value=10, value=3, step=1, key="kmeans_n_components")
            else:
                n_components = 2  # Default value
                
            # Option to find optimal number of clusters
            find_optimal_k = st.checkbox("Find optimal number of clusters", value=False, key="find_optimal_k")
            if find_optimal_k:
                max_k = st.slider("Maximum number of clusters to try:", min_value=2, max_value=15, value=10, step=1, key="max_k")
        
        elif clustering_algorithm == "Hierarchical":
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3, step=1, key="hierarchical_n_clusters")
            with col2:
                linkage = st.selectbox(
                    "Linkage method:",
                    ["ward", "complete", "average", "single"],
                    index=0,
                    key="linkage"
                )
                
            use_pca = st.checkbox("Use PCA for dimensionality reduction", value=True, key="hierarchical_use_pca")
            if use_pca:
                n_components = st.slider("Number of PCA components:", min_value=2, max_value=10, value=3, step=1, key="hierarchical_n_components")
            else:
                n_components = 2  # Default value
        
        elif clustering_algorithm == "DBSCAN":
            col1, col2 = st.columns(2)
            with col1:
                eps = st.slider("Epsilon (neighborhood size):", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="eps")
            with col2:
                min_samples = st.slider("Minimum samples in neighborhood:", min_value=2, max_value=10, value=5, step=1, key="min_samples")
                
            use_pca = st.checkbox("Use PCA for dimensionality reduction", value=True, key="dbscan_use_pca")
            if use_pca:
                n_components = st.slider("Number of PCA components:", min_value=2, max_value=10, value=3, step=1, key="dbscan_n_components")
            else:
                n_components = 2  # Default value
        
        # Property for coloring points
        st.subheader("Visualization Options")
        property_options = ["None"] + [col for col in df.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score'] and pd.api.types.is_numeric_dtype(df[col])]
        color_by = st.selectbox("Color points by property:", property_options, key="cluster_color_by")
        
        # Convert "None" to None for the clustering function
        color_by = None if color_by == "None" else color_by
        
        # Run clustering button
        if st.button("Run Cluster Analysis"):
            if not selected_fragments:
                st.warning("Please select at least one fragment type for analysis.")
            else:
                with st.spinner("Running cluster analysis..."):
                    # Create cluster analyzer
                    cluster_analyzer = ClusterAnalyzer(combinations)
                    
                    # Run analysis based on selected algorithm
                    if clustering_algorithm == "K-means":
                        if find_optimal_k:
                            # Find optimal number of clusters
                            st.write("### Finding Optimal Number of Clusters")
                            optimal_k_results = cluster_analyzer.determine_optimal_k(
                                max_k=max_k,
                                fragment_types=selected_fragments,
                                use_pca=use_pca,
                                n_components=n_components
                            )
                            
                            # Plot elbow method results
                            fig_elbow, _ = cluster_analyzer.plot_elbow_method(optimal_k_results)
                            st.pyplot(fig_elbow)
                            
                            # Find the best k based on silhouette score
                            best_k_idx = np.argmax(optimal_k_results['silhouette_values'])
                            best_k = optimal_k_results['k_values'][best_k_idx]
                            
                            st.write(f"Optimal number of clusters based on silhouette score: **{best_k}**")
                            
                            # Use the optimal k for clustering
                            n_clusters = best_k
                        
                        # Run K-means clustering
                        results = cluster_analyzer.run_kmeans_analysis(
                            n_clusters=n_clusters,
                            fragment_types=selected_fragments,
                            use_pca=use_pca,
                            n_components=n_components
                        )
                        
                    elif clustering_algorithm == "Hierarchical":
                        # Run hierarchical clustering
                        results = cluster_analyzer.run_hierarchical_analysis(
                            n_clusters=n_clusters,
                            fragment_types=selected_fragments,
                            use_pca=use_pca,
                            n_components=n_components,
                            linkage=linkage
                        )
                        
                    elif clustering_algorithm == "DBSCAN":
                        # Run DBSCAN clustering
                        results = cluster_analyzer.run_dbscan_analysis(
                            eps=eps,
                            min_samples=min_samples,
                            fragment_types=selected_fragments,
                            use_pca=use_pca,
                            n_components=n_components
                        )
                    
                    # Display clustering results
                    st.subheader("Clustering Results")
                    
                    # Display silhouette score if available
                    if results['silhouette_score'] is not None:
                        st.write(f"Silhouette Score: **{results['silhouette_score']:.3f}**")
                        st.write("*Higher silhouette score (closer to 1.0) indicates better-defined clusters.*")
                    
                    # Display number of clusters
                    if clustering_algorithm == "DBSCAN":
                        n_clusters = results.get('n_clusters', 0)
                        st.write(f"Number of clusters found: **{n_clusters}**")
                        
                        # Check if we have noise points
                        if -1 in set(results['cluster_labels']):
                            noise_count = np.sum(results['cluster_labels'] == -1)
                            st.write(f"Number of noise points: **{noise_count}**")
                    else:
                        st.write(f"Number of clusters: **{n_clusters}**")
                    
                    # Plot 2D scatter plot
                    st.write("### Cluster Visualization (2D)")
                    fig_2d, _ = cluster_analyzer.plot_clusters_2d()
                    st.pyplot(fig_2d)
                    
                    # Plot 3D scatter plot if we have at least 3 components
                    if n_components >= 3 and use_pca:
                        st.write("### Cluster Visualization (3D)")
                        fig_3d, _ = cluster_analyzer.plot_clusters_3d()
                        st.pyplot(fig_3d)
                    
                    # Display cluster properties
                    st.write("### Cluster Properties")
                    cluster_properties = cluster_analyzer.get_cluster_properties()
                    st.dataframe(cluster_properties)
                    
                    # Display cluster compositions
                    st.write("### Cluster Compositions")
                    compositions = cluster_analyzer.get_cluster_compositions()
                    
                    for fragment_type, composition_df in compositions.items():
                        st.write(f"#### {fragment_type.capitalize()} Composition")
                        st.dataframe(composition_df)
                    
                    # Add cluster labels to dataframe for download
                    df_with_clusters = df.copy()
                    df_with_clusters['cluster'] = results['cluster_labels']
                    
                    # Provide download link for results
                    csv = df_with_clusters.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="cluster_analysis_results.csv">Download Cluster Analysis Results</a>'
                    st.markdown(href, unsafe_allow_html=True)

    with tab6:
        st.header("Property Statistics")
        st.write("View and analyze statistical distributions of ionic liquid properties.")
        
        # Debug information
        st.subheader("Debug Information")
        df = pd.DataFrame(combinations)
        st.write("DataFrame Columns:", df.columns.tolist())
        
        # Check for hydrophobicity-related columns
        hydrophobicity_columns = [col for col in df.columns if 'hydro' in col.lower() or 'logp' in col.lower() or 'log_p' in col.lower()]
        st.write("Hydrophobicity-related columns:", hydrophobicity_columns)
        
        # Display sample data
        st.write("Sample Data (first 5 rows):")
        st.dataframe(df.head())
        st.markdown("---")
        
        # Create statistics analyzer
        statistics_analyzer = StatisticsAnalyzer(combinations)
        
        # Calculate and display statistics table
        stats_table = statistics_analyzer.create_statistics_table()
        
        if not stats_table.empty:
            st.subheader("Property Statistics Summary")
            st.dataframe(stats_table)
            
            # Provide download link for statistics table
            csv = stats_table.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="property_statistics.csv">Download Statistics Table</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Display distribution plots
            st.subheader("Property Distributions")
            
            # Create tabs for different visualization types
            dist_tab1, dist_tab2 = st.tabs(["Histograms", "Boxplots"])
            
            with dist_tab1:
                # Plot histograms
                fig_hist, error_msg = statistics_analyzer.plot_property_distributions()
                if fig_hist:
                    st.pyplot(fig_hist)
                elif error_msg:
                    st.warning(error_msg)
            
            with dist_tab2:
                # Plot boxplots
                fig_box, error_msg = statistics_analyzer.plot_property_boxplots()
                if fig_box:
                    st.pyplot(fig_box)
                elif error_msg:
                    st.warning(error_msg)
        else:
            st.warning("No property data available for statistical analysis.")

if __name__ == "__main__":
    run_advanced_analysis()
