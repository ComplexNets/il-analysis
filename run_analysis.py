import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64

# Import analysis modules
from analysis.lr_mc_analysis import IonicLiquidAnalysis, FragmentAnalyzer, MonteCarloAnalyzer

def run_analysis():
    """Main function to run the analysis page"""
    st.set_page_config(page_title="Ionic Liquid Analysis", layout="wide")
    
    st.title("Ionic Liquid Property Analysis")
    
    # Upload data
    st.write("Upload the exported data from the Ionic Liquid Optimizer")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        combinations = pd.read_csv(uploaded_file)
        
        # Convert to list of dictionaries for compatibility with analysis code
        combinations_list = combinations.to_dict('records')
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Fragment Influence Analysis", "Monte Carlo Uncertainty Analysis"])
        
        with tab1:
            st.header("Fragment Influence Analysis")
            st.write("Analyze how different fragments influence ionic liquid properties using linear regression.")
            
            # Select property for analysis
            property_options = [col for col in combinations.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score']]
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
        
        with tab2:
            st.header("Monte Carlo Uncertainty Analysis")
            st.write("Analyze uncertainty in property predictions using Monte Carlo simulation.")
            
            # Select property for analysis
            property_options = [col for col in combinations.columns if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score']]
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
                    mc_analyzer = MonteCarloAnalyzer(combinations_list)
                    
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
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    run_analysis()
