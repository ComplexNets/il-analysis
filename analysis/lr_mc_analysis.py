"""
Property Analysis Module for Ionic Liquids

This module performs statistical analysis on calculated ionic liquid properties:
1. Fragment-Level Linear Regression: Quantifies the influence of each fragment on properties
2. Monte Carlo Uncertainty Analysis: Assesses uncertainty in property predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import math
import os
import warnings

class IonicLiquidAnalysis:
    def __init__(self, data_file: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the analysis module with either a data file or DataFrame
        
        Args:
            data_file: Path to CSV/Excel file containing ionic liquid data
            data: Pandas DataFrame containing ionic liquid data
        """
        self.data = None
        self.fragment_data = None
        self.regression_results = {}
        self.monte_carlo_results = {}
        
        if data_file:
            self.load_data(data_file)
        elif data is not None:
            self.data = data.copy()
            
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")
    
    def load_data(self, file_path: str) -> None:
        """
        Load data from CSV or Excel file
        
        Args:
            file_path: Path to the data file
        """
        try:
            # Detect file type and load accordingly
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            print(f"Successfully loaded data with {len(self.data)} rows and {len(self.data.columns)} columns")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
    
    def prepare_fragment_data(self, cation_col: str = 'cation', 
                             anion_col: str = 'anion', 
                             alkyl_col: str = 'alkyl_chains',
                             functional_group_col: str = 'functional_group') -> pd.DataFrame:
        """
        Prepare fragment-level data for regression analysis
        
        Args:
            cation_col: Column name for cation data
            anion_col: Column name for anion data
            alkyl_col: Column name for alkyl chain data
            functional_group_col: Column name for functional group data
            
        Returns:
            DataFrame with one-hot encoded fragment data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Extract fragments
        fragments = {}
        
        # Process each fragment type if it exists in the data
        if cation_col in self.data.columns:
            fragments['cation'] = self.data[cation_col]
        
        if anion_col in self.data.columns:
            fragments['anion'] = self.data[anion_col]
        
        if alkyl_col in self.data.columns:
            fragments['alkyl'] = self.data[alkyl_col]
        
        if functional_group_col in self.data.columns:
            # Handle NaN values in functional groups
            self.data[functional_group_col] = self.data[functional_group_col].fillna('None')
            fragments['functional_group'] = self.data[functional_group_col]
        
        # Create one-hot encoding for each fragment type
        encoded_data = {}
        
        for frag_type, values in fragments.items():
            # Create a DataFrame with one column
            temp_df = pd.DataFrame({frag_type: values})
            
            # One-hot encode the column
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(temp_df[[frag_type]])
            
            # Get feature names
            feature_names = [f"{frag_type}_{cat}" for cat in encoder.categories_[0]]
            
            # Create DataFrame with encoded values
            encoded_df = pd.DataFrame(encoded, columns=feature_names)
            
            # Add to the encoded_data dictionary
            encoded_data[frag_type] = encoded_df
        
        # Combine all encoded fragments
        combined_fragments = pd.concat([df for df in encoded_data.values()], axis=1)
        
        # Add any additional columns from the original data (excluding fragment columns)
        exclude_cols = [cation_col, anion_col, alkyl_col, functional_group_col]
        additional_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        if additional_cols:
            combined_fragments = pd.concat([combined_fragments, self.data[additional_cols].reset_index(drop=True)], axis=1)
        
        self.fragment_data = combined_fragments
        print(f"Created fragment data with {len(self.fragment_data.columns)} features")
        
        return self.fragment_data
    
    def run_fragment_regression(self, target_col: str, alpha: float = 0.0, 
                              l1_ratio: float = 0.5, test_size: float = 0.2,
                              normalize: bool = True) -> Dict[str, Any]:
        """
        Run regression analysis to determine fragment influence on properties
        
        Args:
            target_col: Column name of the target property
            alpha: Regularization strength (0 for standard linear regression)
            l1_ratio: Mix of L1/L2 regularization (0 = Ridge, 1 = Lasso)
            test_size: Proportion of data to use for testing
            normalize: Whether to normalize features
            
        Returns:
            Dictionary containing regression results
        """
        if self.fragment_data is None:
            raise ValueError("Fragment data not prepared. Run prepare_fragment_data first.")
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Prepare data
        X = self.fragment_data.drop(columns=[target_col], errors='ignore')
        y = self.data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Set up preprocessing
        preprocessor = StandardScaler() if normalize else None
        
        # Choose model based on alpha
        if alpha == 0:
            model = LinearRegression()
            model_name = "Linear Regression"
        elif l1_ratio == 0:
            model = Ridge(alpha=alpha)
            model_name = f"Ridge Regression (alpha={alpha})"
        elif l1_ratio == 1:
            model = Lasso(alpha=alpha)
            model_name = f"Lasso Regression (alpha={alpha})"
        else:
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model_name = f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})"
        
        # Create pipeline
        if preprocessor:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
        else:
            pipeline = model
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Get coefficients
        if hasattr(pipeline, 'coef_'):
            coefficients = pipeline.coef_
        else:
            coefficients = pipeline.named_steps['model'].coef_
        
        # Create DataFrame of coefficients
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': coefficients
        })
        
        # Sort by absolute coefficient value
        coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('AbsCoefficient', ascending=False).reset_index(drop=True)
        
        # Store results
        results = {
            'model': pipeline,
            'model_name': model_name,
            'target_property': target_col,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'coefficients': coef_df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
        
        # Save results
        self.regression_results[target_col] = results
        
        # Print summary
        print(f"\n=== {model_name} Results for {target_col} ===")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print("\nTop 10 most influential fragments:")
        print(coef_df.head(10)[['Feature', 'Coefficient']])
        
        return results
    
    def plot_fragment_importance(self, target_col: str, top_n: int = 10, 
                               show_plot: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot the most influential fragments for a property
        
        Args:
            target_col: Column name of the target property
            top_n: Number of top fragments to show
            show_plot: Whether to display the plot
            save_path: Path to save the plot (if None, plot is not saved)
        """
        if target_col not in self.regression_results:
            raise ValueError(f"No regression results for '{target_col}'. Run run_fragment_regression first.")
        
        # Get coefficient data
        coef_df = self.regression_results[target_col]['coefficients'].copy()
        
        # Take top N coefficients by absolute value
        top_coef_df = coef_df.head(top_n)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart, sorted by coefficient value (not absolute)
        colors = ['#1f77b4' if c > 0 else '#d62728' for c in top_coef_df['Coefficient']]
        
        ax = sns.barplot(x='Coefficient', y='Feature', data=top_coef_df, palette=colors)
        
        # Add a vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.title(f"Top {top_n} Fragment Contributions to {target_col}", fontsize=16)
        plt.xlabel("Coefficient (Impact on Property)", fontsize=14)
        plt.ylabel("Fragment", fontsize=14)
        
        # Add R² value as text
        train_r2 = self.regression_results[target_col]['train_r2']
        test_r2 = self.regression_results[target_col]['test_r2']
        model_name = self.regression_results[target_col]['model_name']
        
        plt.annotate(f"Model: {model_name}\nTraining R²: {train_r2:.4f}\nTest R²: {test_r2:.4f}",
                   xy=(0.02, 0.02), xycoords='figure fraction', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def monte_carlo_simulation(self, target_col: str, num_simulations: int = 1000,
                              uncertainty_pct: float = 5.0, top_n: int = 10) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to assess uncertainty in property predictions
        
        Args:
            target_col: Column name of the target property
            num_simulations: Number of Monte Carlo simulations to run
            uncertainty_pct: Percentage uncertainty in property values
            top_n: Number of top ILs to track during simulation
            
        Returns:
            Dictionary containing Monte Carlo simulation results
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Get original property values and identify top ILs
        original_values = self.data[target_col].values
        top_indices = np.argsort(original_values)[-top_n:]  # For properties where higher is better
        
        # Store simulation results
        simulation_results = np.zeros((len(original_values), num_simulations))
        
        # Run simulations
        print(f"Running {num_simulations} Monte Carlo simulations for {target_col}...")
        
        for i in range(num_simulations):
            # Add random noise based on uncertainty percentage
            noise_factor = 1 + np.random.uniform(-uncertainty_pct/100, uncertainty_pct/100, size=len(original_values))
            simulated_values = original_values * noise_factor
            simulation_results[:, i] = simulated_values
            
            if (i+1) % 100 == 0:
                print(f"  Completed {i+1} simulations")
        
        # Calculate statistics for each IL
        mean_values = np.mean(simulation_results, axis=1)
        std_values = np.std(simulation_results, axis=1)
        cv_values = std_values / mean_values  # Coefficient of variation
        
        # Calculate confidence intervals (95%)
        lower_ci = np.percentile(simulation_results, 2.5, axis=1)
        upper_ci = np.percentile(simulation_results, 97.5, axis=1)
        
        # Track how often each IL appears in the top N across simulations
        top_n_counts = np.zeros(len(original_values))
        
        for i in range(num_simulations):
            sim_top_indices = np.argsort(simulation_results[:, i])[-top_n:]
            top_n_counts[sim_top_indices] += 1
        
        top_n_frequency = top_n_counts / num_simulations * 100  # Convert to percentage
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Original': original_values,
            'Mean': mean_values,
            'StdDev': std_values,
            'CV': cv_values,
            'LowerCI': lower_ci,
            'UpperCI': upper_ci,
            'TopNFrequency': top_n_frequency
        })
        
        # Add IL identifiers if available
        id_columns = ['Name', 'cation', 'anion', 'alkyl_chains', 'functional_group']
        for col in id_columns:
            if col in self.data.columns:
                results_df[col] = self.data[col].values
        
        # Create dataframe of top ILs by original value
        top_ils = results_df.iloc[top_indices].sort_values('Original', ascending=False).reset_index()
        
        # Store Monte Carlo results
        mc_results = {
            'target_property': target_col,
            'num_simulations': num_simulations,
            'uncertainty_pct': uncertainty_pct,
            'results_df': results_df,
            'top_ils': top_ils,
            'simulation_data': simulation_results,
            'top_indices': top_indices
        }
        
        self.monte_carlo_results[target_col] = mc_results
        
        # Print summary
        print("\n=== Monte Carlo Simulation Results ===")
        print(f"Property: {target_col}")
        print(f"Number of simulations: {num_simulations}")
        print(f"Uncertainty: {uncertainty_pct}%")
        print(f"\nTop {top_n} ILs by original {target_col} value:")
        
        display_cols = ['Original', 'Mean', 'StdDev', 'CV', 'TopNFrequency']
        # Add identifier columns if available
        for col in id_columns:
            if col in top_ils.columns:
                display_cols.append(col)
        
        print(top_ils[display_cols])
        
        return mc_results
    
    def plot_monte_carlo_results(self, target_col: str, plot_type: str = 'boxplot', 
                               num_ils: int = 10, show_plot: bool = True, 
                               save_path: Optional[str] = None) -> None:
        """
        Plot Monte Carlo simulation results
        
        Args:
            target_col: Column name of the target property
            plot_type: Type of plot ('boxplot', 'histogram', 'violin')
            num_ils: Number of ILs to include in the plot
            show_plot: Whether to display the plot
            save_path: Path to save the plot (if None, plot is not saved)
        """
        if target_col not in self.monte_carlo_results:
            raise ValueError(f"No Monte Carlo results for '{target_col}'. Run monte_carlo_simulation first.")
        
        # Get results
        mc_results = self.monte_carlo_results[target_col]
        top_indices = mc_results['top_indices']
        simulation_data = mc_results['simulation_data']
        
        # Get names for ILs if available
        if 'Name' in self.data.columns:
            il_names = self.data['Name'].iloc[top_indices].values
        else:
            # Create names from components if available
            name_components = []
            for col in ['cation', 'anion', 'alkyl_chains']:
                if col in self.data.columns:
                    name_components.append(self.data[col])
            
            if name_components:
                il_names = self.data.apply(
                    lambda row: ' '.join(str(row[col]) for col in ['cation', 'anion', 'alkyl_chains'] 
                                     if col in self.data.columns), 
                    axis=1
                ).iloc[top_indices].values
            else:
                # Default to IL indices
                il_names = [f"IL {i}" for i in top_indices]
        
        # Take top N ILs for plotting
        plot_indices = top_indices[-num_ils:]
        plot_names = il_names[-num_ils:]
        plot_data = simulation_data[plot_indices, :]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'boxplot':
            # Transpose data for boxplot
            plt.boxplot(plot_data.T, vert=False, labels=plot_names)
            plt.xlabel(target_col)
            plt.ylabel('Ionic Liquid')
            
        elif plot_type == 'violin':
            # Transpose data for violin plot
            plt.violinplot(plot_data.T, vert=False, showmeans=True, showmedians=True)
            plt.yticks(range(1, len(plot_names) + 1), plot_names)
            plt.xlabel(target_col)
            plt.ylabel('Ionic Liquid')
            
        elif plot_type == 'histogram':
            # Create subplots for histograms
            fig, axes = plt.subplots(num_ils, 1, figsize=(10, 2*num_ils), sharex=True)
            
            for i, (name, data) in enumerate(zip(plot_names, plot_data)):
                axes[i].hist(data, bins=30, alpha=0.7, color='skyblue')
                axes[i].axvline(np.mean(data), color='red', linestyle='--')
                axes[i].set_ylabel(name)
                
            axes[-1].set_xlabel(target_col)
            plt.tight_layout()
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        # Add title
        plt.title(f"Monte Carlo Simulation Results for {target_col}\n"
                f"({mc_results['num_simulations']} simulations, {mc_results['uncertainty_pct']}% uncertainty)")
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_property_distribution(self, target_col: str, bins: int = 30, 
                                 show_plot: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of a property across all ionic liquids
        
        Args:
            target_col: Column name of the target property
            bins: Number of bins for histogram
            show_plot: Whether to display the plot
            save_path: Path to save the plot (if None, plot is not saved)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot histogram with kernel density estimate
        sns.histplot(self.data[target_col], bins=bins, kde=True)
        
        # Add title and labels
        plt.title(f"Distribution of {target_col} across {len(self.data)} Ionic Liquids", fontsize=16)
        plt.xlabel(target_col, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        
        # Add statistics
        mean_val = self.data[target_col].mean()
        median_val = self.data[target_col].median()
        std_val = self.data[target_col].std()
        
        stats_text = f"Mean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd Dev: {std_val:.4f}"
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                   verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def run_complete_analysis(self, target_cols: List[str], alpha: float = 0.0,
                            uncertainty_pct: float = 5.0, num_simulations: int = 1000,
                            output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run complete analysis pipeline including regression and Monte Carlo
        
        Args:
            target_cols: List of property columns to analyze
            alpha: Regularization strength for regression
            uncertainty_pct: Percentage uncertainty for Monte Carlo
            num_simulations: Number of Monte Carlo simulations
            output_dir: Directory to save plots (if None, plots are not saved)
            
        Returns:
            Dictionary of results for each target property
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        # Prepare fragment data if needed
        if self.fragment_data is None:
            self.prepare_fragment_data()
            
        # Store all results
        all_results = {}
            
        # Analyze each target property
        for target_col in target_cols:
            print(f"\n{'='*50}")
            print(f"Analyzing property: {target_col}")
            print(f"{'='*50}")
            
            # Run regression
            print("\n--- Fragment-Level Regression Analysis ---")
            reg_results = self.run_fragment_regression(target_col, alpha=alpha)
            
            # Plot fragment importance
            save_path = os.path.join(output_dir, f"{target_col}_fragment_importance.png") if output_dir else None
            self.plot_fragment_importance(target_col, show_plot=True, save_path=save_path)
            
            # Run Monte Carlo simulation
            print("\n--- Monte Carlo Uncertainty Analysis ---")
            mc_results = self.monte_carlo_simulation(target_col, 
                                                  num_simulations=num_simulations,
                                                  uncertainty_pct=uncertainty_pct)
            
            # Plot Monte Carlo results
            save_path = os.path.join(output_dir, f"{target_col}_monte_carlo.png") if output_dir else None
            self.plot_monte_carlo_results(target_col, plot_type='boxplot', show_plot=True, save_path=save_path)
            
            # Plot property distribution
            save_path = os.path.join(output_dir, f"{target_col}_distribution.png") if output_dir else None
            self.plot_property_distribution(target_col, show_plot=True, save_path=save_path)
            
            # Store combined results
            all_results[target_col] = {
                'regression': reg_results,
                'monte_carlo': mc_results
            }
            
        return all_results

class FragmentAnalyzer:
    """Class for analyzing fragment influence on ionic liquid properties"""
    
    def __init__(self, combinations):
        """Initialize with combinations data from Excel"""
        self.combinations = combinations
        self.df = pd.DataFrame(combinations)
        
    def analyze_property(self, property_name):
        """Analyze the influence of fragments on a specific property"""
        print(f"Analyzing {property_name} with {len(self.combinations)} combinations")
        
        # Extract fragment columns
        fragment_cols = ['cation', 'anion', 'alkyl_chain']
        if 'functional_group' in self.df.columns:
            fragment_cols.append('functional_group')
            
        # One-hot encode fragments
        encoded_fragments = {}
        for col in fragment_cols:
            if col in self.df.columns:
                # Fill NA values
                self.df[col] = self.df[col].fillna('None')
                
                # One-hot encode
                dummies = pd.get_dummies(self.df[col], prefix=col)
                encoded_fragments[col] = dummies
        
        # Combine all encoded fragments
        X = pd.concat(encoded_fragments.values(), axis=1)
        
        # Get target property
        y = self.df[property_name]
        
        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficients
        coefficients = pd.DataFrame({
            'Fragment': X.columns,
            'Influence': model.coef_
        }).sort_values('Influence', ascending=False)
        
        return coefficients


class MonteCarloAnalyzer:
    """Class for Monte Carlo uncertainty analysis of ionic liquid properties"""
    
    def __init__(self, combinations):
        """Initialize with combinations data from Excel"""
        self.combinations = combinations
        self.df = pd.DataFrame(combinations)
        
    def run_analysis(self, property_name, uncertainty_percentage=10, n_simulations=1000):
        """
        Run Monte Carlo analysis for a given property
        
        Args:
            property_name: Name of the property to analyze
            uncertainty_percentage: Percentage uncertainty in the property values
            n_simulations: Number of Monte Carlo simulations to run
            
        Returns:
            Array of simulated values or dictionary with error message
        """
        # Create DataFrame from list of dictionaries
        if not hasattr(self, 'df') or self.df is None:
            self.df = pd.DataFrame(self.combinations)
        
        # Check if property exists
        if property_name not in self.df.columns:
            return {
                'error': f"Property '{property_name}' not found in data"
            }
        
        # Get property values and ensure they are numeric
        try:
            # First check if the column is already numeric
            if not np.issubdtype(self.df[property_name].dtype, np.number):
                return {
                    'error': f"Property '{property_name}' contains non-numeric values and cannot be used for Monte Carlo analysis"
                }
            
            # Get numeric values
            values = self.df[property_name].copy()
            
            # Drop NaN values
            values = values.dropna()
        except Exception as e:
            return {
                'error': f"Error processing property values: {e}"
            }
        
        if len(values) == 0:
            return {
                'error': "No valid numeric values found for this property"
            }
        
        # Calculate standard deviation based on uncertainty percentage
        std_dev = values.std() * (uncertainty_percentage / 100)
        
        # Run simulations
        all_simulated_values = []
        for _ in range(n_simulations):
            # Add random noise to each value
            noise = np.random.normal(0, std_dev, size=len(values))
            simulated_values = values + noise
            # Store mean of simulated values
            all_simulated_values.append(np.mean(simulated_values))
        
        return np.array(all_simulated_values)

class StatisticsAnalyzer:
    """Class for analyzing and visualizing basic statistics of ionic liquid properties"""
    
    def __init__(self, combinations):
        """Initialize with combinations data"""
        self.combinations = combinations
        self.df = pd.DataFrame(combinations)
        self.property_mapping = {
            'Density (kg/m³)': {'unit': 'kg/m³', 'short_name': 'density'},
            'Heat Capacity (J/mol·K)': {'unit': 'J/mol·K', 'short_name': 'heat_capacity'},
            'Toxicity (IC50 in mM)': {'unit': 'mM', 'short_name': 'toxicity'},
            'Solubility (g/L)': {'unit': 'g/L', 'short_name': 'solubility'},
            'Hydrophobicity (logP)': {'unit': 'logP', 'short_name': 'hydrophobicity'},
            'Viscosity (Pa·s)': {'unit': 'Pa·s', 'short_name': 'viscosity'}
        }
        # Alternative column names that might be in the data
        self.alt_property_names = {
            'density': ['Density (kg/m³)', 'Density', 'density'],
            'heat_capacity': ['Heat Capacity (J/mol·K)', 'Heat Capacity', 'heat_capacity'],
            'toxicity': ['Toxicity (IC50 in mM)', 'Toxicity', 'toxicity'],
            'solubility': ['Solubility (g/L)', 'Solubility', 'solubility'],
            'hydrophobicity': ['Hydrophobicity (logP)', 'Hydrophobicity', 'LogP', 'hydrophobicity', 'log_p', 'logP'],
            'viscosity': ['Viscosity (Pa·s)', 'Viscosity', 'viscosity']
        }
        
        # Print available columns for debugging
        print("Available columns in dataframe:", self.df.columns.tolist())
        
        # Check and handle hydrophobicity specifically
        self.check_and_handle_hydrophobicity()
    
    def check_and_handle_hydrophobicity(self):
        """Specifically check for hydrophobicity property and handle it if found"""
        # Look for any column containing hydrophobicity-related terms
        hydrophobicity_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if 'hydro' in col_lower or 'logp' in col_lower or 'log_p' in col_lower:
                hydrophobicity_col = col
                break
        
        # If found but not in our standard format, add it to alt_property_names
        if hydrophobicity_col and hydrophobicity_col not in self.alt_property_names['hydrophobicity']:
            print(f"Found hydrophobicity column: {hydrophobicity_col}")
            self.alt_property_names['hydrophobicity'].append(hydrophobicity_col)
    
    def get_property_column(self, property_short_name):
        """Find the actual column name in the dataframe for a given property short name"""
        possible_names = self.alt_property_names.get(property_short_name, [])
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def calculate_statistics(self):
        """Calculate basic statistics for all properties"""
        stats_data = {}
        
        # Print which properties are found and which are not
        print("Property detection results:")
        
        for prop_short_name, possible_names in self.alt_property_names.items():
            # Find the actual column name in the dataframe
            col_name = self.get_property_column(prop_short_name)
            
            print(f"  - {prop_short_name}: {'Found' if col_name else 'Not found'} - Looked for {possible_names}")
            
            if col_name is not None and col_name in self.df.columns:
                # Get the unit from property mapping if available
                unit = ''
                for orig_name, info in self.property_mapping.items():
                    if info['short_name'] == prop_short_name:
                        unit = info['unit']
                        break
                
                # Calculate statistics
                values = self.df[col_name].dropna()
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()
                    mean_val = values.mean()
                    
                    stats_data[prop_short_name] = {
                        'name': col_name,
                        'min': min_val,
                        'max': max_val,
                        'range': f"{min_val:.1f} - {max_val:.1f} {unit}",
                        'average': f"{mean_val:.1f} {unit}",
                        'mean': mean_val,
                        'unit': unit
                    }
        
        return stats_data
    
    def plot_property_distributions(self, figsize=(12, 8)):
        """Create histograms for all properties"""
        stats_data = self.calculate_statistics()
        
        if not stats_data:
            return None, "No property data available for plotting"
        
        # Determine number of properties to plot
        n_properties = len(stats_data)
        if n_properties == 0:
            return None, "No properties found for plotting"
        
        # Calculate grid dimensions
        n_cols = min(3, n_properties)
        n_rows = (n_properties + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes array for easy iteration
        if n_properties > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each property
        for i, (prop_name, stats) in enumerate(stats_data.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Get data
                values = self.df[stats['name']].dropna()
                
                # Create histogram with KDE
                sns.histplot(values, kde=True, ax=ax)
                
                # Add vertical lines for mean and range
                ax.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.1f}")
                ax.axvline(stats['min'], color='green', linestyle=':', label=f"Min: {stats['min']:.1f}")
                ax.axvline(stats['max'], color='green', linestyle=':', label=f"Max: {stats['max']:.1f}")
                
                # Set labels - avoid duplicate units
                ax.set_title(f"{stats['name']} Distribution")
                
                # Use just the property name without units for x-axis label, then add units once
                property_name = stats['name']
                if '(' in property_name:
                    # The property name already includes units in parentheses
                    ax.set_xlabel(property_name)
                else:
                    # Add units to the property name
                    ax.set_xlabel(f"{property_name} ({stats['unit']})")
                
                ax.set_ylabel("Frequency")
                ax.legend()
        
        # Hide any unused subplots
        for i in range(n_properties, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig, None
    
    def plot_property_boxplots(self, figsize=(12, 8)):
        """Create individual boxplots for each property"""
        stats_data = self.calculate_statistics()
        
        if not stats_data:
            return None, "No property data available for plotting"
        
        # Determine number of properties to plot
        n_properties = len(stats_data)
        if n_properties == 0:
            return None, "No properties found for plotting"
        
        # Calculate grid dimensions
        n_cols = min(3, n_properties)
        n_rows = (n_properties + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes array for easy iteration
        if n_properties > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each property
        for i, (prop_name, stats) in enumerate(stats_data.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Get data
                values = self.df[stats['name']].dropna()
                
                if len(values) > 0:
                    # Create boxplot
                    bp = ax.boxplot([values], patch_artist=True)
                    
                    # Customize boxplot colors
                    for box in bp['boxes']:
                        box.set(facecolor='lightblue', alpha=0.7)
                    
                    # Set labels - avoid duplicate units
                    ax.set_title(f"{stats['name']} Distribution")
                    
                    # Use just the property name for the title, and add units once to y-axis
                    property_name = stats['name']
                    if '(' in property_name:
                        # Extract just the property name without units
                        clean_name = property_name.split('(')[0].strip()
                        ax.set_ylabel(property_name)
                    else:
                        # Add units to the y-axis label
                        ax.set_ylabel(f"{property_name} ({stats['unit']})")
                    
                    # Remove x-axis labels since we only have one boxplot per subplot
                    ax.set_xticks([])
                    
                    # Add statistics as text
                    stats_text = f"Min: {stats['min']:.1f}\nMax: {stats['max']:.1f}\nMean: {stats['mean']:.1f}"
                    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, 
                           verticalalignment='bottom', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide any unused subplots
        for i in range(n_properties, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig, None
    
    def create_statistics_table(self):
        """Create a formatted table of statistics"""
        stats_data = self.calculate_statistics()
        
        if not stats_data:
            return pd.DataFrame()
        
        # Create DataFrame for display
        table_data = []
        for prop_name, stats in stats_data.items():
            # Use property name without duplicating units
            property_name = stats['name']
            
            table_data.append({
                'Property': property_name,
                'Range': stats['range'],
                'Average': stats['average']
            })
        
        return pd.DataFrame(table_data)

class PCAAnalyzer:
    """Class for Principal Component Analysis of ionic liquid properties"""
    
    def __init__(self, combinations):
        """Initialize with combinations data"""
        self.combinations = combinations
        self.df = pd.DataFrame(combinations)
        self.pca_model = None
        self.transformed_data = None
        self.explained_variance = None
        self.feature_names = None
        self.loadings = None
    
    def prepare_data(self, property_names=None, fragment_types=None):
        """
        Prepare data for PCA analysis by extracting and encoding fragments
        
        Args:
            property_names: List of property names to include in analysis (if None, use all numeric properties)
            fragment_types: List of fragment types to include (e.g., ['cation', 'anion'])
            
        Returns:
            DataFrame with prepared data
        """
        # If no property names specified, use all numeric columns
        if property_names is None:
            # Identify numeric columns (properties)
            property_names = [col for col in self.df.columns 
                            if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score']
                            and pd.api.types.is_numeric_dtype(self.df[col])]
        
        # If no fragment types specified, use all available
        if fragment_types is None:
            fragment_types = [col for col in ['cation', 'anion', 'alkyl_chain', 'functional_group'] 
                             if col in self.df.columns]
        
        # Extract fragment columns
        fragment_cols = [col for col in fragment_types if col in self.df.columns]
            
        # One-hot encode fragments
        encoded_fragments = {}
        for col in fragment_cols:
            # Fill NA values
            self.df[col] = self.df[col].fillna('None')
            
            # One-hot encode
            dummies = pd.get_dummies(self.df[col], prefix=col)
            encoded_fragments[col] = dummies
        
        # Combine all encoded fragments
        X = pd.concat(encoded_fragments.values(), axis=1)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X
    
    def run_analysis(self, n_components=2, property_names=None, fragment_types=None, scale=True):
        """
        Run PCA analysis on the ionic liquid data
        
        Args:
            n_components: Number of principal components to extract
            property_names: List of property names to include in analysis
            fragment_types: List of fragment types to include
            scale: Whether to standardize the data before PCA
            
        Returns:
            Dictionary containing PCA results
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Check if we have data
        if self.df is None or len(self.df) == 0:
            return {
                'pca_model': None,
                'transformed_data': np.array([]),
                'components': np.array([]),
                'explained_variance': np.array([]),
                'cumulative_variance': np.array([]),
                'feature_names': [],
                'loadings': np.array([])
            }
            
        # Validate property names
        valid_property_names = []
        if property_names is not None:
            for prop in property_names:
                if prop in self.df.columns:
                    valid_property_names.append(prop)
                    
        if len(valid_property_names) < 2:
            return {
                'pca_model': None,
                'transformed_data': np.array([]),
                'components': np.array([]),
                'explained_variance': np.array([]),
                'cumulative_variance': np.array([]),
                'feature_names': [],
                'loadings': np.array([])
            }
        
        try:
            # Extract features
            X = self.df[valid_property_names].copy()
            
            # Ensure all columns are numeric
            for col in X.columns:
                if not np.issubdtype(X[col].dtype, np.number):
                    try:
                        X[col] = pd.to_numeric(X[col])
                    except:
                        # If conversion fails, drop the column
                        X = X.drop(columns=[col])
                        valid_property_names.remove(col)
            
            # Check if we still have enough columns
            if len(X.columns) < 2:
                return {
                    'pca_model': None,
                    'transformed_data': np.array([]),
                    'components': np.array([]),
                    'explained_variance': np.array([]),
                    'cumulative_variance': np.array([]),
                    'feature_names': [],
                    'loadings': np.array([])
                }
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale the data if requested
            if scale:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            # Run PCA
            self.pca_model = PCA(n_components=min(n_components, len(valid_property_names), X_scaled.shape[0]))
            self.transformed_data = self.pca_model.fit_transform(X_scaled)
            
            # Store explained variance
            self.explained_variance = self.pca_model.explained_variance_ratio_
            
            # Calculate loadings (correlation between features and components)
            self.loadings = self.pca_model.components_.T * np.sqrt(self.pca_model.explained_variance_)
            
            # Create results dictionary
            results = {
                'pca_model': self.pca_model,
                'transformed_data': self.transformed_data,
                'components': self.pca_model.components_,
                'explained_variance': self.explained_variance,
                'cumulative_variance': np.cumsum(self.explained_variance),
                'feature_names': self.feature_names,
                'loadings': self.loadings
            }
            
            return results
        except Exception as e:
            return {
                'error': f"Failed to run PCA: {e}"
            }
    
    def get_top_features(self, n_features=10):
        """
        Get the top features (fragments) contributing to each principal component
        
        Args:
            n_features: Number of top features to return for each component
            
        Returns:
            Dictionary mapping component indices to lists of (feature, loading) tuples
        """
        if self.pca_model is None:
            raise ValueError("PCA analysis has not been run yet. Call run_analysis first.")
        
        top_features = {}
        
        for i in range(self.pca_model.components_.shape[0]):
            # Get loadings for this component
            component_loadings = self.loadings[:, i]
            
            # Get indices of top features by absolute loading value
            top_indices = np.argsort(np.abs(component_loadings))[::-1][:n_features]
            
            # Get feature names and loadings
            features = [(self.feature_names[idx], component_loadings[idx]) for idx in top_indices]
            
            top_features[i] = features
        
        return top_features
    
    def plot_explained_variance(self, figsize=(10, 6)):
        """
        Plot the explained variance by each principal component
        
        Args:
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        if self.pca_model is None:
            raise ValueError("PCA analysis has not been run yet. Call run_analysis first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual explained variance
        ax.bar(range(1, len(self.explained_variance) + 1), 
               self.explained_variance, 
               alpha=0.7, 
               label='Individual')
        
        # Plot cumulative explained variance
        ax.step(range(1, len(self.explained_variance) + 1), 
                np.cumsum(self.explained_variance), 
                where='mid', 
                label='Cumulative', 
                color='red')
        
        # Add labels and title
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance by Principal Component')
        ax.set_xticks(range(1, len(self.explained_variance) + 1))
        ax.set_ylim([0, 1.05])
        ax.legend()
        
        return fig, ax
    
    def plot_pca_scatter(self, color_by=None, figsize=(12, 8)):
        """
        Create a scatter plot of the first two principal components
        
        Args:
            color_by: Column name to color points by (must be in original dataframe)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        if self.pca_model is None:
            raise ValueError("PCA analysis has not been run yet. Call run_analysis first.")
        
        if self.transformed_data.shape[1] < 2:
            raise ValueError("Need at least 2 components for scatter plot")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Basic scatter plot
        if color_by is None or color_by not in self.df.columns:
            scatter = ax.scatter(self.transformed_data[:, 0], 
                                self.transformed_data[:, 1], 
                                alpha=0.7)
        else:
            # Color by specified column
            scatter = ax.scatter(self.transformed_data[:, 0], 
                                self.transformed_data[:, 1], 
                                c=self.df[color_by], 
                                cmap='viridis', 
                                alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_by)
        
        # Add labels and title
        ax.set_xlabel(f'PC1 ({self.explained_variance[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({self.explained_variance[1]:.2%} variance)')
        ax.set_title('PCA of Ionic Liquid Data')
        
        # Add grid
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    def plot_loading_vectors(self, n_features=5, figsize=(12, 10)):
        """
        Plot loading vectors (feature contributions) for the first two principal components
        
        Args:
            n_features: Number of top features to display for each component
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        if self.pca_model is None:
            raise ValueError("PCA analysis has not been run yet. Call run_analysis first.")
        
        if self.transformed_data.shape[1] < 2:
            raise ValueError("Need at least 2 components for loading vector plot")
        
        # Get top features for first two components
        top_features = self.get_top_features(n_features=n_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot data points
        ax.scatter(self.transformed_data[:, 0], 
                  self.transformed_data[:, 1], 
                  alpha=0.3, 
                  color='gray')
        
        # Plot loading vectors for top features
        for i, (feature, loading) in enumerate(top_features[0]):
            # Get loading for PC1 and PC2
            pc1_loading = self.loadings[self.feature_names.index(feature), 0]
            pc2_loading = self.loadings[self.feature_names.index(feature), 1]
            
            # Scale for visibility
            scale_factor = 1.0
            
            # Plot vector
            ax.arrow(0, 0, 
                    pc1_loading * scale_factor, 
                    pc2_loading * scale_factor, 
                    head_width=0.05, 
                    head_length=0.05, 
                    fc='blue', 
                    ec='blue')
            
            # Add label
            ax.text(pc1_loading * scale_factor * 1.1, 
                   pc2_loading * scale_factor * 1.1, 
                   feature, 
                   fontsize=9)
        
        # Add labels and title
        ax.set_xlabel(f'PC1 ({self.explained_variance[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({self.explained_variance[1]:.2%} variance)')
        ax.set_title('PCA Loading Vectors')
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Make plot square and set limits
        ax.set_aspect('equal')
        
        return fig, ax
    
    def plot_3d_scatter(self, color_by=None, figsize=(12, 10)):
        """
        Create a 3D scatter plot of the first three principal components
        
        Args:
            color_by: Column name to color points by (must be in original dataframe)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        if self.pca_model is None:
            raise ValueError("PCA analysis has not been run yet. Call run_analysis first.")
        
        if self.transformed_data.shape[1] < 3:
            raise ValueError("Need at least 3 components for 3D scatter plot")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Basic scatter plot
        if color_by is None or color_by not in self.df.columns:
            scatter = ax.scatter(self.transformed_data[:, 0], 
                                self.transformed_data[:, 1], 
                                self.transformed_data[:, 2],
                                alpha=0.7)
        else:
            # Color by specified column
            scatter = ax.scatter(self.transformed_data[:, 0], 
                                self.transformed_data[:, 1], 
                                self.transformed_data[:, 2],
                                c=self.df[color_by], 
                                cmap='viridis', 
                                alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_by)
        
        # Add labels and title
        ax.set_xlabel(f'PC1 ({self.explained_variance[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({self.explained_variance[1]:.2%} variance)')
        ax.set_zlabel(f'PC3 ({self.explained_variance[2]:.2%} variance)')
        ax.set_title('3D PCA of Ionic Liquid Data')
        
        return fig, ax

class ClusterAnalyzer:
    """Class for cluster analysis of ionic liquid properties"""
    
    def __init__(self, combinations):
        """Initialize with combinations data"""
        self.combinations = combinations
        self.df = pd.DataFrame(combinations)
        self.reduced_data = None
        self.pca_model = None
    
    def prepare_data(self, property_names=None, fragment_types=None, use_pca=True, n_components=2):
        """
        Prepare data for cluster analysis by extracting and encoding fragments
        
        Args:
            property_names: List of property names to include in analysis (if None, use all numeric properties)
            fragment_types: List of fragment types to include (e.g., ['cation', 'anion'])
            use_pca: Whether to reduce dimensionality using PCA before clustering
            n_components: Number of principal components to use if use_pca is True
            
        Returns:
            DataFrame with prepared data
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # If no property names specified, use all numeric columns
        if property_names is None:
            # Identify numeric columns (properties)
            property_names = [col for col in self.df.columns 
                            if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score']
                            and pd.api.types.is_numeric_dtype(self.df[col])]
            
            if not property_names:
                return {'error': "No numeric properties found for cluster analysis"}
        else:
            # Verify that all specified properties are numeric
            non_numeric_props = [prop for prop in property_names if not pd.api.types.is_numeric_dtype(self.df[prop])]
            if non_numeric_props:
                return {'error': f"The following properties are not numeric and cannot be used for clustering: {', '.join(non_numeric_props)}"}
        
        # If no fragment types specified, use all available
        if fragment_types is None:
            fragment_types = [col for col in ['cation', 'anion', 'alkyl_chain', 'functional_group'] 
                             if col in self.df.columns]
            
            if not fragment_types:
                return {'error': "No fragment type columns found for cluster analysis"}
        
        # Extract fragment columns
        fragment_cols = [col for col in fragment_types if col in self.df.columns]
        
        if not fragment_cols:
            return {'error': f"None of the specified fragment types {fragment_types} were found in the data"}
            
        # One-hot encode fragments
        encoded_fragments = {}
        for col in fragment_cols:
            # Fill NA values
            self.df[col] = self.df[col].fillna('None')
            
            # One-hot encode
            dummies = pd.get_dummies(self.df[col], prefix=col)
            encoded_fragments[col] = dummies
        
        # Combine all encoded fragments
        X = pd.concat(encoded_fragments.values(), axis=1)
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optionally reduce dimensionality with PCA
        if use_pca:
            try:
                if n_components > X_scaled.shape[1]:
                    n_components = X_scaled.shape[1]
                    
                self.pca_model = PCA(n_components=n_components)
                self.reduced_data = self.pca_model.fit_transform(X_scaled)
                return self.reduced_data
            except Exception as e:
                return {'error': f"Error performing PCA: {str(e)}"}
        else:
            self.reduced_data = X_scaled
            return self.reduced_data
    
    def run_kmeans_analysis(self, n_clusters=3, property_names=None, fragment_types=None, use_pca=True, n_components=2):
        """
        Run K-means clustering analysis on the ionic liquid data
        
        Args:
            n_clusters: Number of clusters for K-means
            property_names: List of property names to include in analysis
            fragment_types: List of fragment types to include
            use_pca: Whether to reduce dimensionality using PCA before clustering
            n_components: Number of principal components to use if use_pca is True
            
        Returns:
            Dictionary with cluster labels, reduced data, and silhouette score
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Prepare data
        prepared_data = self.prepare_data(
            property_names=property_names,
            fragment_types=fragment_types,
            use_pca=use_pca,
            n_components=n_components
        )
        
        # Check if there was an error
        if isinstance(prepared_data, dict) and 'error' in prepared_data:
            return prepared_data
        
        try:
            # Run K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.reduced_data)
            
            # Calculate silhouette score if we have more than one cluster
            silhouette = None
            if n_clusters > 1:
                silhouette = silhouette_score(self.reduced_data, cluster_labels)
            
            return {
                'cluster_labels': cluster_labels,
                'reduced_data': self.reduced_data,
                'silhouette_score': silhouette
            }
        except Exception as e:
            return {'error': f"Error running K-means clustering: {str(e)}"}
    
    def run_hierarchical_analysis(self, n_clusters=3, property_names=None, fragment_types=None, 
                                 linkage='ward', use_pca=True, n_components=2):
        """
        Run hierarchical clustering analysis on the ionic liquid data
        
        Args:
            n_clusters: Number of clusters
            property_names: List of property names to include in analysis
            fragment_types: List of fragment types to include
            linkage: Linkage method for hierarchical clustering
            use_pca: Whether to reduce dimensionality using PCA before clustering
            n_components: Number of principal components to use if use_pca is True
            
        Returns:
            Dictionary with cluster labels, reduced data, and silhouette score
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Prepare data
        prepared_data = self.prepare_data(
            property_names=property_names,
            fragment_types=fragment_types,
            use_pca=use_pca,
            n_components=n_components
        )
        
        # Check if there was an error
        if isinstance(prepared_data, dict) and 'error' in prepared_data:
            return prepared_data
        
        try:
            # Run hierarchical clustering
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            cluster_labels = hierarchical.fit_predict(self.reduced_data)
            
            # Calculate silhouette score if we have more than one cluster
            silhouette = None
            if n_clusters > 1:
                silhouette = silhouette_score(self.reduced_data, cluster_labels)
            
            return {
                'cluster_labels': cluster_labels,
                'reduced_data': self.reduced_data,
                'silhouette_score': silhouette
            }
        except Exception as e:
            return {'error': f"Error running hierarchical clustering: {str(e)}"}
    
    def run_dbscan_analysis(self, eps=0.5, min_samples=5, property_names=None, fragment_types=None, 
                           use_pca=True, n_components=2):
        """
        Run DBSCAN clustering analysis on the ionic liquid data
        
        Args:
            eps: Maximum distance between samples for DBSCAN
            min_samples: Minimum number of samples in a neighborhood for DBSCAN
            property_names: List of property names to include in analysis
            fragment_types: List of fragment types to include
            use_pca: Whether to reduce dimensionality using PCA before clustering
            n_components: Number of principal components to use if use_pca is True
            
        Returns:
            Dictionary with cluster labels, reduced data, and silhouette score
        """
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        
        # Prepare data
        prepared_data = self.prepare_data(
            property_names=property_names,
            fragment_types=fragment_types,
            use_pca=use_pca,
            n_components=n_components
        )
        
        # Check if there was an error
        if isinstance(prepared_data, dict) and 'error' in prepared_data:
            return prepared_data
        
        try:
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(self.reduced_data)
            
            # Calculate silhouette score if we have more than one cluster and no noise points
            silhouette = None
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                silhouette = silhouette_score(self.reduced_data, cluster_labels)
            
            return {
                'cluster_labels': cluster_labels,
                'reduced_data': self.reduced_data,
                'silhouette_score': silhouette
            }
        except Exception as e:
            return {'error': f"Error running DBSCAN clustering: {str(e)}"}
    
    def determine_optimal_k(self, max_k=10, property_names=None, fragment_types=None, use_pca=True, n_components=2):
        """
        Determine the optimal number of clusters using the elbow method and silhouette scores
        
        Args:
            max_k: Maximum number of clusters to try
            property_names: List of property names to include in analysis
            fragment_types: List of fragment types to include
            use_pca: Whether to reduce dimensionality using PCA before clustering
            n_components: Number of principal components to use if use_pca is True
            
        Returns:
            Dictionary with k values, inertia values, and silhouette scores
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Prepare data
        prepared_data = self.prepare_data(
            property_names=property_names,
            fragment_types=fragment_types,
            use_pca=use_pca,
            n_components=n_components
        )
        
        # Check if there was an error
        if isinstance(prepared_data, dict) and 'error' in prepared_data:
            return prepared_data
        
        try:
            # Calculate inertia and silhouette score for different k values
            inertia = []
            silhouette = []
            k_values = range(1, max_k + 1)
            
            for k in k_values:
                # Run K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(self.reduced_data)
                inertia.append(kmeans.inertia_)
                
                # Calculate silhouette score if k > 1
                if k > 1:
                    labels = kmeans.labels_
                    silhouette.append(silhouette_score(self.reduced_data, labels))
                else:
                    silhouette.append(0)  # Silhouette score is not defined for k=1
            
            # Find optimal k using the elbow method
            # Simple approach: find the k where the rate of change in inertia slows down
            inertia_diff = np.diff(inertia)
            inertia_diff_rate = np.diff(inertia_diff)
            optimal_k_idx = np.argmin(inertia_diff_rate) + 2  # +2 because we took diff twice
            optimal_k = k_values[optimal_k_idx] if optimal_k_idx < len(k_values) else k_values[-1]
            
            return {
                'k_values': list(k_values),
                'inertia': inertia,
                'silhouette': silhouette,
                'optimal_k': optimal_k
            }
        except Exception as e:
            return {'error': f"Error determining optimal k: {str(e)}"}
    
    def plot_clusters_2d(self, figsize=(12, 8)):
        """
        Create a 2D scatter plot of clusters
        
        Args:
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        if self.reduced_data is None or self.reduced_data.shape[1] < 2:
            raise ValueError("Need at least 2 dimensions for 2D scatter plot.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique labels
        unique_labels = np.unique(self.cluster_labels)
        
        # Create a colormap
        cmap = plt.cm.get_cmap('viridis', len(unique_labels))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Plot noise points as black
                mask = self.cluster_labels == label
                ax.scatter(self.reduced_data[mask, 0], 
                          self.reduced_data[mask, 1], 
                          c='black', 
                          marker='x', 
                          alpha=0.5, 
                          label='Noise')
            else:
                # Plot cluster points
                mask = self.cluster_labels == label
                ax.scatter(self.reduced_data[mask, 0], 
                          self.reduced_data[mask, 1], 
                          c=[cmap(i)], 
                          marker='o', 
                          alpha=0.7, 
                          label=f'Cluster {label}')
        
        # Add labels and title
        if self.pca_model is not None:
            ax.set_xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]:.2%} variance)')
        else:
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        
        ax.set_title('Cluster Analysis of Ionic Liquid Data')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    def plot_clusters_3d(self, figsize=(12, 10)):
        """
        Create a 3D scatter plot of clusters
        
        Args:
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        if self.reduced_data is None or self.reduced_data.shape[1] < 3:
            raise ValueError("Need at least 3 dimensions for 3D scatter plot.")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique labels
        unique_labels = np.unique(self.cluster_labels)
        
        # Create a colormap
        cmap = plt.cm.get_cmap('viridis', len(unique_labels))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Plot noise points as black
                mask = self.cluster_labels == label
                ax.scatter(self.reduced_data[mask, 0], 
                          self.reduced_data[mask, 1], 
                          self.reduced_data[mask, 2],
                          c='black', 
                          marker='x', 
                          alpha=0.5, 
                          label='Noise')
            else:
                # Plot cluster points
                mask = self.cluster_labels == label
                ax.scatter(self.reduced_data[mask, 0], 
                          self.reduced_data[mask, 1], 
                          self.reduced_data[mask, 2],
                          c=[cmap(i)], 
                          marker='o', 
                          alpha=0.7, 
                          label=f'Cluster {label}')
        
        # Add labels and title
        if self.pca_model is not None:
            ax.set_xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_zlabel(f'PC3 ({self.pca_model.explained_variance_ratio_[2]:.2%} variance)')
        else:
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
        
        ax.set_title('3D Cluster Analysis of Ionic Liquid Data')
        
        # Add legend
        ax.legend()
        
        return fig, ax
    
    def plot_elbow_method(self, results, figsize=(12, 6)):
        """
        Plot the elbow method results
        
        Args:
            results: Results from determine_optimal_k method
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot inertia (elbow method)
        ax1.plot(results['k_values'], results['inertia'], 'o-', color='blue')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(alpha=0.3)
        
        # Plot silhouette scores
        ax2.plot(results['k_values'], results['silhouette'], 'o-', color='green')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Method')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        return fig, (ax1, ax2)
    
    def get_cluster_properties(self):
        """
        Calculate average properties for each cluster
        
        Returns:
            DataFrame with average properties for each cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been run yet. Call a clustering method first.")
        
        # Add cluster labels to dataframe
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = self.cluster_labels
        
        # Get numeric columns (properties)
        property_cols = [col for col in df_with_clusters.columns 
                       if col not in ['name', 'cation', 'anion', 'alkyl_chain', 'functional_group', 'in_ilthermo', 'pareto_score', 'cluster']
                       and pd.api.types.is_numeric_dtype(df_with_clusters[col])]
        
        # Calculate average properties for each cluster
        cluster_properties = df_with_clusters.groupby('cluster')[property_cols].mean()
        
        # Add count of ionic liquids in each cluster
        cluster_properties['count'] = df_with_clusters.groupby('cluster').size()
        
        return cluster_properties
    
    def get_cluster_compositions(self):
        """
        Calculate fragment composition for each cluster
        
        Returns:
            Dictionary mapping fragment types to DataFrames with composition for each cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been run yet. Call a clustering method first.")
        
        # Add cluster labels to dataframe
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = self.cluster_labels
        
        # Get fragment columns
        fragment_cols = [col for col in ['cation', 'anion', 'alkyl_chain', 'functional_group'] 
                        if col in df_with_clusters.columns]
        
        # Calculate composition for each fragment type
        compositions = {}
        
        for col in fragment_cols:
            # Count occurrences of each fragment in each cluster
            composition = df_with_clusters.groupby(['cluster', col]).size().unstack(fill_value=0)
            
            # Convert to percentages
            composition = composition.div(composition.sum(axis=1), axis=0) * 100
            
            compositions[col] = composition
        
        return compositions

def test_analysis():
    """Function to test the analysis module with sample data"""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Generate random fragment data
    cations = ['Imidazolium', 'Pyridinium', 'Ammonium', 'Phosphonium']
    anions = ['Chloride', 'Bromide', 'Tetrafluoroborate', 'Hexafluorophosphate']
    alkyls = ['methyl', 'ethyl', 'propyl', 'butyl', 'pentyl']
    
    # Create sample dataframe
    data = pd.DataFrame({
        'cation': np.random.choice(cations, n_samples),
        'anion': np.random.choice(anions, n_samples),
        'alkyl_chains': np.random.choice(alkyls, n_samples),
        'functional_group': np.random.choice(['Hydroxyl', 'Amino', None], n_samples),
    })
    
    # Add effects based on fragments (for simulation)
    effects = {
        'Imidazolium': 100, 'Pyridinium': 120, 'Ammonium': 90, 'Phosphonium': 110,  # Cations
        'Chloride': 30, 'Bromide': 35, 'Tetrafluoroborate': 40, 'Hexafluorophosphate': 45,  # Anions
        'methyl': 10, 'ethyl': 20, 'propyl': 30, 'butyl': 40, 'pentyl': 50,  # Alkyls
        'Hydroxyl': 25, 'Amino': 15, None: 0  # Functional groups
    }
    
    # Calculate properties
    data['heat_capacity'] = data.apply(
        lambda row: effects[row['cation']] + effects[row['anion']] + 
                  effects[row['alkyl_chains']] + effects[row['functional_group']] + 
                  np.random.normal(0, 15),  # Add noise
        axis=1
    )
    
    data['viscosity'] = data.apply(
        lambda row: effects[row['cation']] * 0.5 + effects[row['anion']] * 0.3 + 
                  effects[row['alkyl_chains']] * 2 + (effects[row['functional_group']] * 0.8 if row['functional_group'] else 0) + 
                  np.random.normal(0, 10),  # Add noise
        axis=1
    )
    
    # Create names for ILs
    data['Name'] = data.apply(
        lambda row: f"{row['alkyl_chains']}-{row['cation']} {row['anion']}" + 
                  (f" {row['functional_group']}" if row['functional_group'] else ""),
        axis=1
    )
    
    # Initialize analyzer
    analyzer = IonicLiquidAnalysis(data=data)
    
    # Run complete analysis
    analyzer.run_complete_analysis(
        target_cols=['heat_capacity', 'viscosity'],
        alpha=0.1,
        uncertainty_pct=8.0,
        num_simulations=200,
        output_dir='sample_analysis'
    )
    
    return analyzer

if __name__ == "__main__":
    # Run test
    analyzer = test_analysis()