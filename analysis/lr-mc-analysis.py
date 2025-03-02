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
        
    def run_analysis(self, target_property: str, num_simulations: int = 500, uncertainty_percentage: float = 10):
        """Run Monte Carlo simulation for uncertainty analysis"""
        print(f"Running Monte Carlo analysis for {target_property} with {num_simulations} simulations")
        
        # Get the property values
        values = self.df[target_property].values
        
        # Calculate standard deviation for uncertainty
        std_dev = values.std() * (uncertainty_percentage / 100)
        
        # Run simulations
        simulation_results = []
        for _ in range(num_simulations):
            # Add random noise to each value
            noise = np.random.normal(0, std_dev, size=len(values))
            simulated_values = values + noise
            
            # Calculate mean of simulated values
            simulation_results.append(simulated_values.mean())
            
        return simulation_results

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