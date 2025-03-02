"""
Test script for the PCA analysis functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis.lr_mc_analysis import PCAAnalyzer
from analysis.il_decon import IonicLiquidAnalyzer

def generate_test_data(n_samples=100):
    """Generate test data for PCA analysis"""
    # Create a simple ionic liquid analyzer
    il_analyzer = IonicLiquidAnalyzer()
    
    # Define some common fragments
    cations = ['imidazolium', 'pyridinium', 'ammonium', 'phosphonium']
    anions = ['Cl', 'Br', 'BF4', 'PF6', 'NTf2', 'OTf']
    alkyl_chains = ['methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl', 'octyl']
    functional_groups = [None, 'hydroxy', 'amino', 'carboxyl', 'nitrile']
    
    # Generate random combinations
    combinations = []
    for _ in range(n_samples):
        cation = np.random.choice(cations)
        anion = np.random.choice(anions)
        alkyl_chain = np.random.choice(alkyl_chains)
        functional_group = np.random.choice(functional_groups)
        
        # Create a simple name
        if functional_group:
            name = f"1-{functional_group}{alkyl_chain}-3-methyl{cation} {anion}"
        else:
            name = f"1-{alkyl_chain}-3-methyl{cation} {anion}"
        
        # Generate some properties with dependencies on fragments
        # Density depends on anion and alkyl chain length
        anion_density_factor = {'Cl': 1.2, 'Br': 1.5, 'BF4': 1.3, 'PF6': 1.4, 'NTf2': 1.6, 'OTf': 1.5}
        alkyl_density_factor = {'methyl': 0.1, 'ethyl': 0.2, 'propyl': 0.3, 'butyl': 0.4, 
                                'pentyl': 0.5, 'hexyl': 0.6, 'octyl': 0.8}
        
        density = anion_density_factor[anion] - alkyl_density_factor[alkyl_chain] + np.random.normal(0, 0.05)
        
        # Viscosity depends on alkyl chain length and functional group
        func_viscosity_factor = {None: 0.0, 'hydroxy': 0.5, 'amino': 0.3, 'carboxyl': 0.7, 'nitrile': 0.2}
        viscosity = 0.1 * len(alkyl_chain) + func_viscosity_factor[functional_group] + np.random.normal(0, 0.1)
        
        # Conductivity depends on cation and anion
        cation_cond_factor = {'imidazolium': 0.8, 'pyridinium': 0.6, 'ammonium': 0.4, 'phosphonium': 0.3}
        anion_cond_factor = {'Cl': 0.7, 'Br': 0.6, 'BF4': 0.8, 'PF6': 0.5, 'NTf2': 0.9, 'OTf': 0.7}
        
        conductivity = cation_cond_factor[cation] * anion_cond_factor[anion] + np.random.normal(0, 0.05)
        
        # Create combination dictionary
        combination = {
            'name': name,
            'cation': cation,
            'anion': anion,
            'alkyl_chain': alkyl_chain,
            'functional_group': functional_group,
            'density': density,
            'viscosity': viscosity,
            'conductivity': conductivity
        }
        
        combinations.append(combination)
    
    return combinations

def test_pca_analysis():
    """Test the PCA analysis functionality"""
    print("Generating test data...")
    combinations = generate_test_data(n_samples=200)
    
    print("Creating PCA analyzer...")
    pca_analyzer = PCAAnalyzer(combinations)
    
    print("Running PCA analysis...")
    results = pca_analyzer.run_analysis(
        n_components=3,
        fragment_types=['cation', 'anion', 'alkyl_chain', 'functional_group'],
        scale=True
    )
    
    print("\nPCA Results:")
    print(f"Number of components: {results['pca_model'].n_components_}")
    print(f"Explained variance: {results['explained_variance']}")
    print(f"Cumulative variance: {results['cumulative_variance']}")
    
    print("\nPlotting results...")
    
    # Plot explained variance
    fig1, ax1 = pca_analyzer.plot_explained_variance()
    fig1.savefig("pca_explained_variance.png")
    
    # Plot 2D scatter
    fig2, ax2 = pca_analyzer.plot_pca_scatter(color_by='density')
    fig2.savefig("pca_scatter_2d.png")
    
    # Plot loading vectors
    fig3, ax3 = pca_analyzer.plot_loading_vectors(n_features=10)
    fig3.savefig("pca_loading_vectors.png")
    
    # Plot 3D scatter
    fig4, ax4 = pca_analyzer.plot_3d_scatter(color_by='viscosity')
    fig4.savefig("pca_scatter_3d.png")
    
    print("\nGetting top features...")
    top_features = pca_analyzer.get_top_features(n_features=5)
    
    for i in range(3):  # Show top 3 components
        print(f"\nPrincipal Component {i+1} (Explains {results['explained_variance'][i]:.2%} of variance)")
        for feature, loading in top_features[i]:
            print(f"  {feature}: {loading:.4f}")
    
    print("\nPCA analysis test completed successfully!")

if __name__ == "__main__":
    test_pca_analysis()
