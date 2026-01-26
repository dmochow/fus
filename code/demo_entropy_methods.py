#!/usr/bin/env python3
"""
Demonstration script for different entropy production estimation methods.

This script shows how to use the parameterized main function with different
methods and processing types.
"""

from entropy_production_analysis import main
import pandas as pd

def compare_methods():
    """Compare different entropy estimation methods."""
    
    print("="*80)
    print("COMPARING ENTROPY PRODUCTION ESTIMATION METHODS")
    print("="*80)
    
    methods = ['discrete', 'gaussian']  # 'kde' can be slow for demonstration
    processing_type = 'conservative'  # Use conservative as it was last run
    
    results_summary = []
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"RUNNING WITH METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        try:
            # Run analysis
            active_results, sham_results, comparison_results = main(
                processing_type=processing_type,
                entropy_method=method
            )
            
            # Collect summary statistics
            for epoch, data in comparison_results.items():
                results_summary.append({
                    'method': method,
                    'epoch': epoch,
                    'active_mean': data['active_mean'],
                    'sham_mean': data['sham_mean'],
                    'difference': data['difference'],
                    'p_value': data['p_value']
                })
                
        except Exception as e:
            print(f"Error with method {method}: {str(e)}")
            continue
    
    # Create comparison table
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        
        print(f"\n{'='*80}")
        print("METHODS COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Pivot table for easier comparison
        for epoch in ['pre_stimulation', 'during_stimulation', 'post_stimulation']:
            epoch_data = summary_df[summary_df['epoch'] == epoch]
            
            print(f"\n{epoch.replace('_', ' ').title()}:")
            print("-" * 40)
            for _, row in epoch_data.iterrows():
                print(f"  {row['method'].capitalize():>10}: "
                      f"Active={row['active_mean']:.4f}, "
                      f"Sham={row['sham_mean']:.4f}, "
                      f"Diff={row['difference']:.4f}, "
                      f"p={row['p_value']:.4f}")
        
        # Save comparison
        comparison_file = "/Users/jacekdmochowski/PROJECTS/fus/data/entropy_analysis/methods_comparison.csv"
        summary_df.to_csv(comparison_file, index=False)
        print(f"\nMethods comparison saved: {comparison_file}")

def demo_individual_analysis():
    """Demonstrate how to access individual subject data."""
    
    # Load the individual subject data that was saved
    subject_file = "/Users/jacekdmochowski/PROJECTS/fus/data/entropy_analysis/entropy_production_individual_subjects.csv"
    
    try:
        df = pd.read_csv(subject_file)
        
        print(f"\n{'='*60}")
        print("INDIVIDUAL SUBJECT DATA SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total records: {len(df)}")
        print(f"Subjects: {df['subject'].nunique()}")
        print(f"Conditions: {df['condition'].unique()}")
        print(f"Epochs: {df['epoch'].unique()}")
        
        # Show data for each epoch
        for epoch in df['epoch'].unique():
            epoch_data = df[df['epoch'] == epoch]
            
            print(f"\n{epoch.replace('_', ' ').title()}:")
            print("-" * 30)
            
            for condition in ['ACTIVE', 'SHAM']:
                condition_data = epoch_data[epoch_data['condition'] == condition]['entropy_production']
                print(f"  {condition}: {condition_data.mean():.4f} ± {condition_data.std():.4f} "
                      f"(n={len(condition_data)})")
        
        # Show top and bottom entropy producers by epoch
        print(f"\n{'='*60}")
        print("EXTREME ENTROPY PRODUCERS BY EPOCH")
        print(f"{'='*60}")
        
        for epoch in df['epoch'].unique():
            epoch_data = df[df['epoch'] == epoch]
            
            # Highest entropy
            highest = epoch_data.loc[epoch_data['entropy_production'].idxmax()]
            lowest = epoch_data.loc[epoch_data['entropy_production'].idxmin()]
            
            print(f"\n{epoch.replace('_', ' ').title()}:")
            print(f"  Highest: {highest['subject']} ({highest['condition']}) = {highest['entropy_production']:.4f}")
            print(f"  Lowest:  {lowest['subject']} ({lowest['condition']}) = {lowest['entropy_production']:.4f}")
        
    except FileNotFoundError:
        print("Individual subject data file not found. Run the main analysis first.")

if __name__ == "__main__":
    # Demonstrate individual data analysis
    demo_individual_analysis()
    
    # Optionally compare methods (uncomment to run)
    # compare_methods()
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES:")
    print(f"{'='*80}")
    print("from entropy_production_analysis import main")
    print("")
    print("# Default: minimal preprocessing, discrete method")
    print("results = main()")
    print("")
    print("# Conservative preprocessing with Gaussian method")
    print("results = main(processing_type='conservative', entropy_method='gaussian')")
    print("")
    print("# Kernel density estimation method")
    print("results = main(entropy_method='kde')")
    print("")
    print("# Custom ROI data directory")
    print("results = main(roi_data_dir='/path/to/roi/data')")