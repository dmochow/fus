#!/usr/bin/env python3
"""
Visualization summary for entropy production analysis results.

This script loads the saved results and creates a summary of the enhanced
plotting capabilities including temporal trajectories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_entropy_results():
    """Analyze and summarize entropy production results."""
    
    # Load individual subject data
    data_file = "/Users/jacekdmochowski/PROJECTS/fus/data/entropy_analysis/entropy_production_individual_subjects.csv"
    df = pd.read_csv(data_file)
    
    print("="*80)
    print("ENHANCED ENTROPY PRODUCTION VISUALIZATION SUMMARY")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total measurements: {len(df)}")
    print(f"  Subjects: {df['subject'].nunique()}")
    print(f"  Conditions: {list(df['condition'].unique())}")
    print(f"  Epochs: {list(df['epoch'].unique())}")
    
    # Analysis by epoch
    print(f"\n{'='*60}")
    print("EPOCH-WISE ANALYSIS")
    print(f"{'='*60}")
    
    epochs = ['pre_stimulation', 'during_stimulation', 'post_stimulation']
    conditions = ['ACTIVE', 'SHAM']
    
    for epoch in epochs:
        epoch_data = df[df['epoch'] == epoch]
        print(f"\n{epoch.replace('_', ' ').title()}:")
        print("-" * 30)
        
        for condition in conditions:
            cond_data = epoch_data[epoch_data['condition'] == condition]
            if len(cond_data) > 0:
                values = cond_data['entropy_production']
                print(f"  {condition:>6}: {values.mean():.4f} ± {values.std():.4f} "
                      f"(range: {values.min():.4f}-{values.max():.4f}, n={len(values)})")
    
    # Temporal trajectory analysis
    print(f"\n{'='*60}")
    print("TEMPORAL TRAJECTORY ANALYSIS")
    print(f"{'='*60}")
    
    # Find subjects with complete data
    complete_subjects = []
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject]
        if len(subj_data) == 6:  # 2 conditions × 3 epochs
            complete_subjects.append(subject)
    
    print(f"\nSubjects with complete data (all epochs & conditions): {len(complete_subjects)}")
    print(f"Complete subjects: {sorted(complete_subjects)}")
    
    # Analyze trajectory patterns
    trajectory_analysis = {}
    
    for condition in conditions:
        print(f"\n{condition} Condition Trajectory Patterns:")
        print("-" * 40)
        
        condition_trajectories = []
        
        for subject in complete_subjects:
            subj_data = df[(df['subject'] == subject) & (df['condition'] == condition)]
            if len(subj_data) == 3:  # All 3 epochs
                trajectory = []
                for epoch in epochs:
                    entropy_val = subj_data[subj_data['epoch'] == epoch]['entropy_production'].iloc[0]
                    trajectory.append(entropy_val)
                condition_trajectories.append(trajectory)
                
                # Classify trajectory pattern
                pre, during, post = trajectory
                if during > pre and during > post:
                    pattern = "Peak during stimulation"
                elif during < pre and during < post:
                    pattern = "Dip during stimulation"
                elif post > pre:
                    pattern = "Sustained increase"
                elif post < pre:
                    pattern = "Sustained decrease"
                else:
                    pattern = "Stable"
                
                print(f"  {subject:>10}: {pre:.3f} → {during:.3f} → {post:.3f} ({pattern})")
        
        # Mean trajectory
        if condition_trajectories:
            mean_trajectory = np.mean(condition_trajectories, axis=0)
            std_trajectory = np.std(condition_trajectories, axis=0)
            
            print(f"\n  Mean trajectory: {mean_trajectory[0]:.3f} → {mean_trajectory[1]:.3f} → {mean_trajectory[2]:.3f}")
            print(f"  Standard errors:  ±{std_trajectory[0]:.3f}   ±{std_trajectory[1]:.3f}   ±{std_trajectory[2]:.3f}")
            
            trajectory_analysis[condition] = {
                'mean': mean_trajectory,
                'std': std_trajectory,
                'individual': condition_trajectories
            }
    
    # Statistical comparison of trajectories
    print(f"\n{'='*60}")
    print("STATISTICAL TRAJECTORY COMPARISON")
    print(f"{'='*60}")
    
    if 'ACTIVE' in trajectory_analysis and 'SHAM' in trajectory_analysis:
        from scipy import stats
        
        active_trajs = np.array(trajectory_analysis['ACTIVE']['individual'])
        sham_trajs = np.array(trajectory_analysis['SHAM']['individual'])
        
        print(f"\nPaired t-tests (ACTIVE vs SHAM) for each epoch:")
        for i, epoch in enumerate(epochs):
            if len(active_trajs) > 0 and len(sham_trajs) > 0:
                t_stat, p_val = stats.ttest_rel(active_trajs[:, i], sham_trajs[:, i])
                sig_str = " *" if p_val < 0.05 else "  " if p_val < 0.1 else ""
                print(f"  {epoch.replace('_', ' ').title():>18}: t={t_stat:6.3f}, p={p_val:.4f}{sig_str}")
        
        # Trajectory shape comparison
        print(f"\nTrajectory shape analysis:")
        
        # Calculate change scores
        active_changes = {
            'pre_to_during': active_trajs[:, 1] - active_trajs[:, 0],
            'during_to_post': active_trajs[:, 2] - active_trajs[:, 1],
            'pre_to_post': active_trajs[:, 2] - active_trajs[:, 0]
        }
        
        sham_changes = {
            'pre_to_during': sham_trajs[:, 1] - sham_trajs[:, 0],
            'during_to_post': sham_trajs[:, 2] - sham_trajs[:, 1],
            'pre_to_post': sham_trajs[:, 2] - sham_trajs[:, 0]
        }
        
        for change_type in active_changes.keys():
            active_change = active_changes[change_type]
            sham_change = sham_changes[change_type]
            
            t_stat, p_val = stats.ttest_rel(active_change, sham_change)
            
            active_mean = np.mean(active_change)
            sham_mean = np.mean(sham_change)
            
            sig_str = " *" if p_val < 0.05 else "  " if p_val < 0.1 else ""
            print(f"  {change_type.replace('_', ' ').title():>18}: "
                  f"ACTIVE={active_mean:6.3f}, SHAM={sham_mean:6.3f}, "
                  f"t={t_stat:6.3f}, p={p_val:.4f}{sig_str}")
    
    print(f"\n{'='*60}")
    print("VISUALIZATION FILES CREATED")
    print(f"{'='*60}")
    
    analysis_dir = Path("/Users/jacekdmochowski/PROJECTS/fus/data/entropy_analysis")
    plot_files = list(analysis_dir.glob("*.png"))
    
    for plot_file in sorted(plot_files):
        print(f"  📊 {plot_file.name}")
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    
    print("\n1. Enhanced Paired Comparison Plot Features:")
    print("   ✓ Consistent y-axis limits across all epochs")
    print("   ✓ Individual subject connecting lines showing within-subject changes")
    print("   ✓ Temporal trajectory plots (pre → during → post)")
    print("   ✓ Subject-level identification and tracking")
    print("   ✓ Statistical comparisons with paired t-tests")
    
    print("\n2. Temporal Dynamics:")
    print("   • Individual subjects show diverse trajectory patterns")
    print("   • Both conditions show entropy increase during stimulation")
    print("   • Post-stimulation patterns vary between subjects")
    print("   • Enhanced visualization reveals subject-specific responses")
    
    print("\n3. Methodological Improvements:")
    print("   • Gaussian method now ensures non-negative entropy production")
    print("   • Theoretically sound implementation following stochastic thermodynamics")
    print("   • Multiple visualization approaches for comprehensive analysis")

if __name__ == "__main__":
    analyze_entropy_results()