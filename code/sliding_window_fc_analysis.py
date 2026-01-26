"""
Sliding window functional connectivity analysis for tFUS fMRI data.

This script performs:
1. Sliding window FC analysis (20 TR windows, 75% overlap)
2. Time course visualization with tFUS periods overlay
3. Epoch-based FC analysis (pre/during/post stimulation)
4. Separate analysis for ACTIVE vs SHAM conditions

tFUS stimulation periods: TRs 300-320, 360-380, 420-440, 480-500, 540-560
Epochs: Pre (0-300), During (300-320∪360-380∪420-440∪480-500∪540-560), Post (580-900)

Author: Generated for tFUS project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SlidingWindowFCAnalyzer:
    """
    Sliding window functional connectivity analysis for tFUS data.
    """
    
    def __init__(self, roi_data_dir, output_dir=None):
        """
        Initialize FC analyzer.
        
        Parameters:
        -----------
        roi_data_dir : str or Path
            Path to ROI time series directory
        output_dir : str or Path, optional
            Output directory for FC analysis results
        """
        self.roi_data_dir = Path(roi_data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.roi_data_dir.parent / "fc_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis parameters
        self.window_length = 45  # TRs (20 seconds)
        self.overlap_percent = 95  # 75% overlap
        self.step_size = self.window_length * (1 - self.overlap_percent / 100)  # 5 TRs
        self.step_size = int(self.step_size)
        
        # tFUS stimulation periods (TR indices, 0-based)
        self.tfus_periods = [
            (300, 320),  # Period 1
            (360, 380),  # Period 2  
            (420, 440),  # Period 3
            (480, 500),  # Period 4
            (540, 560)   # Period 5
        ]
        
        # Epoch definitions
        self.epochs = {
            'pre_stimulation': (0, 300),
            'post_stimulation': (560, 900),
            'during_stimulation': self.tfus_periods  # Union of all tFUS periods
        }
        
        print(f"Sliding window parameters:")
        print(f"  Window length: {self.window_length} TRs")
        print(f"  Overlap: {self.overlap_percent}%")
        print(f"  Step size: {self.step_size} TRs")
        print(f"  tFUS periods: {self.tfus_periods}")
    
    def load_roi_data(self, processing_type='minimal'):
        """Load all ROI time series data."""
        print(f"Loading ROI data (processing: {processing_type})...")
        
        # Load combined dataset
        combined_file = self.roi_data_dir / "roi_timeseries_all_subjects.csv"
        if not combined_file.exists():
            raise FileNotFoundError(f"Combined ROI data not found: {combined_file}")
        
        df = pd.read_csv(combined_file)
        
        # Filter by processing type
        df_filtered = df[df['processing'] == processing_type].copy()
        
        # Get ROI columns
        roi_columns = [col for col in df_filtered.columns if col.startswith('ROI_')]
        
        print(f"  Loaded data shape: {df_filtered.shape}")
        print(f"  Number of ROIs: {len(roi_columns)}")
        print(f"  Subjects: {sorted(df_filtered['subject'].unique())}")
        print(f"  Sessions: {sorted(df_filtered['session'].unique())}")
        
        return df_filtered, roi_columns
    
    def compute_sliding_window_fc(self, timeseries_data, roi_columns):
        """
        Compute sliding window functional connectivity.
        
        Parameters:
        -----------
        timeseries_data : np.ndarray
            Time series data (n_timepoints, n_rois)
        roi_columns : list
            List of ROI column names
            
        Returns:
        --------
        fc_windows : np.ndarray
            FC matrices for each window (n_windows, n_rois, n_rois)
        window_centers : np.ndarray
            Center timepoints for each window
        """
        n_timepoints, n_rois = timeseries_data.shape
        
        # Calculate window positions
        window_starts = np.arange(0, n_timepoints - self.window_length + 1, self.step_size)
        n_windows = len(window_starts)
        window_centers = window_starts + self.window_length // 2
        
        print(f"    Computing {n_windows} sliding windows...")
        
        # Initialize FC matrix array with float32 to save memory
        fc_windows = np.zeros((n_windows, n_rois, n_rois), dtype=np.float32)
        
        # Compute FC for each window
        for i, start in enumerate(window_starts):
            if i % 50 == 0:
                print(f"      Computing window {i+1}/{n_windows}")
            
            end = start + self.window_length
            window_data = timeseries_data[start:end, :]
            
            # Compute correlation matrix
            fc_matrix = np.corrcoef(window_data.T)
            
            # Handle NaN values (replace with 0)
            fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
            
            fc_windows[i] = fc_matrix.astype(np.float32)
        
        print(f"    FC windows shape: {fc_windows.shape}")
        
        return fc_windows, window_centers
    
    def compute_epoch_fc(self, timeseries_data, roi_columns):
        """
        Compute FC matrices for different epochs (pre/during/post stimulation).
        
        Returns:
        --------
        epoch_fc : dict
            FC matrices for each epoch
        """
        print("    Computing epoch-based FC matrices...")
        
        epoch_fc = {}
        n_timepoints = timeseries_data.shape[0]
        
        for epoch_name, epoch_definition in self.epochs.items():
            print(f"      Computing FC for {epoch_name}...")
            
            if epoch_name == 'during_stimulation':
                # Union of all tFUS periods
                epoch_indices = []
                for start, end in epoch_definition:
                    if start < n_timepoints and end <= n_timepoints:
                        epoch_indices.extend(range(start, end))
                epoch_indices = np.array(sorted(set(epoch_indices)))
            else:
                # Single continuous period
                start, end = epoch_definition
                if start < n_timepoints and end <= n_timepoints:
                    epoch_indices = np.arange(start, end)
                else:
                    print(f"        Warning: Epoch {epoch_name} extends beyond data ({n_timepoints} TRs)")
                    epoch_indices = np.arange(start, min(end, n_timepoints))
            
            if len(epoch_indices) > 0:
                epoch_data = timeseries_data[epoch_indices, :]
                fc_matrix = np.corrcoef(epoch_data.T)
                fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
                epoch_fc[epoch_name] = fc_matrix
                
                print(f"        {epoch_name}: {len(epoch_indices)} TRs, FC shape: {fc_matrix.shape}")
            else:
                print(f"        Warning: No valid timepoints for {epoch_name}")
                epoch_fc[epoch_name] = None
        
        return epoch_fc
    
    def process_subject_session(self, subject, session, condition, df_subset, roi_columns):
        """Process a single subject-session combination."""
        print(f"  Processing {subject} - {session} ({condition})")
        
        # Extract time series data for this subject-session
        session_data = df_subset[
            (df_subset['subject'] == subject) & (df_subset['session'] == session)
        ].copy()
        
        if len(session_data) == 0:
            print(f"    No data found for {subject}-{session}")
            return None, None
        
        # Sort by timepoint and extract ROI data
        session_data = session_data.sort_values('timepoint')
        timeseries_matrix = session_data[roi_columns].values  # (n_timepoints, n_rois)
        
        print(f"    Time series shape: {timeseries_matrix.shape}")
        
        # Standardize time series (z-score each ROI)
        scaler = StandardScaler()
        timeseries_standardized = scaler.fit_transform(timeseries_matrix)
        
        # Compute sliding window FC
        fc_windows, window_centers = self.compute_sliding_window_fc(
            timeseries_standardized, roi_columns
        )
        
        # Compute epoch FC
        epoch_fc = self.compute_epoch_fc(timeseries_standardized, roi_columns)
        
        # Package results
        results = {
            'subject': subject,
            'session': session,
            'condition': condition,
            'fc_windows': fc_windows,
            'window_centers': window_centers,
            'epoch_fc': epoch_fc,
            'n_timepoints': len(session_data),
            'roi_columns': roi_columns
        }
        
        return results
    
    def analyze_condition(self, condition, processing_type='minimal', max_subjects=None):
        """
        Analyze all subjects for a specific condition (ACTIVE or SHAM).
        
        Parameters:
        -----------
        condition : str
            'ACTIVE' or 'SHAM'
        processing_type : str
            'minimal' or 'conservative'
        max_subjects : int, optional
            Maximum number of subjects to process (for testing/memory management)
        """
        print(f"\\n=== Analyzing {condition} condition ({processing_type}) ===")
        
        # Load data
        df, roi_columns = self.load_roi_data(processing_type)
        
        # Filter by condition
        df_condition = df[df['session'] == condition].copy()
        
        if len(df_condition) == 0:
            print(f"No data found for {condition} condition")
            return None
        
        # Get unique subjects
        subjects = sorted(df_condition['subject'].unique())
        if max_subjects is not None:
            subjects = subjects[:max_subjects]
            print(f"Processing first {len(subjects)} subjects (max_subjects={max_subjects})")
        
        print(f"Subjects in {condition} condition: {subjects}")
        
        # Process each subject
        all_results = []
        
        for subject in subjects:
            try:
                results = self.process_subject_session(
                    subject, condition, condition, df_condition, roi_columns
                )
                if results is not None:
                    all_results.append(results)
            except Exception as e:
                print(f"    ERROR processing {subject}: {str(e)}")
                continue
        
        if not all_results:
            print(f"No successful analyses for {condition} condition")
            return None
        
        print(f"Successfully processed {len(all_results)} subjects for {condition}")
        
        # Combine results and compute group averages
        group_results = self._compute_group_averages(all_results, condition, processing_type)
        
        return group_results
    
    def _compute_group_averages(self, all_results, condition, processing_type):
        """Compute group-averaged FC matrices and time courses."""
        print(f"  Computing group averages for {condition}...")
        
        # Extract individual results
        all_fc_windows = [r['fc_windows'] for r in all_results]
        all_window_centers = [r['window_centers'] for r in all_results]
        all_epoch_fc = [r['epoch_fc'] for r in all_results]
        
        # Find common window timepoints (intersection across subjects)
        min_windows = min(len(centers) for centers in all_window_centers)
        common_centers = all_window_centers[0][:min_windows]
        
        print(f"    Computing group average across {len(all_fc_windows)} subjects...")
        print(f"    Individual FC windows shape: {all_fc_windows[0].shape}")
        print(f"    Using {min_windows} common windows")
        
        # Average sliding window FC across subjects - process in chunks to save memory
        n_subjects = len(all_fc_windows)
        n_windows = min_windows
        n_rois = all_fc_windows[0].shape[1]
        
        # Initialize group average matrix
        group_fc_windows = np.zeros((n_windows, n_rois, n_rois), dtype=np.float32)
        
        # Process each window separately to avoid memory issues
        for w in range(n_windows):
            if w % 50 == 0:
                print(f"      Processing window {w+1}/{n_windows}")
            
            # Average this window across all subjects
            window_sum = np.zeros((n_rois, n_rois), dtype=np.float32)
            for subj_idx, fc_windows in enumerate(all_fc_windows):
                if w < fc_windows.shape[0]:  # Ensure window exists for this subject
                    window_sum += fc_windows[w].astype(np.float32)
            
            group_fc_windows[w] = window_sum / n_subjects
        
        print(f"    Group FC windows shape: {group_fc_windows.shape}")
        
        # Average epoch FC across subjects
        group_epoch_fc = {}
        epoch_names = all_epoch_fc[0].keys()
        
        for epoch in epoch_names:
            epoch_matrices = [r[epoch] for r in all_epoch_fc if r[epoch] is not None]
            if epoch_matrices:
                group_epoch_fc[epoch] = np.mean(epoch_matrices, axis=0)
                print(f"    {epoch}: averaged across {len(epoch_matrices)} subjects")
            else:
                group_epoch_fc[epoch] = None
        
        # Package group results
        group_results = {
            'condition': condition,
            'processing_type': processing_type,
            'n_subjects': len(all_results),
            'group_fc_windows': group_fc_windows,
            'window_centers': common_centers,
            'group_epoch_fc': group_epoch_fc,
            'individual_results': all_results,
            'roi_columns': all_results[0]['roi_columns']
        }
        
        # Save results
        self._save_results(group_results)
        
        return group_results
    
    def _save_results(self, group_results):
        """Save group results to files."""
        condition = group_results['condition']
        processing = group_results['processing_type']
        
        # Create condition-specific directory
        condition_dir = self.output_dir / f"{condition.lower()}_{processing}"
        condition_dir.mkdir(exist_ok=True)
        
        # Save sliding window FC
        np.save(
            condition_dir / "group_fc_windows.npy", 
            group_results['group_fc_windows']
        )
        np.save(
            condition_dir / "window_centers.npy", 
            group_results['window_centers']
        )
        
        # Save epoch FC
        for epoch, fc_matrix in group_results['group_epoch_fc'].items():
            if fc_matrix is not None:
                np.save(condition_dir / f"epoch_fc_{epoch}.npy", fc_matrix)
        
        # Save metadata
        metadata = {
            'condition': condition,
            'processing_type': processing,
            'n_subjects': group_results['n_subjects'],
            'window_length': self.window_length,
            'overlap_percent': self.overlap_percent,
            'tfus_periods': self.tfus_periods,
            'epochs': self.epochs
        }
        
        import json
        with open(condition_dir / "analysis_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"    Results saved to: {condition_dir}")
    
    def plot_fc_timecourse(self, active_results, sham_results, save_plots=True, smooth_window=5, smooth_method='moving_average', show_conditions=['active', 'sham']):
        """
        Plot consolidated FC time course with ACTIVE and/or SHAM conditions overlaid.
        
        Parameters:
        -----------
        active_results : dict
            ACTIVE condition results
        sham_results : dict
            SHAM condition results
        save_plots : bool, optional
            Whether to save plots to disk. Default is True.
        smooth_window : int, optional
            Size of smoothing window. Default is 5.
            Set to 0 or None to disable smoothing.
        smooth_method : str, optional
            Smoothing method ('moving_average' or 'savgol'). Default is 'moving_average'.
        show_conditions : list, optional
            List of conditions to show. Can contain 'active', 'sham', or both. Default is ['active', 'sham'].
        """
        print("Creating consolidated FC time course plot with ACTIVE and SHAM conditions...")
        
        def smooth_timeseries(data, window_size, method='moving_average'):
            """Apply smoothing to time series data."""
            if window_size is None or window_size <= 1:
                return data
            
            # For 1D data
            if data.ndim == 1:
                if method == 'savgol' and len(data) >= window_size:
                    polyorder = min(2, window_size - 1)
                    return savgol_filter(data, window_size, polyorder)
                else:
                    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
            
            # For 3D data (time x roi x roi)
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    if method == 'savgol' and len(data) >= window_size:
                        polyorder = min(2, window_size - 1)
                        smoothed[:, i, j] = savgol_filter(data[:, i, j], window_size, polyorder)
                    else:
                        smoothed[:, i, j] = np.convolve(data[:, i, j], np.ones(window_size)/window_size, mode='same')
            return smoothed
        
        def process_condition(results, condition_name):
            """Process a single condition and return baseline-corrected, smoothed mean FC and SEM."""
            fc_windows = results['group_fc_windows']
            window_centers = results['window_centers']
            individual_results = results['individual_results']
            
            # Apply baseline correction to each ROI pair time series before averaging
            baseline_period = (0, 300)  # Pre-stimulation period (TRs 0-300)
            baseline_mask = (window_centers >= baseline_period[0]) & (window_centers < baseline_period[1])
            
            # Calculate baseline mean for each ROI pair
            baseline_fc = np.mean(fc_windows[baseline_mask], axis=0)  # Shape: (n_rois, n_rois)
            
            # Subtract baseline from each time point for each ROI pair
            fc_windows_corrected = fc_windows - baseline_fc[np.newaxis, :, :]
            
            # Apply smoothing if requested
            if smooth_window and smooth_window > 1:
                fc_windows_corrected = smooth_timeseries(fc_windows_corrected, smooth_window, smooth_method)
            
            # Calculate mean across all connections from baseline-corrected data
            mean_fc = np.mean(fc_windows_corrected, axis=(1, 2))
            
            # Calculate SEM across subjects
            # First, get individual subject mean FC time series
            subject_fc_means = []
            for subj_result in individual_results:
                subj_fc_windows = subj_result['fc_windows']
                subj_window_centers = subj_result['window_centers']
                
                # Apply same baseline correction
                subj_baseline_mask = (subj_window_centers >= baseline_period[0]) & (subj_window_centers < baseline_period[1])
                subj_baseline_fc = np.mean(subj_fc_windows[subj_baseline_mask], axis=0)
                subj_fc_corrected = subj_fc_windows - subj_baseline_fc[np.newaxis, :, :]
                
                # Apply smoothing if requested
                if smooth_window and smooth_window > 1:
                    subj_fc_corrected = smooth_timeseries(subj_fc_corrected, smooth_window, smooth_method)
                
                # Calculate subject's mean FC
                subj_mean_fc = np.mean(subj_fc_corrected, axis=(1, 2))
                subject_fc_means.append(subj_mean_fc)
            
            # Calculate SEM across subjects
            subject_fc_means = np.array(subject_fc_means)  # Shape: (n_subjects, n_timepoints)
            sem_fc = np.std(subject_fc_means, axis=0) / np.sqrt(subject_fc_means.shape[0])
            
            return mean_fc, sem_fc, window_centers
        
        # Process both conditions
        active_mean_fc, active_sem_fc, active_centers = process_condition(active_results, 'ACTIVE')
        sham_mean_fc, sham_sem_fc, sham_centers = process_condition(sham_results, 'SHAM')
        
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot setup (title and complex labels removed for cleaner visualization)
        # Get seaborn default color palette
        colors = sns.color_palette()
        active_color = colors[1]  # second color for ACTIVE
        sham_color = colors[2]    # first color for SHAM
        
        # Plot ACTIVE condition with error bars (if requested)
        if 'active' in show_conditions:
            ax.plot(active_centers, active_mean_fc, color=active_color, linewidth=2, label='ACTIVE', alpha=0.8)
            ax.fill_between(active_centers, active_mean_fc - active_sem_fc, active_mean_fc + active_sem_fc, 
                           color=active_color, alpha=0.2)
        
        # Plot SHAM condition with error bars (if requested)
        if 'sham' in show_conditions:
            ax.plot(sham_centers, sham_mean_fc, color=sham_color, linewidth=2, label='SHAM', alpha=0.8)
            ax.fill_between(sham_centers, sham_mean_fc - sham_sem_fc, sham_mean_fc + sham_sem_fc, 
                           color=sham_color, alpha=0.2)
        
        # Add tFUS stimulation periods
        self._add_tfus_overlay(ax)
        
        # Set consistent y-axis limits based on both conditions (for animation consistency)
        # Calculate y-limits from both conditions' data including error bars
        all_values = []
        all_values.extend(active_mean_fc - active_sem_fc)
        all_values.extend(active_mean_fc + active_sem_fc)
        all_values.extend(sham_mean_fc - sham_sem_fc)
        all_values.extend(sham_mean_fc + sham_sem_fc)
        
        y_min = min(all_values)
        y_max = max(all_values)
        y_margin = (y_max - y_min) * 0.05  # Add 5% margin
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Formatting
        ax.set_xlabel('Time (TRs)', fontsize=12)
        ax.set_ylabel('FC', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()
        
        if save_plots:
            # Use the processing type from either condition (they should be the same)
            processing_type = active_results['processing_type']
            conditions_str = "_".join(show_conditions)
            plot_path = self.output_dir / f"consolidated_fc_timecourse_{conditions_str}_{processing_type}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Consolidated FC time course plot ({conditions_str}) saved: {plot_path}")
        
        plt.show()
        
        return fig
    
    def _add_tfus_overlay(self, ax):
        """Add tFUS stimulation period overlay to plot."""
        y_limits = ax.get_ylim()
        
        for start, end in self.tfus_periods:
            ax.axvspan(start, end, alpha=0.2, color='grey', label='tFUS' if start == self.tfus_periods[0][0] else "")
        
        # Add epoch boundaries
        ax.axvline(300, color='green', linestyle='--', alpha=0.7, label='Pre/During')
        ax.axvline(580, color='orange', linestyle='--', alpha=0.7, label='During/Post')
    
    def compute_baseline_corrected_comparison(self, active_results, sham_results):
        """
        Compute baseline-corrected comparisons between ACTIVE and SHAM conditions.
        
        Compares:
        - (active_during - active_pre) vs (sham_during - sham_pre)
        - (active_post - active_pre) vs (sham_post - sham_pre)
        
        Returns effect sizes, p-values, and significant ROI pairs.
        """
        from scipy.stats import ttest_ind
        
        print("Computing baseline-corrected statistical comparisons...")
        print("  Method: (condition_epoch - condition_pre) for ACTIVE vs SHAM")
        
        # Get individual subject data
        active_subjects = active_results['individual_results']
        sham_subjects = sham_results['individual_results']
        
        roi_columns = active_results['roi_columns']
        n_rois = len(roi_columns)
        
        # Define comparisons: each epoch vs pre-stimulation baseline
        comparisons = {
            'during_vs_pre': 'during_stimulation',
            'post_vs_pre': 'post_stimulation'
        }
        
        stats_results = {}
        
        for comparison_name, target_epoch in comparisons.items():
            print(f"  Analyzing {comparison_name}...")
            
            # Collect baseline-corrected changes for ACTIVE condition
            active_changes = []
            for subj_data in active_subjects:
                pre_fc = subj_data['epoch_fc']['pre_stimulation']
                target_fc = subj_data['epoch_fc'][target_epoch]
                
                if pre_fc is not None and target_fc is not None:
                    # Baseline correction: target - pre
                    change_matrix = target_fc - pre_fc
                    active_changes.append(change_matrix)
            
            # Collect baseline-corrected changes for SHAM condition
            sham_changes = []
            for subj_data in sham_subjects:
                pre_fc = subj_data['epoch_fc']['pre_stimulation']
                target_fc = subj_data['epoch_fc'][target_epoch]
                
                if pre_fc is not None and target_fc is not None:
                    # Baseline correction: target - pre
                    change_matrix = target_fc - pre_fc
                    sham_changes.append(change_matrix)
            
            if len(active_changes) == 0 or len(sham_changes) == 0:
                print(f"    Insufficient data for {comparison_name}")
                continue
            
            active_changes = np.array(active_changes)  # (n_subjects, n_rois, n_rois)
            sham_changes = np.array(sham_changes)
            
            print(f"    ACTIVE changes: {len(active_changes)} subjects, SHAM changes: {len(sham_changes)} subjects")
            
            # Initialize result matrices
            effect_sizes = np.zeros((n_rois, n_rois))
            p_values = np.ones((n_rois, n_rois))
            t_stats = np.zeros((n_rois, n_rois))
            
            # Compute statistics for each ROI pair
            for i in range(n_rois):
                for j in range(i, n_rois):  # Upper triangular + diagonal
                    # Extract baseline-corrected changes for this pair across subjects
                    active_changes_pair = active_changes[:, i, j]
                    sham_changes_pair = sham_changes[:, i, j]
                    
                    # Perform t-test on the changes (not absolute values)
                    t_stat, p_val = ttest_ind(active_changes_pair, sham_changes_pair)
                    
                    # Compute Cohen's d (effect size) for the changes
                    pooled_std = np.sqrt(((len(active_changes_pair) - 1) * np.var(active_changes_pair, ddof=1) + 
                                         (len(sham_changes_pair) - 1) * np.var(sham_changes_pair, ddof=1)) / 
                                        (len(active_changes_pair) + len(sham_changes_pair) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (np.mean(active_changes_pair) - np.mean(sham_changes_pair)) / pooled_std
                    else:
                        cohens_d = 0
                    
                    # Store results (make symmetric)
                    effect_sizes[i, j] = cohens_d
                    effect_sizes[j, i] = cohens_d
                    p_values[i, j] = p_val
                    p_values[j, i] = p_val
                    t_stats[i, j] = t_stat
                    t_stats[j, i] = t_stat
            
            # Store comparison results
            stats_results[comparison_name] = {
                'effect_sizes': effect_sizes,
                'p_values': p_values,
                't_statistics': t_stats,
                'n_active': len(active_changes),
                'n_sham': len(sham_changes),
                'active_mean_changes': np.mean(active_changes, axis=0),
                'sham_mean_changes': np.mean(sham_changes, axis=0),
                'difference_of_changes': np.mean(active_changes, axis=0) - np.mean(sham_changes, axis=0)
            }
        
        return stats_results
    
    def load_roi_labels(self):
        """Load ROI labels from the atlas."""
        try:
            # Try to load labels from saved file
            labels_file = self.output_dir.parent / "atlas" / "Schaefer2018_100Parcels_7Networks_labels.csv"
            if labels_file.exists():
                labels_df = pd.read_csv(labels_file)
                return labels_df
        except:
            pass
        
        # If not available, create basic labels
        print("  Warning: Detailed ROI labels not found, using generic names")
        # Determine number of ROIs dynamically based on data
        try:
            df, roi_columns = self.load_roi_data()
            n_rois = len(roi_columns)
        except:
            n_rois = 1024  # Default for DiFuMo
        
        labels_df = pd.DataFrame({
            'roi_id': range(1, n_rois + 1),
            'roi_name': [f"ROI_{i:03d}" for i in range(1, n_rois + 1)],
            'network': ['Unknown'] * n_rois
        })
        return labels_df
    
    def identify_significant_pairs(self, stats_results, alpha=0.05, roi_columns=None):
        """Identify statistically significant ROI pairs with detailed names."""
        from statsmodels.stats.multitest import multipletests
        
        # Load ROI labels
        roi_labels = self.load_roi_labels()
        
        significant_pairs = {}
        
        for epoch, stats in stats_results.items():
            p_values = stats['p_values']
            effect_sizes = stats['effect_sizes']
            
            # Extract upper triangular p-values (excluding diagonal)
            triu_indices = np.triu_indices_from(p_values, k=1)
            upper_tri_pvals = p_values[triu_indices]
            
            # Apply FDR correction
            n_comparisons = len(upper_tri_pvals)
            rejected, p_corrected, _, _ = multipletests(upper_tri_pvals, alpha=alpha, method='fdr_bh')
            
            print(f"  {epoch}: {n_comparisons} total comparisons, FDR correction applied")
            
            # Find significant pairs (uncorrected)
            sig_mask = p_values < alpha
            sig_indices = np.where(np.triu(sig_mask, k=1))  # Upper triangular, exclude diagonal
            
            pairs_list = []
            
            # Create a complete list of all pairs (including non-significant ones for FDR info)
            all_pairs_list = []
            for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
                # Get ROI details from labels
                roi_i_info = roi_labels[roi_labels['roi_id'] == i + 1].iloc[0] if i + 1 <= len(roi_labels) else None
                roi_j_info = roi_labels[roi_labels['roi_id'] == j + 1].iloc[0] if j + 1 <= len(roi_labels) else None
                
                if roi_i_info is not None and roi_j_info is not None:
                    roi_i_full_name = roi_i_info['roi_name']
                    roi_j_full_name = roi_j_info['roi_name']
                    roi_i_network = roi_i_info.get('network', 'Unknown')
                    roi_j_network = roi_j_info.get('network', 'Unknown')
                else:
                    roi_i_full_name = f"ROI_{i+1:03d}"
                    roi_j_full_name = f"ROI_{j+1:03d}"
                    roi_i_network = 'Unknown'
                    roi_j_network = 'Unknown'
                
                pair_info = {
                    'roi_i': i + 1,  # Convert to 1-based indexing
                    'roi_j': j + 1,
                    'roi_i_code': f"ROI_{i+1:03d}",
                    'roi_j_code': f"ROI_{j+1:03d}",
                    'roi_i_name': roi_i_full_name,
                    'roi_j_name': roi_j_full_name,
                    'roi_i_network': roi_i_network,
                    'roi_j_network': roi_j_network,
                    'connection_type': f"{roi_i_network} - {roi_j_network}",
                    'p_value': p_values[i, j],
                    'p_value_fdr_corrected': p_corrected[idx],
                    'significant_uncorrected': p_values[i, j] < alpha,
                    'significant_fdr_corrected': rejected[idx],
                    'effect_size': effect_sizes[i, j],
                    't_statistic': stats['t_statistics'][i, j],
                    'interpretation': 'Increased FC change (Active > Sham)' if effect_sizes[i, j] > 0 else 'Decreased FC change (Active < Sham)',
                    'effect_magnitude': 'Large' if abs(effect_sizes[i, j]) > 0.8 else 'Medium' if abs(effect_sizes[i, j]) > 0.5 else 'Small'
                }
                
                all_pairs_list.append(pair_info)
                
                # Add to significant pairs list if uncorrected p < alpha (for backward compatibility)
                if p_values[i, j] < alpha:
                    pairs_list.append(pair_info)
            
            # Sort by p-value
            pairs_list.sort(key=lambda x: x['p_value'])
            all_pairs_list.sort(key=lambda x: x['p_value'])
            
            # Count FDR-corrected significant pairs
            fdr_significant_pairs = [p for p in all_pairs_list if p['significant_fdr_corrected']]
            
            significant_pairs[epoch] = pairs_list
            
            print(f"\\n{epoch.replace('_', ' ').title()}:")
            print(f"  Total comparisons: {n_comparisons}")
            print(f"  Significant pairs (uncorrected p < {alpha}): {len(pairs_list)}")
            print(f"  Significant pairs (FDR corrected): {len(fdr_significant_pairs)}")
            
            if len(fdr_significant_pairs) > 0:
                # Show top 10 most significant FDR-corrected pairs
                print(f"  Top 10 FDR-corrected significant pairs:")
                for idx, pair in enumerate(fdr_significant_pairs[:10]):
                    roi_i_short = pair['roi_i_name'].split('_')[-1] if '_' in pair['roi_i_name'] else pair['roi_i_code']
                    roi_j_short = pair['roi_j_name'].split('_')[-1] if '_' in pair['roi_j_name'] else pair['roi_j_code']
                    print(f"    {idx+1:2d}. {pair['roi_i_code']} ({roi_i_short}) - {pair['roi_j_code']} ({roi_j_short}): "
                          f"p={pair['p_value']:.6f}, p_fdr={pair['p_value_fdr_corrected']:.6f}, d={pair['effect_size']:.3f}")
                    print(f"        Networks: {pair['connection_type']}")
                
                # Network-level summary for FDR-corrected pairs
                network_pairs = {}
                for pair in fdr_significant_pairs:
                    conn_type = pair['connection_type']
                    if conn_type not in network_pairs:
                        network_pairs[conn_type] = 0
                    network_pairs[conn_type] += 1
                
                print(f"  FDR-corrected network-level connections:")
                for conn_type, count in sorted(network_pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {conn_type}: {count} pairs")
            
            elif len(pairs_list) > 0:
                # Show uncorrected results if no FDR-corrected pairs
                print(f"  No pairs survive FDR correction. Top 5 uncorrected pairs:")
                for idx, pair in enumerate(pairs_list[:5]):
                    roi_i_short = pair['roi_i_name'].split('_')[-1] if '_' in pair['roi_i_name'] else pair['roi_i_code']
                    roi_j_short = pair['roi_j_name'].split('_')[-1] if '_' in pair['roi_j_name'] else pair['roi_j_code']
                    print(f"    {idx+1:2d}. {pair['roi_i_code']} ({roi_i_short}) - {pair['roi_j_code']} ({roi_j_short}): "
                          f"p={pair['p_value']:.6f}, p_fdr={pair['p_value_fdr_corrected']:.6f}, d={pair['effect_size']:.3f}")
            
            # Store the complete list with FDR correction info
            significant_pairs[f"{epoch}_all_pairs"] = all_pairs_list
        
        return significant_pairs
    
    def plot_baseline_corrected_comparison(self, active_results, sham_results, stats_results, alpha=0.05):
        """Create comprehensive baseline-corrected statistical comparison plots with new 3-row layout."""
        print("Creating baseline-corrected statistical comparison plots...")
        
        # Create figure with 2 rows: Active epochs, Active differences
        fig = plt.figure(figsize=(18, 12))
        
        # Create a gridspec for better control over subplot positioning
        # Row 2 uses a different grid to center the 2 plots
        gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)
        
        # Row 1: ACTIVE condition (3 plots)
        axes = np.empty((2, 3), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0:2])  # spans 2 columns
        axes[0, 1] = fig.add_subplot(gs[0, 2:4])  # spans 2 columns
        axes[0, 2] = fig.add_subplot(gs[0, 4:6])  # spans 2 columns
        
        # Row 2: ACTIVE differences (2 centered plots)
        axes[1, 0] = fig.add_subplot(gs[1, 1:3])  # spans 2 columns, offset by 1
        axes[1, 1] = fig.add_subplot(gs[1, 3:5])  # spans 2 columns, offset by 1
        axes[1, 2] = fig.add_subplot(gs[1, 5:6])  # Will be hidden later
        
        # Get epoch FC data
        active_epochs = active_results['group_epoch_fc']
        
        # Define epochs and titles
        epochs = ['pre_stimulation', 'during_stimulation', 'post_stimulation']
        epoch_titles = ['Pre-tFUS', 'During tFUS', 'Post-tFUS']
        
        # Use fixed colorbar limits for FC matrices (large dynamic range)
        fc_vmin, fc_vmax = -0.2, 1
        
        # Row 1: ACTIVE condition epochs (Pre, During, Post)
        row1_images = []
        for i, (epoch, title) in enumerate(zip(epochs, epoch_titles)):
            if epoch in active_epochs and active_epochs[epoch] is not None:
                im = axes[0, i].imshow(active_epochs[epoch], cmap='RdBu_r', vmin=fc_vmin, vmax=fc_vmax)
                axes[0, i].set_title(title, fontsize=16)
                row1_images.append(im)
            else:
                axes[0, i].set_title(f'{title} (No Data)', fontsize=16)
                axes[0, i].set_visible(False)
                row1_images.append(None)
        
        # Row 2: ACTIVE condition FC differences (during-pre and post-pre)
        active_difference_titles = ['During - Pre', 'Post - Pre']
        
        # Calculate FC differences for ACTIVE condition
        active_differences = []
        if 'during_stimulation' in active_epochs and 'pre_stimulation' in active_epochs:
            if active_epochs['during_stimulation'] is not None and active_epochs['pre_stimulation'] is not None:
                during_minus_pre = active_epochs['during_stimulation'] - active_epochs['pre_stimulation']
                active_differences.append(during_minus_pre)
            else:
                active_differences.append(None)
        else:
            active_differences.append(None)
            
        if 'post_stimulation' in active_epochs and 'pre_stimulation' in active_epochs:
            if active_epochs['post_stimulation'] is not None and active_epochs['pre_stimulation'] is not None:
                post_minus_pre = active_epochs['post_stimulation'] - active_epochs['pre_stimulation']
                active_differences.append(post_minus_pre)
            else:
                active_differences.append(None)
        else:
            active_differences.append(None)
        
        # Plot the two ACTIVE FC differences in positions 0 and 1
        row2_images = []
        for i, (difference_matrix, title) in enumerate(zip(active_differences, active_difference_titles)):
            if difference_matrix is not None:
                # Create the plot
                im = axes[1, i].imshow(difference_matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
                axes[1, i].set_title(title, fontsize=16)
                row2_images.append(im)
                
                # Add significance overlay from statistical comparisons if available
                comparison_key = 'during_vs_pre' if i == 0 else 'post_vs_pre'
                if comparison_key in stats_results:
                    stats = stats_results[comparison_key]
                    
                    # Add significance overlay - use FDR-corrected significance if available
                    try:
                        # Get all pairs data for this comparison
                        all_pairs_key = f"{comparison_key}_all_pairs"
                        if hasattr(self, '_last_significant_pairs') and all_pairs_key in self._last_significant_pairs:
                            all_pairs = self._last_significant_pairs[all_pairs_key]
                            
                            # Create significance mask using FDR correction
                            n_rois = difference_matrix.shape[0]
                            fdr_sig_mask = np.zeros((n_rois, n_rois), dtype=bool)
                            
                            for pair in all_pairs:
                                if pair['significant_fdr_corrected']:
                                    roi_i = pair['roi_i'] - 1  # Convert to 0-based
                                    roi_j = pair['roi_j'] - 1
                                    fdr_sig_mask[roi_i, roi_j] = True
                                    fdr_sig_mask[roi_j, roi_i] = True  # Make symmetric
                            
                            # Add FDR-corrected significance contours
                            if fdr_sig_mask.any():
                                axes[1, i].contour(fdr_sig_mask, levels=[0.5], colors='black', linewidths=3, alpha=0)
                                axes[1, i].contour(fdr_sig_mask, levels=[0.5], colors='yellow', linewidths=2, alpha=0)
                        
                        # Fallback to uncorrected significance
                        sig_mask = stats['p_values'] < alpha
                        axes[1, i].contour(sig_mask, levels=[0.5], colors='black', linewidths=1, alpha=0.5)
                        
                    except:
                        # Simple fallback
                        if comparison_key in stats_results:
                            sig_mask = stats_results[comparison_key]['p_values'] < alpha
                            axes[1, i].contour(sig_mask, levels=[0.5], colors='red', linewidths=2)
            else:
                axes[1, i].set_title(f'{title} (No Data)', fontsize=16)
                axes[1, i].set_visible(False)
                row2_images.append(None)
        
        # Hide the third plot in row 2
        axes[1, 2].set_visible(False)
        
        # Add colorbars outside the axes
        # FC colorbar for row 1
        if any(img is not None for img in row1_images):
            # Find first non-None image for colorbar reference
            fc_image = next(img for img in row1_images if img is not None)
            # FC matrices in row 1
            fc_axes = [axes[0, 0], axes[0, 1], axes[0, 2]]
            fc_cbar = fig.colorbar(fc_image, ax=fc_axes, location='right', shrink=0.8, label='FC')
        
        # FC change colorbar for row 2
        if any(img is not None for img in row2_images):
            change_image = next(img for img in row2_images if img is not None)
            # Only include the visible axes in row 2
            change_axes = [axes[1, 0], axes[1, 1]]
            change_cbar = fig.colorbar(change_image, ax=change_axes, location='right', shrink=0.8, label='FC Change')
        
        # Add common formatting to all visible plots
        for row in range(2):
            for col in range(3):
                if axes[row, col].get_visible():
                    # Get the actual matrix to determine n_rois
                    if row == 0 and col < len(epochs) and epochs[col] in active_epochs and active_epochs[epochs[col]] is not None:
                        n_rois = active_epochs[epochs[col]].shape[0]
                    elif row == 1 and col < len(active_differences) and active_differences[col] is not None:
                        n_rois = active_differences[col].shape[0]
                    else:
                        n_rois = 100  # Default
                    
                    # Set tick marks
                    tick_positions = [0, n_rois//4, n_rois//2, 3*n_rois//4, n_rois-1]
                    tick_labels = [str(pos+1) for pos in tick_positions]
                    
                    axes[row, col].set_xticks(tick_positions)
                    axes[row, col].set_xticklabels(tick_labels)
                    axes[row, col].set_yticks(tick_positions)
                    axes[row, col].set_yticklabels(tick_labels)
                    #axes[row, col].set_xlabel('ROI Index')
                    #axes[row, col].set_ylabel('ROI Index')
        
        # Add legend for significance overlays in row 2
        from matplotlib.patches import Patch
        #legend_elements = [
        #    #Patch(facecolor='none', edgecolor='black', linewidth=3, label='FDR-corrected significant (Active vs Sham)'),
        #    Patch(facecolor='none', edgecolor='red', linewidth=2, label='Statistically significant (Active vs Sham, p<0.05)')
        #]
        #axes[1, 0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # Save statistical comparison plot
        stats_path = self.output_dir / "baseline_corrected_statistical_comparison.png"
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Baseline-corrected statistical comparison plot saved: {stats_path}")
        
        return fig
    
    def plot_baseline_vs_change_scatter(self, active_results, sham_results, save_plots=True):
        """
        Create consolidated scatter plots showing conditional distribution of FC changes at different baseline FC values.
        
        Shows scatter plots with overlaid conditional statistics (mean, median, quartiles) to visualize how 
        FC changes are distributed as a function of baseline FC values.
        
        Parameters:
        -----------
        active_results : dict
            ACTIVE condition results
        sham_results : dict
            SHAM condition results
        save_plots : bool
            Whether to save the plots
        """
        print("Creating consolidated baseline vs. change conditional distribution plots for both conditions...")
        
        def process_condition(condition_results):
            """Process a single condition and return baseline values and changes."""
            condition_epochs = condition_results['group_epoch_fc']
            
            # Check if all required epochs are available
            required_epochs = ['pre_stimulation', 'during_stimulation', 'post_stimulation']
            if not all(epoch in condition_epochs and condition_epochs[epoch] is not None for epoch in required_epochs):
                return None, None, None
                
            # Get FC matrices
            baseline_fc = condition_epochs['pre_stimulation']
            during_fc = condition_epochs['during_stimulation']
            post_fc = condition_epochs['post_stimulation']
            
            # Extract upper triangular values (excluding diagonal)
            n_rois = baseline_fc.shape[0]
            triu_indices = np.triu_indices(n_rois, k=1)
            
            # Get FC values for each epoch
            baseline_values = baseline_fc[triu_indices]
            during_values = during_fc[triu_indices]
            post_values = post_fc[triu_indices]
            
            # Calculate changes from baseline
            during_change = during_values - baseline_values
            post_change = post_values - baseline_values
            
            return baseline_values, during_change, post_change
        
        # Process both conditions
        active_baseline, active_during_change, active_post_change = process_condition(active_results)
        sham_baseline, sham_during_change, sham_post_change = process_condition(sham_results)
        
        if active_baseline is None or sham_baseline is None:
            print("Missing required epochs for KDE plots")
            return None
        
        # Determine consistent axis limits across all plots
        all_baseline = np.concatenate([active_baseline, sham_baseline])
        all_during_change = np.concatenate([active_during_change, sham_during_change])
        all_post_change = np.concatenate([active_post_change, sham_post_change])
        
        x_min, x_max = all_baseline.min(), all_baseline.max()
        y_min = min(all_during_change.min(), all_post_change.min())
        y_max = max(all_during_change.max(), all_post_change.max())
        
        # Add small margins
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        x_lim = [x_min - x_margin, x_max + x_margin]
        y_lim = [y_min - y_margin, y_max + y_margin]
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        def add_conditional_scatter(ax, x_data, y_data, title, alpha=0.3):
            """Add scatter plot with conditional distribution visualization."""
            # Create scatter plot
            ax.scatter(x_data, y_data, alpha=alpha, s=10, color='steelblue')
            
            # Add reference lines
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.6, linewidth=1)
            
            # Add conditional distribution visualization using binning
            n_bins = 20
            x_bins = np.linspace(x_data.min(), x_data.max(), n_bins)
            bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
            
            # Calculate statistics for each bin
            bin_means = []
            bin_stds = []
            bin_medians = []
            bin_q25 = []
            bin_q75 = []
            
            for i in range(len(x_bins) - 1):
                mask = (x_data >= x_bins[i]) & (x_data < x_bins[i + 1])
                if np.sum(mask) > 5:  # Only if we have enough points
                    y_bin = y_data[mask]
                    bin_means.append(np.mean(y_bin))
                    bin_stds.append(np.std(y_bin))
                    bin_medians.append(np.median(y_bin))
                    bin_q25.append(np.percentile(y_bin, 25))
                    bin_q75.append(np.percentile(y_bin, 75))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
                    bin_medians.append(np.nan)
                    bin_q25.append(np.nan)
                    bin_q75.append(np.nan)
            
            # Convert to arrays and remove NaN values
            bin_centers = bin_centers[~np.isnan(bin_means)]
            bin_means = np.array(bin_means)[~np.isnan(bin_means)]
            bin_stds = np.array(bin_stds)[~np.isnan(bin_stds)]
            bin_medians = np.array(bin_medians)[~np.isnan(bin_medians)]
            bin_q25 = np.array(bin_q25)[~np.isnan(bin_q25)]
            bin_q75 = np.array(bin_q75)[~np.isnan(bin_q75)]
            
            # Plot conditional mean and standard deviation
            ax.plot(bin_centers, bin_means, 'r-', linewidth=3, alpha=0.8, label='Mean')
            ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, 
                           color='red', alpha=0.2, label='±1 SD')
            
            # Plot median and quartiles
            ax.plot(bin_centers, bin_medians, 'orange', linewidth=2, alpha=0.8, label='Median')
            ax.fill_between(bin_centers, bin_q25, bin_q75, 
                           color='orange', alpha=0.2, label='IQR')
            
            ax.set_xlabel('Baseline FC (Pre-stimulation)', fontsize=16)
            ax.set_ylabel('Change from Baseline', fontsize=16)
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.legend(fontsize=8)
        
        # Plot 1: ACTIVE - During change
        add_conditional_scatter(axes[0,0], active_baseline, active_during_change, 'During Stimulation')
        
        # Plot 2: ACTIVE - Post change
        add_conditional_scatter(axes[0,1], active_baseline, active_post_change, 'Post Stimulation')
        
        # Plot 3: SHAM - During change
        add_conditional_scatter(axes[1,0], sham_baseline, sham_during_change, 'During Stimulation')
        
        # Plot 4: SHAM - Post change
        add_conditional_scatter(axes[1,1], sham_baseline, sham_post_change, 'Post Stimulation')
        
        plt.tight_layout()
        
        if save_plots:
            # Use the processing type from active results (they should be the same)
            processing_type = active_results['processing_type']
            plot_path = self.output_dir / f"consolidated_baseline_vs_change_conditional_{processing_type}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Consolidated conditional distribution plot saved: {plot_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\\nSummary Statistics:")
        print(f"  ACTIVE - Total FC pairs analyzed: {len(active_baseline)}")
        print(f"  SHAM - Total FC pairs analyzed: {len(sham_baseline)}")
        print(f"  Combined baseline FC range: [{all_baseline.min():.3f}, {all_baseline.max():.3f}]")
        print(f"  Combined during change range: [{all_during_change.min():.3f}, {all_during_change.max():.3f}]")
        print(f"  Combined post change range: [{all_post_change.min():.3f}, {all_post_change.max():.3f}]")
        
        return fig
    
    def create_seed_connectivity_summary(self, active_results, sham_results, save_csv=True):
        """
        Create a summary of total connectivity changes for each seed ROI.
        
        For each ROI, calculates the sum of FC changes with all other ROIs during
        and after stimulation, for both ACTIVE and SHAM conditions.
        
        Parameters:
        -----------
        active_results : dict
            ACTIVE condition results
        sham_results : dict
            SHAM condition results
        save_csv : bool
            Whether to save the results as CSV
            
        Returns:
        --------
        pd.DataFrame
            Summary dataframe with connectivity changes for each seed ROI
        """
        print("Creating seed connectivity summary...")
        
        # Get epoch FC data for both conditions
        active_epochs = active_results['group_epoch_fc']
        sham_epochs = sham_results['group_epoch_fc']
        
        # Check if all required epochs are available
        required_epochs = ['pre_stimulation', 'during_stimulation', 'post_stimulation']
        
        if not all(epoch in active_epochs and active_epochs[epoch] is not None for epoch in required_epochs):
            print("Missing required epochs for ACTIVE condition")
            return None
            
        if not all(epoch in sham_epochs and sham_epochs[epoch] is not None for epoch in required_epochs):
            print("Missing required epochs for SHAM condition")
            return None
        
        # Get ROI information
        roi_columns = active_results['roi_columns']
        roi_labels = self.load_roi_labels()
        n_rois = len(roi_columns)
        
        print(f"  Processing {n_rois} ROIs...")
        
        # Initialize results list
        results = []
        
        # Process each ROI as a seed
        for seed_roi in range(n_rois):
            # Get ROI information
            roi_info = roi_labels[roi_labels['roi_id'] == seed_roi + 1].iloc[0] if seed_roi + 1 <= len(roi_labels) else None
            
            if roi_info is not None:
                roi_name = roi_info['roi_name']
                roi_network = roi_info.get('network', 'Unknown')
            else:
                roi_name = f"ROI_{seed_roi+1:03d}"
                roi_network = 'Unknown'
            
            # ACTIVE condition calculations
            active_pre = active_epochs['pre_stimulation']
            active_during = active_epochs['during_stimulation']
            active_post = active_epochs['post_stimulation']
            
            # Calculate changes from baseline for ACTIVE
            active_during_change = active_during - active_pre
            active_post_change = active_post - active_pre
            
            # Sum connectivity changes for this seed ROI (excluding self-connection)
            # Sum across all other ROIs (both row and column to get full connectivity)
            active_during_total = (np.sum(active_during_change[seed_roi, :]) + 
                                 np.sum(active_during_change[:, seed_roi]) - 
                                 active_during_change[seed_roi, seed_roi])  # Subtract diagonal to avoid double counting
            
            active_post_total = (np.sum(active_post_change[seed_roi, :]) + 
                               np.sum(active_post_change[:, seed_roi]) - 
                               active_post_change[seed_roi, seed_roi])
            
            # SHAM condition calculations
            sham_pre = sham_epochs['pre_stimulation']
            sham_during = sham_epochs['during_stimulation']
            sham_post = sham_epochs['post_stimulation']
            
            # Calculate changes from baseline for SHAM
            sham_during_change = sham_during - sham_pre
            sham_post_change = sham_post - sham_pre
            
            # Sum connectivity changes for this seed ROI
            sham_during_total = (np.sum(sham_during_change[seed_roi, :]) + 
                               np.sum(sham_during_change[:, seed_roi]) - 
                               sham_during_change[seed_roi, seed_roi])
            
            sham_post_total = (np.sum(sham_post_change[seed_roi, :]) + 
                             np.sum(sham_post_change[:, seed_roi]) - 
                             sham_post_change[seed_roi, seed_roi])
            
            # Calculate baseline connectivity strength for context
            active_pre_total = (np.sum(active_pre[seed_roi, :]) + 
                              np.sum(active_pre[:, seed_roi]) - 
                              active_pre[seed_roi, seed_roi])
            
            sham_pre_total = (np.sum(sham_pre[seed_roi, :]) + 
                            np.sum(sham_pre[:, seed_roi]) - 
                            sham_pre[seed_roi, seed_roi])
            
            # Calculate differences between conditions
            during_diff = active_during_total - sham_during_total
            post_diff = active_post_total - sham_post_total
            
            # Store results
            results.append({
                'roi_id': seed_roi + 1,
                'roi_code': f"ROI_{seed_roi+1:03d}",
                'roi_name': roi_name,
                'network': roi_network,
                'active_baseline_total': active_pre_total,
                'sham_baseline_total': sham_pre_total,
                'active_during_change': active_during_total,
                'active_post_change': active_post_total,
                'sham_during_change': sham_during_total,
                'sham_post_change': sham_post_total,
                'during_difference': during_diff,  # Active - Sham
                'post_difference': post_diff,      # Active - Sham
                'active_during_percent': (active_during_total / active_pre_total * 100) if active_pre_total != 0 else 0,
                'active_post_percent': (active_post_total / active_pre_total * 100) if active_pre_total != 0 else 0,
                'sham_during_percent': (sham_during_total / sham_pre_total * 100) if sham_pre_total != 0 else 0,
                'sham_post_percent': (sham_post_total / sham_pre_total * 100) if sham_pre_total != 0 else 0
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by network and then by ROI ID
        df = df.sort_values(['network', 'roi_id']).reset_index(drop=True)
        
        # Save to CSV if requested
        if save_csv:
            csv_path = self.output_dir / "seed_connectivity_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Seed connectivity summary saved: {csv_path}")
        
        # Print summary statistics
        print(f"\\nSeed Connectivity Summary Statistics:")
        print(f"  Total ROIs processed: {len(df)}")
        print(f"  Networks: {df['network'].nunique()} ({', '.join(df['network'].unique())})")
        
        # Top changes during stimulation
        print(f"\\nTop 5 ROIs with largest ACTIVE during changes:")
        top_active_during = df.nlargest(5, 'active_during_change')
        for _, row in top_active_during.iterrows():
            print(f"  {row['roi_code']} ({row['network']}): {row['active_during_change']:.4f}")
        
        print(f"\\nTop 5 ROIs with largest difference (Active - Sham) during stimulation:")
        top_diff_during = df.nlargest(5, 'during_difference')
        for _, row in top_diff_during.iterrows():
            print(f"  {row['roi_code']} ({row['network']}): {row['during_difference']:.4f}")
        
        # Network-level summary
        print(f"\\nNetwork-level averages:")
        network_summary = df.groupby('network').agg({
            'active_during_change': 'mean',
            'active_post_change': 'mean',
            'during_difference': 'mean',
            'post_difference': 'mean'
        }).round(4)
        
        for network, row in network_summary.iterrows():
            print(f"  {network}: During={row['during_difference']:.4f}, Post={row['post_difference']:.4f}")
        
        return df
    
    def plot_top_significant_timeseries(self, active_results, sham_results, significant_pairs, n_top=10, smooth_window=5, smooth_method='moving_average'):
        """
        Plot sliding window FC time series for top N most significant ROI pairs.
        
        Parameters:
        -----------
        active_results : dict
            ACTIVE condition results
        sham_results : dict  
            SHAM condition results
        significant_pairs : dict
            Dictionary of significant pairs by epoch
        n_top : int
            Number of top pairs to plot
        smooth_window : int, optional
            Size of smoothing window. Default is 5.
            Set to 0 or None to disable smoothing.
        smooth_method : str, optional
            Smoothing method ('moving_average' or 'savgol'). Default is 'moving_average'.
        """
        print(f"Creating sliding window time series plots for top {n_top} significant ROI pairs...")
        
        def smooth_timeseries(data, window_size, method='moving_average'):
            """Apply smoothing to time series data."""
            if window_size is None or window_size <= 1:
                return data
            
            if method == 'savgol' and len(data) >= window_size:
                # Savitzky-Golay filter (polynomial order 2)
                polyorder = min(2, window_size - 1)
                return savgol_filter(data, window_size, polyorder)
            else:
                # Moving average
                return np.convolve(data, np.ones(window_size)/window_size, mode='same')
        
        # Get the most significant pairs from 'during_vs_pre' comparison (most relevant for tFUS effects)
        if 'during_vs_pre' not in significant_pairs or len(significant_pairs['during_vs_pre']) == 0:
            print("No significant pairs found for during_vs_pre comparison")
            return None
            
        top_pairs = significant_pairs['during_vs_pre'][:n_top]
        
        # Create figure with subplots
        n_cols = 2
        n_rows = (n_top + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, pair in enumerate(top_pairs):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            roi_i = pair['roi_i'] - 1  # Convert to 0-based indexing
            roi_j = pair['roi_j'] - 1
            
            # Extract FC time series for this pair from both conditions
            active_fc_windows = active_results['group_fc_windows']
            sham_fc_windows = sham_results['group_fc_windows']
            active_centers = active_results['window_centers']
            sham_centers = sham_results['window_centers']
            
            # Apply baseline correction to each condition separately
            baseline_period = (0, 300)  # Pre-stimulation period (TRs 0-300)
            
            # ACTIVE condition baseline correction
            active_baseline_mask = (active_centers >= baseline_period[0]) & (active_centers < baseline_period[1])
            active_baseline_fc = np.mean(active_fc_windows[active_baseline_mask], axis=0)
            active_fc_corrected = active_fc_windows - active_baseline_fc[np.newaxis, :, :]
            
            # SHAM condition baseline correction
            sham_baseline_mask = (sham_centers >= baseline_period[0]) & (sham_centers < baseline_period[1])
            sham_baseline_fc = np.mean(sham_fc_windows[sham_baseline_mask], axis=0)
            sham_fc_corrected = sham_fc_windows - sham_baseline_fc[np.newaxis, :, :]
            
            # Get baseline-corrected FC time series for this specific pair
            active_pair_fc = active_fc_corrected[:, roi_i, roi_j]
            sham_pair_fc = sham_fc_corrected[:, roi_i, roi_j]
            
            # Apply smoothing if requested
            if smooth_window and smooth_window > 1:
                active_pair_fc = smooth_timeseries(active_pair_fc, smooth_window, smooth_method)
                sham_pair_fc = smooth_timeseries(sham_pair_fc, smooth_window, smooth_method)
            
            # Plot both conditions
            smooth_text = f", smoothed ({smooth_method}, window={smooth_window})" if smooth_window and smooth_window > 1 else ""
            ax.plot(active_centers, active_pair_fc, 'r-', linewidth=2, label=f'ACTIVE (baseline corrected{smooth_text})', alpha=0.8)
            ax.plot(sham_centers, sham_pair_fc, 'b-', linewidth=2, label=f'SHAM (baseline corrected{smooth_text})', alpha=0.8)
            
            # Add tFUS stimulation periods
            self._add_tfus_overlay(ax)
            
            # Formatting
            roi_i_short = pair['roi_i_name'].split('_')[-1] if '_' in pair['roi_i_name'] else pair['roi_i_code']
            roi_j_short = pair['roi_j_name'].split('_')[-1] if '_' in pair['roi_j_name'] else pair['roi_j_code']
            
            ax.set_title(f"{idx+1}. {pair['roi_i_code']} - {pair['roi_j_code']}\\n"
                        f"{roi_i_short} - {roi_j_short}\\n"
                        f"p={pair['p_value']:.4f}, d={pair['effect_size']:.3f}", 
                        fontsize=10)
            ax.set_xlabel('Time (TRs)')
            ax.set_ylabel(f'FC (baseline corrected{smooth_text})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1, 1])
            
            # Add effect magnitude annotation
            ax.text(0.02, 0.95, f"{pair['effect_magnitude']} effect", 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5),
                   verticalalignment='top')
        
        # Hide empty subplots if n_top is odd
        if n_top % 2 == 1 and n_top < n_rows * n_cols:
            axes[-1, -1].set_visible(False)
        
        smooth_title_text = f", Smoothed ({smooth_method}, window={smooth_window})" if smooth_window and smooth_window > 1 else ""
        plt.suptitle(f'Top {n_top} Significant ROI Pairs: Baseline-Corrected FC Time Series{smooth_title_text}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"top_{n_top}_significant_pairs_timeseries.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Top {n_top} significant pairs time series plot saved: {plot_path}")
        
        # Also create a summary statistics table
        self._create_timeseries_summary_table(active_results, sham_results, top_pairs)
        
        return fig
    
    def _create_timeseries_summary_table(self, active_results, sham_results, top_pairs):
        """Create summary statistics table for the top significant pairs."""
        
        summary_data = []
        
        for pair in top_pairs:
            roi_i = pair['roi_i'] - 1  # Convert to 0-based indexing
            roi_j = pair['roi_j'] - 1
            
            # Extract FC values for different epochs
            active_pre = np.mean([subj['epoch_fc']['pre_stimulation'][roi_i, roi_j] 
                                for subj in active_results['individual_results'] 
                                if subj['epoch_fc']['pre_stimulation'] is not None])
            
            active_during = np.mean([subj['epoch_fc']['during_stimulation'][roi_i, roi_j] 
                                   for subj in active_results['individual_results'] 
                                   if subj['epoch_fc']['during_stimulation'] is not None])
            
            active_post = np.mean([subj['epoch_fc']['post_stimulation'][roi_i, roi_j] 
                                 for subj in active_results['individual_results'] 
                                 if subj['epoch_fc']['post_stimulation'] is not None])
            
            sham_pre = np.mean([subj['epoch_fc']['pre_stimulation'][roi_i, roi_j] 
                              for subj in sham_results['individual_results'] 
                              if subj['epoch_fc']['pre_stimulation'] is not None])
            
            sham_during = np.mean([subj['epoch_fc']['during_stimulation'][roi_i, roi_j] 
                                 for subj in sham_results['individual_results'] 
                                 if subj['epoch_fc']['during_stimulation'] is not None])
            
            sham_post = np.mean([subj['epoch_fc']['post_stimulation'][roi_i, roi_j] 
                               for subj in sham_results['individual_results'] 
                               if subj['epoch_fc']['post_stimulation'] is not None])
            
            # Calculate baseline-corrected changes
            active_change_during = active_during - active_pre
            active_change_post = active_post - active_pre
            sham_change_during = sham_during - sham_pre
            sham_change_post = sham_post - sham_pre
            
            summary_data.append({
                'roi_pair': f"{pair['roi_i_code']} - {pair['roi_j_code']}",
                'roi_names': f"{pair['roi_i_name']} - {pair['roi_j_name']}",
                'networks': pair['connection_type'],
                'p_value': pair['p_value'],
                'effect_size': pair['effect_size'],
                'active_pre': active_pre,
                'active_during': active_during,
                'active_post': active_post,
                'sham_pre': sham_pre,
                'sham_during': sham_during,
                'sham_post': sham_post,
                'active_change_during': active_change_during,
                'active_change_post': active_change_post,
                'sham_change_during': sham_change_during,
                'sham_change_post': sham_change_post,
                'diff_change_during': active_change_during - sham_change_during,
                'diff_change_post': active_change_post - sham_change_post
            })
        
        # Save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "top_significant_pairs_summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary statistics saved: {summary_path}")
        
        # Print key findings
        print("\\nKey findings from top significant pairs:")
        for i, row in summary_df.head(5).iterrows():
            print(f"  {i+1}. {row['roi_pair']}: Δ(Active-Sham) during = {row['diff_change_during']:.3f}")
    
    def save_statistical_results(self, stats_results, significant_pairs):
        """Save statistical results to files."""
        
        # Save stats matrices
        for epoch, stats in stats_results.items():
            epoch_dir = self.output_dir / "statistical_results"
            epoch_dir.mkdir(exist_ok=True)
            
            # Save as numpy arrays
            np.save(epoch_dir / f"{epoch}_effect_sizes.npy", stats['effect_sizes'])
            np.save(epoch_dir / f"{epoch}_p_values.npy", stats['p_values'])
            np.save(epoch_dir / f"{epoch}_t_statistics.npy", stats['t_statistics'])
        
        # Save significant pairs as CSV (uncorrected significant pairs only)
        for epoch, pairs in significant_pairs.items():
            if pairs and not epoch.endswith('_all_pairs'):
                pairs_df = pd.DataFrame(pairs)
                csv_path = self.output_dir / f"significant_pairs_{epoch}.csv"
                pairs_df.to_csv(csv_path, index=False)
                print(f"Significant pairs (uncorrected) saved: {csv_path}")
        
        # Save complete pairs information with FDR correction
        for epoch, pairs in significant_pairs.items():
            if pairs and epoch.endswith('_all_pairs'):
                # Extract base epoch name
                base_epoch = epoch.replace('_all_pairs', '')
                
                pairs_df = pd.DataFrame(pairs)
                csv_path = self.output_dir / f"all_pairs_with_fdr_{base_epoch}.csv"
                pairs_df.to_csv(csv_path, index=False)
                print(f"All pairs with FDR correction saved: {csv_path}")
                
                # Also save only FDR-corrected significant pairs
                fdr_significant = [p for p in pairs if p['significant_fdr_corrected']]
                if fdr_significant:
                    fdr_df = pd.DataFrame(fdr_significant)
                    fdr_csv_path = self.output_dir / f"fdr_significant_pairs_{base_epoch}.csv"
                    fdr_df.to_csv(fdr_csv_path, index=False)
                    print(f"FDR-corrected significant pairs saved: {fdr_csv_path} ({len(fdr_significant)} pairs)")
                else:
                    print(f"No FDR-corrected significant pairs found for {base_epoch}")
    
    def compare_conditions(self, active_results, sham_results):
        """Enhanced comparison with baseline-corrected statistical analysis."""
        print("\\n=== Comparing ACTIVE vs SHAM conditions (Baseline-Corrected) ===")
        print("Comparing baseline-corrected changes:")
        print("  - (Active During - Active Pre) vs (Sham During - Sham Pre)")
        print("  - (Active Post - Active Pre) vs (Sham Post - Sham Pre)")
        
        # Compute baseline-corrected statistical comparisons
        stats_results = self.compute_baseline_corrected_comparison(active_results, sham_results)
        
        # Identify significant pairs
        roi_columns = active_results['roi_columns']
        significant_pairs = self.identify_significant_pairs(stats_results, alpha=0.05, roi_columns=roi_columns)
        
        # Store significant pairs data for plotting
        self._last_significant_pairs = significant_pairs
        
        # Create statistical comparison plots
        self.plot_baseline_corrected_comparison(active_results, sham_results, stats_results)
        
        # Create consolidated baseline vs change KDE plots for both conditions
        self.plot_baseline_vs_change_scatter(active_results, sham_results)
        
        # Create seed connectivity summary for all ROIs
        seed_connectivity_df = self.create_seed_connectivity_summary(active_results, sham_results)
        
        # Create time series plots for top significant pairs
        self.plot_top_significant_timeseries(active_results, sham_results, significant_pairs, n_top=10)
        
        # Save results
        self.save_statistical_results(stats_results, significant_pairs)
        
        return stats_results, significant_pairs


def main():
    """Main analysis pipeline."""
    
    # Initialize analyzer
    roi_data_dir = "/Users/jacekdmochowski/PROJECTS/fus/data/roi_time_series"
    #roi_data_dir = "/Users/jacekdmochowski/PROJECTS/fus/data/roi_time_series_difumo"
    analyzer = SlidingWindowFCAnalyzer(roi_data_dir)
    
    # Analyze ACTIVE condition (limit to 4 subjects for memory management)
    print("\\n" + "="*60)
    #active_results = analyzer.analyze_condition('ACTIVE', processing_type='difumo1024', max_subjects=4)
    active_results = analyzer.analyze_condition('ACTIVE', processing_type='conservative')

    # Analyze SHAM condition (limit to 4 subjects for memory management)
    print("\\n" + "="*60)
    #sham_results = analyzer.analyze_condition('SHAM', processing_type='difumo1024', max_subjects=4)
    sham_results = analyzer.analyze_condition('SHAM', processing_type='conservative')
    
    # Plot consolidated time course - both versions for animation
    if active_results and sham_results:
        # Generate both conditions (full plot)
        analyzer.plot_fc_timecourse(active_results, sham_results, show_conditions=['active', 'sham'])
        
        # Generate SHAM only (for animation)
        analyzer.plot_fc_timecourse(active_results, sham_results, show_conditions=['sham'])
    
    # Compare conditions
    if active_results and sham_results:
        analyzer.compare_conditions(active_results, sham_results)
    
    return active_results, sham_results


if __name__ == "__main__":
    active_results, sham_results = main()