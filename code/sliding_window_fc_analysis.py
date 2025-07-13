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
        self.window_length = 20  # TRs (20 seconds)
        self.overlap_percent = 80  # 75% overlap
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
        
        # Initialize FC matrix array
        fc_windows = np.zeros((n_windows, n_rois, n_rois))
        
        # Compute FC for each window
        for i, start in enumerate(window_starts):
            end = start + self.window_length
            window_data = timeseries_data[start:end, :]
            
            # Compute correlation matrix
            fc_matrix = np.corrcoef(window_data.T)
            
            # Handle NaN values (replace with 0)
            fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
            
            fc_windows[i] = fc_matrix
        
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
    
    def analyze_condition(self, condition, processing_type='minimal'):
        """
        Analyze all subjects for a specific condition (ACTIVE or SHAM).
        
        Parameters:
        -----------
        condition : str
            'ACTIVE' or 'SHAM'
        processing_type : str
            'minimal' or 'conservative'
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
        
        # Average sliding window FC across subjects
        truncated_fc_windows = [fc[:min_windows] for fc in all_fc_windows]
        group_fc_windows = np.mean(truncated_fc_windows, axis=0)
        
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
    
    def plot_fc_timecourse(self, group_results, roi_pairs=None, save_plots=True):
        """
        Plot FC time course with tFUS periods overlay.
        
        Parameters:
        -----------
        group_results : dict
            Group analysis results
        roi_pairs : list of tuples, optional
            Specific ROI pairs to plot. If None, plots summary measures.
        """
        condition = group_results['condition']
        fc_windows = group_results['group_fc_windows']
        window_centers = group_results['window_centers']
        
        print(f"Plotting FC time course for {condition}...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Mean FC across all connections
        mean_fc = np.mean(fc_windows, axis=(1, 2))
        axes[0].plot(window_centers, mean_fc, 'b-', linewidth=2, label='Mean FC')
        self._add_tfus_overlay(axes[0])
        axes[0].set_title(f'{condition}: Mean FC Across All Connections')
        axes[0].set_xlabel('Time (TRs)')
        axes[0].set_ylabel('Mean FC')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: FC variability
        std_fc = np.std(fc_windows, axis=(1, 2))
        axes[1].plot(window_centers, std_fc, 'r-', linewidth=2, label='FC Variability')
        self._add_tfus_overlay(axes[1])
        axes[1].set_title(f'{condition}: FC Variability')
        axes[1].set_xlabel('Time (TRs)')
        axes[1].set_ylabel('FC Standard Deviation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Within-network FC (if we can identify networks)
        # For now, plot a sample of specific connections
        if roi_pairs is None:
            # Plot a few representative connections
            roi_pairs = [(0, 1), (10, 20), (30, 40), (50, 60)]
        
        for i, (roi1, roi2) in enumerate(roi_pairs[:4]):
            if roi1 < fc_windows.shape[1] and roi2 < fc_windows.shape[2]:
                fc_pair = fc_windows[:, roi1, roi2]
                axes[2].plot(window_centers, fc_pair, linewidth=1.5, 
                           label=f'ROI {roi1+1}-{roi2+1}', alpha=0.8)
        
        self._add_tfus_overlay(axes[2])
        axes[2].set_title(f'{condition}: Sample ROI Pair Connections')
        axes[2].set_xlabel('Time (TRs)')
        axes[2].set_ylabel('FC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: FC matrix at different epochs
        epochs_to_plot = ['pre_stimulation', 'during_stimulation', 'post_stimulation']
        epoch_fc = group_results['group_epoch_fc']
        
        for i, epoch in enumerate(epochs_to_plot):
            if epoch in epoch_fc and epoch_fc[epoch] is not None:
                # Compute mean FC for this epoch
                epoch_mean = np.mean(epoch_fc[epoch][np.triu_indices_from(epoch_fc[epoch], k=1)])
                axes[3].bar(i, epoch_mean, alpha=0.7, label=epoch.replace('_', ' ').title())
        
        axes[3].set_title(f'{condition}: Mean FC by Epoch')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Mean FC')
        axes[3].set_xticks(range(len(epochs_to_plot)))
        axes[3].set_xticklabels([e.replace('_', ' ').title() for e in epochs_to_plot], rotation=45)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            condition_dir = self.output_dir / f"{condition.lower()}_{group_results['processing_type']}"
            plot_path = condition_dir / "fc_timecourse_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Plot saved: {plot_path}")
        
        plt.show()
        
        return fig
    
    def _add_tfus_overlay(self, ax):
        """Add tFUS stimulation period overlay to plot."""
        y_limits = ax.get_ylim()
        
        for start, end in self.tfus_periods:
            ax.axvspan(start, end, alpha=0.2, color='red', label='tFUS' if start == self.tfus_periods[0][0] else "")
        
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
        labels_df = pd.DataFrame({
            'roi_id': range(1, 101),
            'roi_name': [f"ROI_{i:03d}" for i in range(1, 101)],
            'network': ['Unknown'] * 100
        })
        return labels_df
    
    def identify_significant_pairs(self, stats_results, alpha=0.05):
        """Identify statistically significant ROI pairs with detailed names."""
        
        # Load ROI labels
        roi_labels = self.load_roi_labels()
        
        significant_pairs = {}
        
        for epoch, stats in stats_results.items():
            p_values = stats['p_values']
            effect_sizes = stats['effect_sizes']
            
            # Find significant pairs (uncorrected)
            sig_mask = p_values < alpha
            sig_indices = np.where(np.triu(sig_mask, k=1))  # Upper triangular, exclude diagonal
            
            pairs_list = []
            for i, j in zip(sig_indices[0], sig_indices[1]):
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
                
                pairs_list.append({
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
                    'effect_size': effect_sizes[i, j],
                    't_statistic': stats['t_statistics'][i, j],
                    'interpretation': 'Increased FC change (Active > Sham)' if effect_sizes[i, j] > 0 else 'Decreased FC change (Active < Sham)',
                    'effect_magnitude': 'Large' if abs(effect_sizes[i, j]) > 0.8 else 'Medium' if abs(effect_sizes[i, j]) > 0.5 else 'Small'
                })
            
            # Sort by p-value
            pairs_list.sort(key=lambda x: x['p_value'])
            
            significant_pairs[epoch] = pairs_list
            
            print(f"\\n{epoch.replace('_', ' ').title()}:")
            print(f"  Total significant pairs (p < {alpha}): {len(pairs_list)}")
            
            if len(pairs_list) > 0:
                # Show top 10 most significant with names
                print(f"  Top 10 most significant pairs:")
                for idx, pair in enumerate(pairs_list[:10]):
                    roi_i_short = pair['roi_i_name'].split('_')[-1] if '_' in pair['roi_i_name'] else pair['roi_i_code']
                    roi_j_short = pair['roi_j_name'].split('_')[-1] if '_' in pair['roi_j_name'] else pair['roi_j_code']
                    print(f"    {idx+1:2d}. {pair['roi_i_code']} ({roi_i_short}) - {pair['roi_j_code']} ({roi_j_short}): "
                          f"p={pair['p_value']:.6f}, d={pair['effect_size']:.3f} ({pair['effect_magnitude']} effect)")
                    print(f"        Networks: {pair['connection_type']}")
                
                # Check for lower-right quadrant (ROIs 80-100)
                lr_quadrant_pairs = [p for p in pairs_list 
                                   if p['roi_i'] >= 80 and p['roi_j'] >= 80]
                if lr_quadrant_pairs:
                    print(f"  Significant pairs in lower-right quadrant (ROIs 80-100): {len(lr_quadrant_pairs)}")
                    
                # Network-level summary
                network_pairs = {}
                for pair in pairs_list:
                    conn_type = pair['connection_type']
                    if conn_type not in network_pairs:
                        network_pairs[conn_type] = 0
                    network_pairs[conn_type] += 1
                
                print(f"  Network-level connections:")
                for conn_type, count in sorted(network_pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {conn_type}: {count} pairs")
        
        return significant_pairs
    
    def plot_baseline_corrected_comparison(self, active_results, sham_results, stats_results, alpha=0.05):
        """Create comprehensive baseline-corrected statistical comparison plots."""
        print("Creating baseline-corrected statistical comparison plots...")
        
        comparisons = ['during_vs_pre', 'post_vs_pre']
        comparison_titles = ['During vs Pre', 'Post vs Pre']
        
        # Create main comparison figure (4 x 2: changes, differences, effect sizes, p-values)
        fig, axes = plt.subplots(4, 2, figsize=(12, 20))
        
        for i, (comparison, title) in enumerate(zip(comparisons, comparison_titles)):
            if comparison not in stats_results:
                continue
                
            stats = stats_results[comparison]
            
            # Row 1: ACTIVE changes (from baseline)
            im1 = axes[0, i].imshow(stats['active_mean_changes'], cmap='RdBu_r', vmin=-0.3, vmax=0.3)
            axes[0, i].set_title(f'ACTIVE Changes: {title}')
            if i == 1:  # Add colorbar to last plot
                plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Row 2: Difference of changes [(Active-Pre) - (Sham-Pre)]
            im2 = axes[1, i].imshow(stats['difference_of_changes'], cmap='RdBu_r', vmin=-0.2, vmax=0.2)
            axes[1, i].set_title(f'Difference of Changes: {title}')
            if i == 1:
                plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # Row 3: Effect sizes (Cohen's d)
            im3 = axes[2, i].imshow(stats['effect_sizes'], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[2, i].set_title(f'Effect Size (Cohens d): {title}')
            if i == 1:
                plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
            
            # Row 4: P-values (log scale, with significance overlay)
            p_vals_log = -np.log10(stats['p_values'] + 1e-10)  # Add small value to avoid log(0)
            im4 = axes[3, i].imshow(p_vals_log, cmap='viridis', vmin=0, vmax=3)
            
            # Overlay significance mask
            sig_mask = stats['p_values'] < alpha
            axes[3, i].contour(sig_mask, levels=[0.5], colors='red', linewidths=2)
            
            axes[3, i].set_title(f'P-values (-log10): {title}')
            if i == 1:
                cbar = plt.colorbar(im4, ax=axes[3, i], fraction=0.046, pad=0.04)
                cbar.set_label('-log10(p-value)')
            
            # Add ROI tick marks and highlights for all rows
            for row in range(4):
                axes[row, i].set_xticks([0, 25, 50, 75, 99])
                axes[row, i].set_xticklabels(['1', '25', '50', '75', '100'])
                axes[row, i].set_yticks([0, 25, 50, 75, 99])
                axes[row, i].set_yticklabels(['1', '25', '50', '75', '100'])
                
                # Highlight lower-right quadrant (ROIs 80-100)
                axes[row, i].add_patch(plt.Rectangle((79, 79), 20, 20, 
                                                   fill=False, edgecolor='yellow', linewidth=2))
        
        plt.tight_layout()
        
        # Save statistical comparison plot
        stats_path = self.output_dir / "baseline_corrected_statistical_comparison.png"
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Baseline-corrected statistical comparison plot saved: {stats_path}")
        
        return fig
    
    def plot_top_significant_timeseries(self, active_results, sham_results, significant_pairs, n_top=10):
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
        """
        print(f"Creating sliding window time series plots for top {n_top} significant ROI pairs...")
        
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
            
            # Get FC time series for this specific pair
            active_pair_fc = active_fc_windows[:, roi_i, roi_j]
            sham_pair_fc = sham_fc_windows[:, roi_i, roi_j]
            
            # Plot both conditions
            ax.plot(active_centers, active_pair_fc, 'r-', linewidth=2, label='ACTIVE', alpha=0.8)
            ax.plot(sham_centers, sham_pair_fc, 'b-', linewidth=2, label='SHAM', alpha=0.8)
            
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
            ax.set_ylabel('FC')
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
        
        # Save significant pairs as CSV
        for epoch, pairs in significant_pairs.items():
            if pairs:
                pairs_df = pd.DataFrame(pairs)
                csv_path = self.output_dir / f"significant_pairs_{epoch}.csv"
                pairs_df.to_csv(csv_path, index=False)
                print(f"Significant pairs saved: {csv_path}")
    
    def compare_conditions(self, active_results, sham_results):
        """Enhanced comparison with baseline-corrected statistical analysis."""
        print("\\n=== Comparing ACTIVE vs SHAM conditions (Baseline-Corrected) ===")
        print("Comparing baseline-corrected changes:")
        print("  - (Active During - Active Pre) vs (Sham During - Sham Pre)")
        print("  - (Active Post - Active Pre) vs (Sham Post - Sham Pre)")
        
        # Compute baseline-corrected statistical comparisons
        stats_results = self.compute_baseline_corrected_comparison(active_results, sham_results)
        
        # Identify significant pairs
        significant_pairs = self.identify_significant_pairs(stats_results, alpha=0.05)
        
        # Create statistical comparison plots
        self.plot_baseline_corrected_comparison(active_results, sham_results, stats_results)
        
        # Create time series plots for top significant pairs
        self.plot_top_significant_timeseries(active_results, sham_results, significant_pairs, n_top=10)
        
        # Save results
        self.save_statistical_results(stats_results, significant_pairs)
        
        return stats_results, significant_pairs


def main():
    """Main analysis pipeline."""
    
    # Initialize analyzer
    roi_data_dir = "/Users/jacekdmochowski/PROJECTS/fus/data/roi_time_series"
    analyzer = SlidingWindowFCAnalyzer(roi_data_dir)
    
    # Analyze ACTIVE condition
    print("\\n" + "="*60)
    active_results = analyzer.analyze_condition('ACTIVE', processing_type='conservative')
    
    # Analyze SHAM condition  
    print("\\n" + "="*60)
    sham_results = analyzer.analyze_condition('SHAM', processing_type='conservative')
    
    # Plot time courses
    if active_results:
        analyzer.plot_fc_timecourse(active_results)
    
    if sham_results:
        analyzer.plot_fc_timecourse(sham_results)
    
    # Compare conditions
    if active_results and sham_results:
        analyzer.compare_conditions(active_results, sham_results)
    
    return active_results, sham_results


if __name__ == "__main__":
    active_results, sham_results = main()