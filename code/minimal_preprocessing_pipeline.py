"""
Minimal preprocessing pipeline for tFUS fMRI data.

This pipeline applies only essential preprocessing steps to preserve tFUS-induced signals:
1. Quality control assessment
2. Minimal motion parameter regression
3. Gentle high-pass filtering
4. Optional conservative nuisance regression

Author: Generated for tFUS project
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class MinimalPreprocessor:
    """
    Conservative preprocessing for tFUS fMRI data.
    """
    
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        data_dir : str or Path
            Path to resampled_bold_flywheel directory
        output_dir : str or Path, optional
            Output directory for processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir.parent / "processed_bold"
        self.output_dir.mkdir(exist_ok=True)
        
        # Quality control thresholds
        self.fd_threshold = 0.5  # mm
        self.dvars_threshold = 1.5  # standardized units
        self.high_pass_cutoff = 0.008  # Hz (125s)
        
    def get_subject_sessions(self):
        """Get all subject-session combinations."""
        sessions = []
        for folder in self.data_dir.glob("resampled_bold_sub-*"):
            if folder.is_dir():
                parts = folder.name.split('_')
                subject = parts[2].split('-')[1]
                session = parts[3].split('-')[1]
                sessions.append((subject, session, folder))
        return sessions
    
    def load_bold_data(self, session_folder):
        """Load BOLD data and confounds for a session."""
        # Find the resampled BOLD file
        bold_files = list(session_folder.glob("output/*_desc-preproc_bold_resampled.nii.gz"))
        if not bold_files:
            raise FileNotFoundError(f"No resampled BOLD file found in {session_folder}")
        
        bold_file = bold_files[0]
        bold_img = nib.load(bold_file)
        bold_data = bold_img.get_fdata()
        
        # Find confounds file in input directory
        input_dirs = list(session_folder.glob("input/*"))
        confounds_file = None
        for input_dir in input_dirs:
            if input_dir.is_dir():
                confounds_files = list(input_dir.glob("**/func/*_desc-confounds_timeseries.tsv"))
                if confounds_files:
                    confounds_file = confounds_files[0]
                    break
        
        if not confounds_file:
            raise FileNotFoundError(f"No confounds file found for {session_folder}")
        
        confounds = pd.read_csv(confounds_file, sep='\t')
        
        # Debug information
        print(f"    BOLD data shape: {bold_data.shape} (nx, ny, nz, n_TR)")
        print(f"    Confounds shape: {confounds.shape}")
        print(f"    Confounds file: {confounds_file.name}")
        
        # Standard fMRI: time is last dimension in BOLD, first dimension in confounds
        n_timepoints = bold_data.shape[-1]  # Last dimension is time
        
        # Check for shape mismatch
        if n_timepoints != confounds.shape[0]:
            raise ValueError(f"Shape mismatch: BOLD has {n_timepoints} timepoints, "
                           f"confounds has {confounds.shape[0]} rows")
        
        print(f"    Confirmed: {n_timepoints} timepoints match between BOLD and confounds")
        
        # Move time axis to first dimension for processing: (n_TR, nx, ny, nz)
        bold_data = np.moveaxis(bold_data, -1, 0)
        print(f"    Reorganized BOLD shape: {bold_data.shape} (n_TR, nx, ny, nz)")
        
        return bold_data, bold_img, confounds, bold_file.name
    
    def diagnose_confounds(self, confounds):
        """Diagnose confounds file for debugging."""
        print(f"    Available confound columns ({len(confounds.columns)}):")
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        
        for col in motion_cols:
            if col in confounds.columns:
                print(f"      ✓ {col}")
            else:
                print(f"      ✗ {col} (MISSING)")
        
        if 'csf' in confounds.columns:
            print("      ✓ csf")
        else:
            print("      ✗ csf (MISSING)")
            
        print(f"    First few confound column names: {list(confounds.columns[:10])}")
    
    def quality_control_assessment(self, confounds, subject, session):
        """
        Assess data quality and motion artifacts.
        
        Returns:
        --------
        qc_report : dict
            Quality control metrics
        """
        qc_report = {
            'subject': subject,
            'session': session,
            'n_timepoints': len(confounds),
            'mean_fd': confounds['framewise_displacement'].mean(),
            'max_fd': confounds['framewise_displacement'].max(),
            'n_high_motion': (confounds['framewise_displacement'] > self.fd_threshold).sum(),
            'percent_high_motion': (confounds['framewise_displacement'] > self.fd_threshold).mean() * 100,
            'mean_dvars': confounds['std_dvars'].mean(),
            'max_dvars': confounds['std_dvars'].max(),
            'n_dvars_outliers': (confounds['std_dvars'] > self.dvars_threshold).sum(),
        }
        
        # Check for excessive motion
        if qc_report['percent_high_motion'] > 20:
            qc_report['motion_warning'] = True
        else:
            qc_report['motion_warning'] = False
            
        return qc_report
    
    def create_motion_regressors(self, confounds):
        """
        Create minimal motion regressor matrix.
        
        Returns:
        --------
        motion_regressors : np.ndarray
            6 motion parameters only
        """
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        
        # Check if all motion columns exist
        missing_cols = [col for col in motion_cols if col not in confounds.columns]
        if missing_cols:
            raise ValueError(f"Missing motion columns in confounds: {missing_cols}")
        
        motion_regressors = confounds[motion_cols].values
        
        # Handle NaN values (first timepoint often has NaN derivatives)
        motion_regressors = np.nan_to_num(motion_regressors, nan=0.0)
        
        print(f"    Motion regressors shape: {motion_regressors.shape}")
        
        return motion_regressors
    
    def create_conservative_regressors(self, confounds):
        """
        Create conservative regressor matrix with motion + CSF.
        
        Returns:
        --------
        regressors : np.ndarray
            Motion parameters + CSF signal
        """
        motion_regressors = self.create_motion_regressors(confounds)
        
        # Check if CSF column exists
        if 'csf' not in confounds.columns:
            raise ValueError("CSF column not found in confounds")
        
        # Add CSF signal only
        csf_signal = confounds[['csf']].values
        csf_signal = np.nan_to_num(csf_signal, nan=0.0)
        
        regressors = np.hstack([motion_regressors, csf_signal])
        
        print(f"    Conservative regressors shape: {regressors.shape}")
        
        return regressors
    
    def high_pass_filter(self, data, tr=1.0):
        """
        Apply gentle high-pass filter to remove low-frequency drift.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data (time x voxels)
        tr : float
            Repetition time in seconds
            
        Returns:
        --------
        filtered_data : np.ndarray
            High-pass filtered data
        """
        nyquist = 0.5 / tr
        high_pass_norm = self.high_pass_cutoff / nyquist
        
        # Design Butterworth high-pass filter
        b, a = signal.butter(2, high_pass_norm, btype='high')
        
        # Apply filter to each voxel time series
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                for k in range(data.shape[3]):
                    voxel_ts = data[:, i, j, k]
                    if np.std(voxel_ts) > 0:  # Only filter non-zero voxels
                        filtered_data[:, i, j, k] = signal.filtfilt(b, a, voxel_ts)
                    else:
                        filtered_data[:, i, j, k] = voxel_ts
        
        return filtered_data
    
    def regress_confounds(self, data, regressors):
        """
        Apply confound regression to BOLD data.
        
        Parameters:
        -----------
        data : np.ndarray
            BOLD data (time x x x y x z)
        regressors : np.ndarray
            Confound regressors (time x regressors)
            
        Returns:
        --------
        cleaned_data : np.ndarray
            Data with confounds regressed out
        """
        original_shape = data.shape
        data_2d = data.reshape(original_shape[0], -1)
        
        # Fit regression model for each voxel
        reg_model = LinearRegression(fit_intercept=True)
        cleaned_data_2d = np.zeros_like(data_2d)
        
        for voxel in range(data_2d.shape[1]):
            voxel_ts = data_2d[:, voxel]
            if np.std(voxel_ts) > 0:  # Only process non-zero voxels
                reg_model.fit(regressors, voxel_ts)
                residuals = voxel_ts - reg_model.predict(regressors)
                cleaned_data_2d[:, voxel] = residuals
            else:
                cleaned_data_2d[:, voxel] = voxel_ts
        
        return cleaned_data_2d.reshape(original_shape)
    
    def process_session(self, subject, session, session_folder, processing_level='minimal'):
        """
        Process a single session.
        
        Parameters:
        -----------
        processing_level : str
            'minimal' (motion only), 'conservative' (motion + CSF)
        """
        print(f"Processing {subject} - {session} ({processing_level})")
        
        # Load data
        bold_data, bold_img, confounds, original_filename = self.load_bold_data(session_folder)
        
        # Diagnose confounds for debugging
        self.diagnose_confounds(confounds)
        
        # Quality control
        qc_report = self.quality_control_assessment(confounds, subject, session)
        
        if qc_report['motion_warning']:
            print(f"  WARNING: High motion detected ({qc_report['percent_high_motion']:.1f}% high-motion volumes)")
        
        # Create regressors based on processing level
        if processing_level == 'minimal':
            regressors = self.create_motion_regressors(confounds)
            suffix = 'minimal'
        elif processing_level == 'conservative':
            regressors = self.create_conservative_regressors(confounds)
            suffix = 'conservative'
        else:
            raise ValueError("processing_level must be 'minimal' or 'conservative'")
        
        # Apply confound regression
        print(f"  Applying {regressors.shape[1]} confound regressors...")
        cleaned_data = self.regress_confounds(bold_data, regressors)
        
        # Apply high-pass filter
        print("  Applying high-pass filter...")
        filtered_data = self.high_pass_filter(cleaned_data, tr=1.0)
        
        # Save processed data
        output_filename = f"sub-{subject}_ses-{session}_task-prefuspost_desc-{suffix}_bold.nii.gz"
        output_path = self.output_dir / output_filename
        
        # Restore original dimension order: (n_TR, nx, ny, nz) -> (nx, ny, nz, n_TR)
        filtered_data_original_order = np.moveaxis(filtered_data, 0, -1)
        print(f"  Final output shape: {filtered_data_original_order.shape} (nx, ny, nz, n_TR)")
        
        # Create new NIfTI image with same header
        processed_img = nib.Nifti1Image(filtered_data_original_order, bold_img.affine, bold_img.header)
        nib.save(processed_img, output_path)
        
        print(f"  Saved: {output_filename}")
        
        return qc_report, output_path
    
    def process_all_sessions(self, processing_level='minimal'):
        """Process all sessions in the dataset."""
        sessions = self.get_subject_sessions()
        qc_reports = []
        
        print(f"Found {len(sessions)} sessions to process")
        print(f"Processing level: {processing_level}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        for subject, session, session_folder in sessions:
            try:
                qc_report, output_path = self.process_session(
                    subject, session, session_folder, processing_level
                )
                qc_reports.append(qc_report)
            except Exception as e:
                print(f"  ERROR processing {subject}-{session}: {str(e)}")
                continue
        
        # Save QC report
        qc_df = pd.DataFrame(qc_reports)
        qc_path = self.output_dir / f"quality_control_report_{processing_level}.csv"
        qc_df.to_csv(qc_path, index=False)
        
        print(f"\nProcessing complete!")
        print(f"Quality control report saved: {qc_path}")
        
        return qc_df
    
    def plot_quality_metrics(self, qc_df):
        """Create quality control plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Framewise displacement
        axes[0,0].hist(qc_df['mean_fd'], bins=20, alpha=0.7)
        axes[0,0].axvline(self.fd_threshold, color='red', linestyle='--', label=f'Threshold ({self.fd_threshold}mm)')
        axes[0,0].set_xlabel('Mean Framewise Displacement (mm)')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Motion: Framewise Displacement')
        axes[0,0].legend()
        
        # Percent high motion
        axes[0,1].hist(qc_df['percent_high_motion'], bins=20, alpha=0.7)
        axes[0,1].axvline(20, color='red', linestyle='--', label='Warning threshold (20%)')
        axes[0,1].set_xlabel('Percent High-Motion Volumes (%)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('Motion: High-Motion Volumes')
        axes[0,1].legend()
        
        # DVARS
        axes[1,0].hist(qc_df['mean_dvars'], bins=20, alpha=0.7)
        axes[1,0].set_xlabel('Mean DVARS')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('DVARS Distribution')
        
        # Session comparison
        if 'session' in qc_df.columns:
            sns.boxplot(data=qc_df, x='session', y='mean_fd', ax=axes[1,1])
            axes[1,1].set_title('Motion by Session Type')
            axes[1,1].set_ylabel('Mean FD (mm)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "quality_control_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path


def main():
    """Example usage of the preprocessing pipeline."""
    
    # Initialize preprocessor
    data_dir = "/Users/jacekdmochowski/PROJECTS/fus/data/resampled_bold_flywheel"
    preprocessor = MinimalPreprocessor(data_dir)
    
    # Process all sessions with minimal preprocessing
    print("=== MINIMAL PREPROCESSING (Motion parameters only) ===")
    qc_minimal = preprocessor.process_all_sessions(processing_level='minimal')
    
    # Process all sessions with conservative preprocessing
    print("\n=== CONSERVATIVE PREPROCESSING (Motion + CSF) ===")
    qc_conservative = preprocessor.process_all_sessions(processing_level='conservative')
    
    # Generate quality control plots
    print("\n=== QUALITY CONTROL PLOTS ===")
    preprocessor.plot_quality_metrics(qc_minimal)
    
    return qc_minimal, qc_conservative


if __name__ == "__main__":
    qc_minimal, qc_conservative = main()