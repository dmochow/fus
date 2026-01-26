"""
ROI time series extraction using Schaefer 100-parcel atlas.

This script extracts mean time series from each ROI for all preprocessed subjects,
focusing on the Schaefer 100-parcel 7-network atlas which is optimal for:
- Moderate number of ROIs (100) for complex modeling
- Good subgenual cingulate coverage
- Resting-state network organization

Author: Generated for tFUS project
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets, image, masking
from nilearn.plotting import plot_roi, show
import warnings
warnings.filterwarnings('ignore')

class ROITimeSeriesExtractor:
    """
    Extract ROI time series from preprocessed fMRI data.
    """
    
    def __init__(self, processed_data_dir, output_dir=None):
        """
        Initialize ROI extractor.
        
        Parameters:
        -----------
        processed_data_dir : str or Path
            Path to processed BOLD data directory
        output_dir : str or Path, optional
            Output directory for ROI time series
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.processed_data_dir.parent / "roi_time_series"
        self.output_dir.mkdir(exist_ok=True)
        
        # Download and prepare atlas
        self.atlas, self.labels = self._get_schaefer_atlas()
        self._identify_subgenual_rois()
        
    def _get_schaefer_atlas(self):
        """Download Schaefer 100-parcel atlas using nilearn."""
        print("Downloading Schaefer 100-parcel 7-network atlas...")
        
        # Download atlas from nilearn
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
        
        # Load atlas image and labels
        atlas_img = nib.load(atlas.maps)
        
        # Convert byte strings to regular strings
        labels_list = [label.decode('utf-8') if isinstance(label, bytes) else str(label) for label in atlas.labels]
        
        labels_df = pd.DataFrame({
            'roi_id': range(1, 101),
            'roi_name': labels_list,
            'network': [label.split('_')[2] for label in labels_list]
        })
        
        print(f"Atlas loaded: {atlas_img.shape} with {len(labels_df)} ROIs")
        print("Networks included:", labels_df['network'].unique())
        
        # Save atlas for reference
        atlas_path = self.output_dir.parent / "atlas" / "Schaefer2018_100Parcels_7Networks_MNI152NLin2009cAsym.nii.gz"
        atlas_path.parent.mkdir(exist_ok=True)
        nib.save(atlas_img, atlas_path)
        
        labels_path = self.output_dir.parent / "atlas" / "Schaefer2018_100Parcels_7Networks_labels.csv"
        labels_df.to_csv(labels_path, index=False)
        
        print(f"Atlas saved to: {atlas_path}")
        print(f"Labels saved to: {labels_path}")
        
        return atlas_img, labels_df
    
    def _identify_subgenual_rois(self):
        """Identify ROIs containing subgenual anterior cingulate cortex."""
        # Look for cingulate cortex ROIs, particularly subgenual/ventral regions
        cingulate_rois = self.labels[
            self.labels['roi_name'].str.contains('Limbic.*Cing|DefaultMode.*Cing', case=False, na=False)
        ].copy()
        
        print("\\nCingulate cortex ROIs (potential tFUS targets):")
        for _, roi in cingulate_rois.iterrows():
            print(f"  ROI {roi['roi_id']:2d}: {roi['roi_name']}")
        
        # Store for later reference
        self.cingulate_rois = cingulate_rois
        
    def get_processed_files(self):
        """Get all processed BOLD files."""
        processed_files = []
        
        for processing_type in ['minimal', 'conservative']:
            files = list(self.processed_data_dir.glob(f"*_desc-{processing_type}_bold.nii.gz"))
            for file in files:
                # Parse filename to extract subject/session info
                parts = file.stem.split('_')
                subject = parts[0].split('-')[1]
                session = parts[1].split('-')[1]
                
                processed_files.append({
                    'subject': subject,
                    'session': session,
                    'processing': processing_type,
                    'file_path': file
                })
        
        return pd.DataFrame(processed_files)
    
    def extract_roi_timeseries(self, bold_file, atlas_img):
        """
        Extract mean time series from each ROI.
        
        Parameters:
        -----------
        bold_file : str or Path
            Path to preprocessed BOLD file
        atlas_img : nibabel image
            Atlas image with ROI labels
            
        Returns:
        --------
        roi_timeseries : np.ndarray
            Time series array (n_timepoints, n_rois)
        """
        # Load BOLD data
        bold_img = nib.load(bold_file)
        
        # Resample atlas to BOLD space if needed
        if bold_img.shape[:3] != atlas_img.shape[:3]:
            print(f"    Resampling atlas from {atlas_img.shape} to {bold_img.shape[:3]}")
            atlas_resampled = image.resample_to_img(atlas_img, bold_img, interpolation='nearest')
        else:
            atlas_resampled = atlas_img
        
        # Get BOLD and atlas data
        bold_data = bold_img.get_fdata()
        atlas_data = atlas_resampled.get_fdata()
        
        # Get unique ROI labels (excluding 0 = background)
        roi_labels = np.unique(atlas_data)[1:]  # Exclude 0
        
        # Reshape BOLD data for easier processing: (x, y, z, time) -> (voxels, time)
        original_shape = bold_data.shape
        bold_2d = bold_data.reshape(-1, original_shape[-1])  # (n_voxels, n_timepoints)
        atlas_1d = atlas_data.flatten()  # (n_voxels,)
        
        # Compute mean time series for each ROI
        n_timepoints = original_shape[-1]
        n_rois = len(roi_labels)
        roi_means = np.zeros((n_timepoints, n_rois))
        
        for i, roi_label in enumerate(roi_labels):
            # Find voxels belonging to this ROI
            roi_voxel_indices = np.where(atlas_1d == roi_label)[0]
            
            if len(roi_voxel_indices) > 0:
                # Extract time series for all voxels in this ROI
                roi_voxel_timeseries = bold_2d[roi_voxel_indices, :]  # (n_roi_voxels, n_timepoints)
                # Compute mean across voxels
                roi_means[:, i] = np.mean(roi_voxel_timeseries, axis=0)
            else:
                print(f"    Warning: No voxels found for ROI {roi_label}")
                roi_means[:, i] = 0
        
        print(f"    Extracted time series: {roi_means.shape} (timepoints x ROIs)")
        
        return roi_means, roi_labels.astype(int)
    
    def process_subject_session(self, subject, session, processing_type):
        """Process a single subject-session combination."""
        print(f"Processing {subject} - {session} ({processing_type})")
        
        # Find the processed file
        file_pattern = f"sub-{subject}_ses-{session}_*_desc-{processing_type}_bold.nii.gz"
        files = list(self.processed_data_dir.glob(file_pattern))
        
        if not files:
            print(f"  No processed file found for {subject}-{session}-{processing_type}")
            return None
        
        bold_file = files[0]
        
        # Extract ROI time series
        roi_timeseries, roi_labels = self.extract_roi_timeseries(bold_file, self.atlas)
        
        # Create DataFrame with proper column names
        roi_columns = [f"ROI_{roi_id:03d}" for roi_id in roi_labels]
        
        roi_df = pd.DataFrame(
            roi_timeseries,
            columns=roi_columns
        )
        
        # Add metadata
        roi_df['subject'] = subject
        roi_df['session'] = session
        roi_df['processing'] = processing_type
        roi_df['timepoint'] = range(len(roi_df))
        
        # Save individual subject file
        output_file = self.output_dir / f"roi_timeseries_sub-{subject}_ses-{session}_desc-{processing_type}.csv"
        roi_df.to_csv(output_file, index=False)
        
        print(f"  Saved: {output_file.name}")
        
        return roi_df
    
    def process_all_subjects(self, processing_types=['minimal', 'conservative']):
        """Process all subjects and sessions."""
        processed_files = self.get_processed_files()
        
        print(f"Found {len(processed_files)} processed files")
        print(f"Processing types: {processing_types}")
        print("-" * 50)
        
        all_roi_data = []
        
        for processing_type in processing_types:
            print(f"\\n=== Processing {processing_type.upper()} data ===")
            
            # Filter files for this processing type
            files_subset = processed_files[processed_files['processing'] == processing_type]
            
            for _, file_info in files_subset.iterrows():
                try:
                    roi_df = self.process_subject_session(
                        file_info['subject'], 
                        file_info['session'], 
                        file_info['processing']
                    )
                    if roi_df is not None:
                        all_roi_data.append(roi_df)
                        
                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    continue
        
        if all_roi_data:
            # Combine all data
            combined_df = pd.concat(all_roi_data, ignore_index=True)
            
            # Save combined dataset
            combined_file = self.output_dir / "roi_timeseries_all_subjects.csv"
            combined_df.to_csv(combined_file, index=False)
            
            print(f"\\nCombined dataset saved: {combined_file}")
            print(f"Final dataset shape: {combined_df.shape}")
            
            # Create summary
            self._create_summary_report(combined_df)
            
            return combined_df
        else:
            print("No data processed successfully")
            return None
    
    def _create_summary_report(self, combined_df):
        """Create summary report of extracted data."""
        summary = {
            'n_subjects': combined_df['subject'].nunique(),
            'n_sessions_total': len(combined_df.groupby(['subject', 'session'])),
            'n_timepoints_per_session': combined_df.groupby(['subject', 'session'])['timepoint'].count().iloc[0],
            'n_rois': len([col for col in combined_df.columns if col.startswith('ROI_')]),
            'processing_types': combined_df['processing'].unique().tolist(),
            'subjects': sorted(combined_df['subject'].unique().tolist()),
            'sessions': sorted(combined_df['session'].unique().tolist())
        }
        
        # Session count by type
        session_counts = combined_df.groupby(['subject', 'session']).size().reset_index(name='count')
        session_type_counts = session_counts['session'].value_counts()
        
        print(f"\\n=== ROI EXTRACTION SUMMARY ===")
        print(f"Subjects: {summary['n_subjects']}")
        print(f"Total sessions: {summary['n_sessions_total']}")
        print(f"Session types: {dict(session_type_counts)}")
        print(f"ROIs: {summary['n_rois']}")
        print(f"Timepoints per session: {summary['n_timepoints_per_session']}")
        print(f"Processing types: {summary['processing_types']}")
        
        # Save summary
        summary_file = self.output_dir / "extraction_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Show cingulate ROI info
        print(f"\\n=== CINGULATE CORTEX ROIs (tFUS target region) ===")
        for _, roi in self.cingulate_rois.iterrows():
            roi_col = f"ROI_{roi['roi_id']:03d}"
            if roi_col in combined_df.columns:
                roi_data = combined_df[roi_col]
                print(f"ROI {roi['roi_id']:2d} ({roi['network']:12s}): {roi['roi_name']}")
                print(f"  Mean signal: {roi_data.mean():.3f} ± {roi_data.std():.3f}")
    
    def plot_atlas_overview(self):
        """Create visualization of the atlas."""
        print("Creating atlas overview plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot atlas
        plot_roi(self.atlas, title="Schaefer 100-parcel Atlas", 
                 axes=axes[0,0], display_mode='z', cut_coords=[0])
        
        # Network distribution
        network_counts = self.labels['network'].value_counts()
        axes[0,1].bar(range(len(network_counts)), network_counts.values)
        axes[0,1].set_xticks(range(len(network_counts)))
        axes[0,1].set_xticklabels(network_counts.index, rotation=45, ha='right')
        axes[0,1].set_title('ROIs per Network')
        axes[0,1].set_ylabel('Number of ROIs')
        
        # Cingulate ROIs
        if hasattr(self, 'cingulate_rois'):
            cingulate_counts = self.cingulate_rois['network'].value_counts()
            axes[1,0].bar(range(len(cingulate_counts)), cingulate_counts.values)
            axes[1,0].set_xticks(range(len(cingulate_counts)))
            axes[1,0].set_xticklabels(cingulate_counts.index, rotation=45, ha='right')
            axes[1,0].set_title('Cingulate ROIs by Network')
            axes[1,0].set_ylabel('Number of ROIs')
        
        # ROI ID distribution
        axes[1,1].hist(self.labels['roi_id'], bins=20, alpha=0.7)
        axes[1,1].set_title('ROI ID Distribution')
        axes[1,1].set_xlabel('ROI ID')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "atlas_overview.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path


def main():
    """Example usage of ROI time series extraction."""
    
    # Initialize extractor
    processed_data_dir = "/Users/jacekdmochowski/PROJECTS/fus/data/processed_bold"
    extractor = ROITimeSeriesExtractor(processed_data_dir)
    
    # Create atlas overview
    extractor.plot_atlas_overview()
    
    # Process all subjects
    print("\\n" + "="*60)
    roi_data = extractor.process_all_subjects()
    
    return roi_data


if __name__ == "__main__":
    roi_data = main()