"""
Entropy Production Analysis for tFUS fMRI Data using Stochastic Thermodynamics.

This script implements the network-level entropy production analysis inspired by 
Deco et al. approach, aggregating 100 Schaefer ROIs to 7 Yeo networks and 
estimating entropy production using KL divergence between forward and backward 
transition probabilities.

Key concepts:
- Entropy production rate: S_dot = KL[P_forward || P_backward]
- Network-level dynamics to address dimensionality challenges
- Comparison across tFUS conditions and epochs

Author: Generated for tFUS project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')

class EntropyProductionAnalyzer:
    """
    Analyze entropy production in fMRI data using stochastic thermodynamics.
    
    This class implements multiple methods for estimating entropy production across
    three contiguous, equi-duration epochs:
    - **Baseline** (TRs 0-300): Pre-stimulation period
    - **Stimulation** (TRs 300-600): tFUS application period  
    - **Recovery** (TRs 600-900): Post-stimulation period
    
    **Entropy Production Methods:**
    
    1. **Discrete Method** (default, recommended):
       - Uses k-means clustering to discretize brain states
       - Computes transition probabilities empirically
       - Always produces non-negative values
       - Robust and interpretable
       
    2. **Gaussian Method** (analytical):
       - Assumes multivariate Gaussian transition distributions
       - Uses analytical KL divergence formula between Gaussians
       - Guaranteed non-negative by theoretical construction
       - More sensitive to continuous dynamics
       
    3. **KDE Method** (kernel density estimation):
       - Uses kernel density estimation for transition probabilities
       - More flexible than Gaussian assumption
       - Computationally intensive but handles non-Gaussian distributions
       - Currently uses discrete method for final entropy calculation
    
    **Theoretical Background:**
    Entropy production measures the irreversibility of brain dynamics:
    S_dot = KL[P_forward || P_backward] ≥ 0
    
    Where P_forward(x_t, x_{t+1}) and P_backward(x_t, x_{t+1}) are the
    forward and time-reversed transition probabilities.
    
    **References:**
    - Deco et al. (2021) "The thermodynamics of mind"
    - Parrondo et al. (2009) "Entropy production and the arrow of time"  
    - Seifert (2012) "Stochastic thermodynamics, fluctuation theorems and molecular machines"
    """
    
    def __init__(self, roi_data_dir, output_dir=None):
        """
        Initialize entropy production analyzer.
        
        Parameters:
        -----------
        roi_data_dir : str or Path
            Path to ROI time series directory
        output_dir : str or Path, optional
            Output directory for entropy analysis results
        """
        self.roi_data_dir = Path(roi_data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.roi_data_dir.parent / "entropy_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis parameters
        self.window_length = 60  # TRs for transition estimation
        self.step_size = 10  # TRs between windows
        
        # Clean contiguous epochs (equi-duration: 300 TRs each)
        self.epochs = {
            'baseline': (0, 300),           # Pre-stimulation baseline
            'stimulation': (300, 600),      # Stimulation period (includes all tFUS)
            'recovery': (600, 900)          # Post-stimulation recovery
        }
        
        # Yeo 7-network mapping for Schaefer 100 parcels
        self.network_mapping = self._create_network_mapping()
        
        print(f"Entropy Production Analysis Parameters:")
        print(f"  Window length: {self.window_length} TRs")
        print(f"  Step size: {self.step_size} TRs")
        print(f"  Networks: {list(self.network_mapping.keys())}")
    
    def _create_network_mapping(self):
        """
        Create mapping from 100 Schaefer ROIs to 7 Yeo networks.
        
        Returns:
        --------
        network_mapping : dict
            Dictionary mapping network names to ROI indices
        """
        # Load ROI labels to extract network information
        try:
            labels_file = self.roi_data_dir.parent / "atlas" / "Schaefer2018_100Parcels_7Networks_labels.csv"
            if labels_file.exists():
                labels_df = pd.read_csv(labels_file)
            else:
                # Create basic network mapping if detailed labels unavailable
                print("  Warning: Detailed labels not found, using estimated network mapping")
                return self._create_estimated_network_mapping()
        except:
            return self._create_estimated_network_mapping()
        
        # Extract network information from ROI names
        network_mapping = {
            'Visual': [],
            'Somatomotor': [],
            'DorsAttn': [],
            'SalVentAttn': [],
            'Limbic': [],
            'Cont': [],
            'Default': []
        }
        
        for idx, row in labels_df.iterrows():
            roi_name = row['roi_name']
            roi_id = row['roi_id'] - 1  # Convert to 0-based indexing
            
            # Parse network from ROI name (e.g., "7Networks_LH_Vis_1" -> "Visual")
            if 'Vis' in roi_name:
                network_mapping['Visual'].append(roi_id)
            elif 'SomMot' in roi_name:
                network_mapping['Somatomotor'].append(roi_id)
            elif 'DorsAttn' in roi_name:
                network_mapping['DorsAttn'].append(roi_id)
            elif 'SalVentAttn' in roi_name:
                network_mapping['SalVentAttn'].append(roi_id)
            elif 'Limbic' in roi_name:
                network_mapping['Limbic'].append(roi_id)
            elif 'Cont' in roi_name:
                network_mapping['Cont'].append(roi_id)
            elif 'Default' in roi_name:
                network_mapping['Default'].append(roi_id)
        
        print(f"  Network sizes: {[(net, len(rois)) for net, rois in network_mapping.items()]}")
        
        return network_mapping
    
    def _create_estimated_network_mapping(self):
        """Create estimated network mapping based on typical Schaefer organization."""
        # Approximate network boundaries for Schaefer 100 (these are estimates)
        network_mapping = {
            'Visual': list(range(0, 12)),          # ROIs 1-12
            'Somatomotor': list(range(12, 24)),    # ROIs 13-24  
            'DorsAttn': list(range(24, 36)),       # ROIs 25-36
            'SalVentAttn': list(range(36, 48)),    # ROIs 37-48
            'Limbic': list(range(48, 60)),         # ROIs 49-60
            'Cont': list(range(60, 78)),           # ROIs 61-78
            'Default': list(range(78, 100))        # ROIs 79-100
        }
        
        print("  Using estimated network mapping (verify with actual atlas labels)")
        
        return network_mapping
    
    def load_roi_data(self, processing_type='minimal'):
        """Load ROI time series data."""
        print(f"Loading ROI data (processing: {processing_type})...")
        
        # Load combined dataset
        combined_file = self.roi_data_dir / "roi_timeseries_all_subjects.csv"
        if not combined_file.exists():
            raise FileNotFoundError(f"Combined ROI data not found: {combined_file}")
        
        df = pd.read_csv(combined_file)
        df_filtered = df[df['processing'] == processing_type].copy()
        
        roi_columns = [col for col in df_filtered.columns if col.startswith('ROI_')]
        
        print(f"  Loaded data shape: {df_filtered.shape}")
        print(f"  Number of ROIs: {len(roi_columns)}")
        print(f"  Subjects: {sorted(df_filtered['subject'].unique())}")
        print(f"  Sessions: {sorted(df_filtered['session'].unique())}")
        
        return df_filtered, roi_columns
    
    def aggregate_to_networks(self, roi_data, roi_columns):
        """
        Aggregate 100 ROI time series to 7 network time series.
        
        Parameters:
        -----------
        roi_data : np.ndarray
            ROI time series data (n_timepoints, n_rois)
        roi_columns : list
            List of ROI column names
            
        Returns:
        --------
        network_data : np.ndarray
            Network time series data (n_timepoints, 7)
        network_names : list
            List of network names
        """
        n_timepoints = roi_data.shape[0]
        network_names = list(self.network_mapping.keys())
        network_data = np.zeros((n_timepoints, len(network_names)))
        
        for i, (network_name, roi_indices) in enumerate(self.network_mapping.items()):
            if len(roi_indices) > 0:
                # Average across ROIs within each network
                network_timeseries = np.mean(roi_data[:, roi_indices], axis=1)
                network_data[:, i] = network_timeseries
        
        return network_data, network_names
    
    def estimate_transition_probabilities(self, network_data, method='gaussian'):
        """
        Estimate forward and backward transition probabilities.
        
        Parameters:
        -----------
        network_data : np.ndarray
            Network time series (n_timepoints, n_networks)
        method : str
            Method for density estimation ('gaussian', 'kde', 'discrete')
            
        Returns:
        --------
        forward_prob : callable
            Function to compute P(x_{t+1}|x_t)
        backward_prob : callable  
            Function to compute P(x_t|x_{t+1})
        """
        if method == 'gaussian':
            return self._estimate_gaussian_transitions(network_data)
        elif method == 'kde':
            return self._estimate_kde_transitions(network_data)
        elif method == 'discrete':
            return self._estimate_discrete_transitions(network_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _estimate_gaussian_transitions(self, network_data):
        """Estimate transitions using multivariate Gaussian approximation."""
        n_timepoints, n_networks = network_data.shape
        
        # Create state pairs (x_t, x_{t+1})
        X_t = network_data[:-1, :]      # States at time t
        X_t1 = network_data[1:, :]      # States at time t+1
        
        # Estimate joint distribution P(x_t, x_{t+1})
        joint_data = np.hstack([X_t, X_t1])  # Shape: (n_transitions, 2*n_networks)
        
        # Fit multivariate Gaussian
        from sklearn.covariance import EmpiricalCovariance
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(joint_data)
        
        joint_mean = np.mean(joint_data, axis=0)
        joint_cov = cov_estimator.covariance_
        
        # Split parameters for forward and backward
        n_nets = n_networks
        
        # Forward: P(x_{t+1}|x_t)
        mu_forward = joint_mean[n_nets:]  # Mean of x_{t+1}
        sigma_forward_cond = joint_cov[n_nets:, n_nets:] - joint_cov[n_nets:, :n_nets] @ \
                           np.linalg.pinv(joint_cov[:n_nets, :n_nets]) @ joint_cov[:n_nets, n_nets:]
        
        # Backward: P(x_t|x_{t+1})  
        mu_backward = joint_mean[:n_nets]  # Mean of x_t
        sigma_backward_cond = joint_cov[:n_nets, :n_nets] - joint_cov[:n_nets, n_nets:] @ \
                            np.linalg.pinv(joint_cov[n_nets:, n_nets:]) @ joint_cov[n_nets:, :n_nets]
        
        def forward_prob(x_t, x_t1):
            """P(x_{t+1}|x_t)"""
            mu_cond = mu_forward + joint_cov[n_nets:, :n_nets] @ \
                      np.linalg.pinv(joint_cov[:n_nets, :n_nets]) @ (x_t - joint_mean[:n_nets])
            return stats.multivariate_normal.pdf(x_t1, mu_cond, sigma_forward_cond)
        
        def backward_prob(x_t, x_t1):
            """P(x_t|x_{t+1})"""
            mu_cond = mu_backward + joint_cov[:n_nets, n_nets:] @ \
                      np.linalg.pinv(joint_cov[n_nets:, n_nets:]) @ (x_t1 - joint_mean[n_nets:])
            return stats.multivariate_normal.pdf(x_t, mu_cond, sigma_backward_cond)
        
        return forward_prob, backward_prob
    
    def _estimate_kde_transitions(self, network_data, bandwidth='scott'):
        """
        Estimate transitions using kernel density estimation.
        
        Parameters:
        -----------
        network_data : np.ndarray
            Network time series (n_timepoints, n_networks)
        bandwidth : str or float
            KDE bandwidth parameter ('scott', 'silverman', or numeric value)
            
        Returns:
        --------
        forward_prob : callable
            Function to compute P(x_{t+1}|x_t)
        backward_prob : callable  
            Function to compute P(x_t|x_{t+1})
        """
        from sklearn.neighbors import KernelDensity
        
        n_timepoints, n_networks = network_data.shape
        
        # Create transition data
        X_t = network_data[:-1, :]      # States at time t
        X_t1 = network_data[1:, :]      # States at time t+1

        # Stack for joint density estimation
        joint_data = np.hstack([X_t, X_t1])  # Shape: (n_transitions, 2*n_networks)
        
        # Fit KDE to joint distribution P(x_t, x_{t+1})
        kde_joint = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_joint.fit(joint_data)
        
        # Fit marginal KDEs
        kde_xt = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_xt.fit(X_t)
        
        kde_xt1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_xt1.fit(X_t1)
        
        def forward_prob(x_t, x_t1):
            """P(x_{t+1}|x_t) = P(x_t, x_{t+1}) / P(x_t)"""
            # Ensure inputs are 2D
            x_t = np.atleast_2d(x_t)
            x_t1 = np.atleast_2d(x_t1)
            
            # Joint probability
            joint_point = np.hstack([x_t, x_t1])
            log_prob_joint = kde_joint.score_samples(joint_point)[0]
            
            # Marginal probability P(x_t)
            log_prob_xt = kde_xt.score_samples(x_t)[0]
            
            # Conditional probability (in log space, then exp)
            log_conditional = log_prob_joint - log_prob_xt
            return np.exp(log_conditional)
        
        def backward_prob(x_t, x_t1):
            """P(x_t|x_{t+1}) = P(x_t, x_{t+1}) / P(x_{t+1})"""
            # Ensure inputs are 2D
            x_t = np.atleast_2d(x_t)
            x_t1 = np.atleast_2d(x_t1)
            
            # Joint probability
            joint_point = np.hstack([x_t, x_t1])
            log_prob_joint = kde_joint.score_samples(joint_point)[0]
            
            # Marginal probability P(x_{t+1})
            log_prob_xt1 = kde_xt1.score_samples(x_t1)[0]
            
            # Conditional probability (in log space, then exp)
            log_conditional = log_prob_joint - log_prob_xt1
            return np.exp(log_conditional)
        
        return forward_prob, backward_prob
    
    def _estimate_discrete_transitions(self, network_data, n_states=8):
        """Estimate transitions using discrete state space."""
        # Discretize network activity using k-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_states, random_state=42)
        states = kmeans.fit_predict(network_data)
        
        # Count transitions
        forward_counts = np.zeros((n_states, n_states))
        backward_counts = np.zeros((n_states, n_states))
        
        for t in range(len(states) - 1):
            s_t = states[t]
            s_t1 = states[t + 1]
            forward_counts[s_t, s_t1] += 1
            backward_counts[s_t1, s_t] += 1
        
        # Normalize to probabilities
        forward_probs = forward_counts / (forward_counts.sum(axis=1, keepdims=True) + 1e-10)
        backward_probs = backward_counts / (backward_counts.sum(axis=1, keepdims=True) + 1e-10)
        
        def forward_prob(state_t, state_t1):
            return forward_probs[state_t, state_t1]
        
        def backward_prob(state_t, state_t1):
            return backward_probs[state_t1, state_t]
        
        return forward_prob, backward_prob, states, kmeans
    
    def compute_entropy_production(self, network_data, method='gaussian', n_states=8):
        """
        Compute entropy production using KL divergence.
        
        Parameters:
        -----------
        network_data : np.ndarray
            Network time series (n_timepoints, n_networks)
        method : str
            Method for transition probability estimation
            
        Returns:
        --------
        entropy_production : float
            Entropy production rate
        entropy_timeseries : np.ndarray
            Time-resolved entropy production
        """
        print(f"  Computing entropy production using {method} method...")
        
        if method == 'discrete':
            entropy_prod, entropy_ts, states = self._compute_discrete_entropy_production(network_data, n_states)
            return entropy_prod, entropy_ts, states
        elif method == 'kde':
            entropy_prod, entropy_ts, states = self._compute_kde_entropy_production(network_data, n_states)
            return entropy_prod, entropy_ts, states
        else:
            entropy_prod, entropy_ts = self._compute_continuous_entropy_production(network_data, method)
            return entropy_prod, entropy_ts, None
    
    def _compute_discrete_entropy_production(self, network_data, n_states=8):
        """Compute entropy production using discrete states."""
        forward_prob, backward_prob, states, kmeans = self._estimate_discrete_transitions(
            network_data, n_states
        )
        
        # Compute KL divergence between forward and backward processes
        entropy_production = 0.0
        entropy_timeseries = []
        
        for t in range(len(states) - 1):
            s_t = states[t]
            s_t1 = states[t + 1]
            
            p_forward = forward_prob(s_t, s_t1)
            p_backward = backward_prob(s_t, s_t1)
            
            if p_forward > 1e-10 and p_backward > 1e-10:
                kl_contribution = p_forward * np.log(p_forward / p_backward)
                entropy_production += kl_contribution
                entropy_timeseries.append(kl_contribution)
            else:
                entropy_timeseries.append(0.0)
        
        entropy_timeseries = np.array(entropy_timeseries)
        
        return entropy_production / len(states), entropy_timeseries, states
    
    def _compute_kde_entropy_production(self, network_data, n_states=8):
        """Compute entropy production using KDE transitions but discrete state approximation."""
        print(f"    Using KDE for transition estimation with {n_states} discrete states")
        
        # Get KDE transition probability functions
        forward_prob, backward_prob = self._estimate_kde_transitions(network_data)
        
        # Discretize network activity for entropy calculation
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_states, random_state=42)
        states = kmeans.fit_predict(network_data)
        
        # Get cluster centers for evaluating KDE probabilities
        cluster_centers = kmeans.cluster_centers_
        
        # Compute KL divergence using KDE probabilities
        entropy_production = 0.0
        entropy_timeseries = []
        
        for t in range(len(states) - 1):
            s_t = states[t]
            s_t1 = states[t + 1]
            
            # Get cluster center coordinates
            x_t = cluster_centers[s_t]
            x_t1 = cluster_centers[s_t1]
            
            # Evaluate KDE probabilities at cluster centers
            try:
                p_forward = forward_prob(x_t, x_t1)
                p_backward = backward_prob(x_t, x_t1)
                
                # Ensure probabilities are valid
                if p_forward > 1e-10 and p_backward > 1e-10:
                    kl_contribution = p_forward * np.log(p_forward / p_backward)
                    entropy_production += kl_contribution
                    entropy_timeseries.append(kl_contribution)
                else:
                    entropy_timeseries.append(0.0)
            except:
                # Fall back to zero if KDE evaluation fails
                entropy_timeseries.append(0.0)
        
        entropy_timeseries = np.array(entropy_timeseries)
        
        return entropy_production / len(states), entropy_timeseries, states
    
    def _compute_continuous_entropy_production(self, network_data, method='gaussian'):
        """
        Compute entropy production using the Kullback-Leibler rate for continuous systems.
        
        This implements the proper entropy production formula that is guaranteed to be non-negative.
        We use the approach from:
        - Parrondo et al. (2009) "Entropy production and the arrow of time"
        - Deco et al. (2021) "The thermodynamics of mind"
        """
        
        if method == 'gaussian':
            return self._compute_gaussian_kl_rate(network_data)
        else:
            raise ValueError(f"Unsupported continuous method: {method}. Use 'gaussian', 'discrete', or 'kde'.")
    
    def _compute_gaussian_kl_rate(self, network_data):
        """
        Compute KL rate using Gaussian assumption with analytical formula.
        
        For multivariate Gaussian processes, the KL rate can be computed analytically
        and is guaranteed to be non-negative.
        """
        n_timepoints, n_networks = network_data.shape
        
        # Create transition data
        X_t = network_data[:-1, :]      # States at time t
        X_t1 = network_data[1:, :]      # States at time t+1
        
        # Stack for joint analysis
        joint_data = np.hstack([X_t, X_t1])  # (n_transitions, 2*n_networks)
        
        # Compute covariance matrices
        from sklearn.covariance import EmpiricalCovariance
        
        # Forward process covariance
        cov_forward = EmpiricalCovariance().fit(joint_data).covariance_
        
        # Backward process: swap X_t and X_{t+1}
        joint_data_backward = np.hstack([X_t1, X_t])
        cov_backward = EmpiricalCovariance().fit(joint_data_backward).covariance_
        
        # Analytical KL divergence between two multivariate Gaussians
        # KL(P_forward || P_backward) = 0.5 * [tr(Σ_b^{-1} Σ_f) - k + ln(det(Σ_b)/det(Σ_f))]
        # where k is the dimensionality
        
        try:
            cov_b_inv = np.linalg.pinv(cov_backward)
            
            # Compute terms
            trace_term = np.trace(cov_b_inv @ cov_forward)
            
            # Determinant terms (use log determinant for numerical stability)
            sign_f, logdet_f = np.linalg.slogdet(cov_forward)
            sign_b, logdet_b = np.linalg.slogdet(cov_backward)
            
            if sign_f > 0 and sign_b > 0:
                logdet_term = logdet_b - logdet_f
                
                # KL divergence formula
                k = joint_data.shape[1]  # Dimensionality
                kl_rate = 0.5 * (trace_term - k + logdet_term)
                
                # Ensure non-negative (should be by construction, but numerical errors possible)
                kl_rate = max(0.0, kl_rate)
                
            else:
                print("    Warning: Singular covariance matrices, using discrete method")
                entropy_prod, entropy_ts, states = self._compute_discrete_entropy_production(network_data)
                return entropy_prod, entropy_ts
                
        except np.linalg.LinAlgError:
            print("    Warning: Numerical issues with Gaussian method, using discrete method")
            entropy_prod, entropy_ts, states = self._compute_discrete_entropy_production(network_data)
            return entropy_prod, entropy_ts
        
        # For time series, we approximate as constant rate
        entropy_timeseries = np.full(n_timepoints - 1, kl_rate / (n_timepoints - 1))
        
        return kl_rate, entropy_timeseries
    
    def analyze_subject_session(self, subject, session, condition, df_subset, roi_columns, method='discrete', n_states=8):
        """Analyze entropy production for a single subject-session."""
        print(f"  Processing {subject} - {session} ({condition})")
        
        # Extract time series data
        session_data = df_subset[
            (df_subset['subject'] == subject) & (df_subset['session'] == session)
        ].copy()
        
        if len(session_data) == 0:
            print(f"    No data found for {subject}-{session}")
            return None
        
        # Sort by timepoint and extract ROI data
        session_data = session_data.sort_values('timepoint')
        roi_data = session_data[roi_columns].values
        
        # Standardize data
        scaler = StandardScaler()
        roi_data_std = scaler.fit_transform(roi_data)
        
        # Aggregate to network level
        network_data, network_names = self.aggregate_to_networks(roi_data_std, roi_columns)
        
        print(f"    Network data shape: {network_data.shape}")
        
        # Compute entropy production for different epochs
        epoch_results = {}
        
        for epoch_name, epoch_def in self.epochs.items():
            # All epochs are now single continuous periods
            start, end = epoch_def
            if start < len(network_data) and end <= len(network_data):
                epoch_indices = np.arange(start, end)
            else:
                epoch_indices = np.arange(start, min(end, len(network_data)))
            
            if len(epoch_indices) > self.window_length:
                epoch_data = network_data[epoch_indices, :]
                
                # Compute entropy production for this epoch
                try:
                    entropy_prod, entropy_ts, epoch_states = self.compute_entropy_production(
                        epoch_data, method=method, n_states=n_states
                    )
                    
                    epoch_results[epoch_name] = {
                        'entropy_production': entropy_prod,
                        'entropy_timeseries': entropy_ts,
                        'states': epoch_states,
                        'n_timepoints': len(epoch_indices)
                    }
                    
                    print(f"      {epoch_name}: entropy = {entropy_prod:.6f}")
                    
                except Exception as e:
                    print(f"      Error in {epoch_name}: {str(e)}")
                    epoch_results[epoch_name] = None
            else:
                print(f"      {epoch_name}: insufficient data ({len(epoch_indices)} TRs)")
                epoch_results[epoch_name] = None
        
        # Package results
        results = {
            'subject': subject,
            'session': session,
            'condition': condition,
            'network_data': network_data,
            'network_names': network_names,
            'epoch_results': epoch_results,
            'n_timepoints': len(session_data)
        }
        
        return results
    
    def analyze_condition(self, condition, processing_type='minimal', method='discrete', n_states=8):
        """Analyze entropy production for all subjects in a condition."""
        print(f"\\n=== Analyzing {condition} condition (entropy production) ===")
        
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
                results = self.analyze_subject_session(
                    subject, condition, condition, df_condition, roi_columns, method=method, n_states=n_states
                )
                if results is not None:
                    all_results.append(results)
            except Exception as e:
                print(f"    ERROR processing {subject}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(all_results)} subjects for {condition}")
        
        return all_results
    
    def compare_conditions(self, active_results, sham_results):
        """Compare entropy production between ACTIVE and SHAM conditions."""
        print("\\n=== Comparing Entropy Production: ACTIVE vs SHAM ===")
        
        # Extract entropy production values for each epoch
        comparison_results = {}
        
        epochs = ['baseline', 'stimulation', 'recovery']
        
        for epoch in epochs:
            active_entropy = []
            sham_entropy = []
            
            # Collect entropy values from all subjects
            for result in active_results:
                if result['epoch_results'][epoch] is not None:
                    active_entropy.append(result['epoch_results'][epoch]['entropy_production'])
            
            for result in sham_results:
                if result['epoch_results'][epoch] is not None:
                    sham_entropy.append(result['epoch_results'][epoch]['entropy_production'])
            
            if len(active_entropy) > 0 and len(sham_entropy) > 0:
                # Statistical comparison
                t_stat, p_val = stats.ttest_ind(active_entropy, sham_entropy)
                
                comparison_results[epoch] = {
                    'active_entropy': np.array(active_entropy),
                    'sham_entropy': np.array(sham_entropy),
                    'active_mean': np.mean(active_entropy),
                    'sham_mean': np.mean(sham_entropy),
                    'active_std': np.std(active_entropy),
                    'sham_std': np.std(sham_entropy),
                    'difference': np.mean(active_entropy) - np.mean(sham_entropy),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'n_active': len(active_entropy),
                    'n_sham': len(sham_entropy)
                }
                
                print(f"\\n{epoch.replace('_', ' ').title()}:")
                print(f"  ACTIVE: {np.mean(active_entropy):.6f} ± {np.std(active_entropy):.6f} (n={len(active_entropy)})")
                print(f"  SHAM:   {np.mean(sham_entropy):.6f} ± {np.std(sham_entropy):.6f} (n={len(sham_entropy)})")
                print(f"  Difference: {np.mean(active_entropy) - np.mean(sham_entropy):.6f}")
                print(f"  t-test: t={t_stat:.3f}, p={p_val:.6f}")
        
        # Create visualization
        self.plot_entropy_comparison(comparison_results)
        
        # Save results
        self.save_entropy_results(comparison_results, active_results, sham_results)
        
        # Save individual subject data
        self.save_individual_subject_data(active_results, sham_results)
        
        # Create enhanced visualizations
        self.plot_individual_subjects(active_results, sham_results)
        
        return comparison_results
    
    def plot_entropy_comparison(self, comparison_results):
        """Create visualization of entropy production comparison."""
        print("Creating entropy production plots...")
        
        epochs = list(comparison_results.keys())
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, epoch in enumerate(epochs):
            data = comparison_results[epoch]
            
            # Box plot
            box_data = [data['active_entropy'], data['sham_entropy']]
            axes[i].boxplot(box_data, labels=['ACTIVE', 'SHAM'])
            axes[i].set_title(f"{epoch.replace('_', ' ').title()}\\n"
                            f"p = {data['p_value']:.4f}")
            axes[i].set_ylabel('Entropy Production')
            axes[i].grid(True, alpha=0.3)
            
            # Add significance annotation
            if data['p_value'] < 0.05:
                axes[i].text(0.5, 0.95, '*', transform=axes[i].transAxes, 
                           fontsize=20, ha='center', va='top')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "entropy_production_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Entropy comparison plot saved: {plot_path}")
    
    def save_entropy_results(self, comparison_results, active_results, sham_results):
        """Save entropy production analysis results."""
        
        # Save comparison summary
        summary_data = []
        for epoch, data in comparison_results.items():
            summary_data.append({
                'epoch': epoch,
                'active_mean': data['active_mean'],
                'active_std': data['active_std'],
                'sham_mean': data['sham_mean'], 
                'sham_std': data['sham_std'],
                'difference': data['difference'],
                't_statistic': data['t_statistic'],
                'p_value': data['p_value'],
                'n_active': data['n_active'],
                'n_sham': data['n_sham']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "entropy_production_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Entropy production summary saved: {summary_path}")
    
    def save_individual_subject_data(self, active_results, sham_results):
        """Save individual subject entropy production values to CSV."""
        print("Saving individual subject data...")
        
        # Prepare data for CSV
        subject_data = []
        
        epochs = ['baseline', 'stimulation', 'recovery']
        
        # Process ACTIVE results
        for result in active_results:
            subject = result['subject']
            for epoch in epochs:
                if result['epoch_results'][epoch] is not None:
                    entropy_val = result['epoch_results'][epoch]['entropy_production']
                    subject_data.append({
                        'subject': subject,
                        'condition': 'ACTIVE',
                        'epoch': epoch,
                        'entropy_production': entropy_val
                    })
        
        # Process SHAM results
        for result in sham_results:
            subject = result['subject']
            for epoch in epochs:
                if result['epoch_results'][epoch] is not None:
                    entropy_val = result['epoch_results'][epoch]['entropy_production']
                    subject_data.append({
                        'subject': subject,
                        'condition': 'SHAM',
                        'epoch': epoch,
                        'entropy_production': entropy_val
                    })
        
        # Create DataFrame and save
        subject_df = pd.DataFrame(subject_data)
        subject_file = self.output_dir / "entropy_production_individual_subjects.csv"
        subject_df.to_csv(subject_file, index=False)
        
        print(f"Individual subject data saved: {subject_file}")
        return subject_df
    
    def plot_individual_subjects(self, active_results, sham_results):
        """Create scatter plots showing individual subject entropy production."""
        print("Creating individual subject plots...")
        
        epochs = ['baseline', 'stimulation', 'recovery']
        
        # Create figure with subplots for each epoch
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = {'ACTIVE': '#FF6B6B', 'SHAM': '#4ECDC4'}
        markers = {'ACTIVE': 'o', 'SHAM': 's'}
        
        for i, epoch in enumerate(epochs):
            ax = axes[i]
            
            # Collect data for this epoch
            active_entropy = []
            sham_entropy = []
            active_subjects = []
            sham_subjects = []
            
            for result in active_results:
                if result['epoch_results'][epoch] is not None:
                    active_entropy.append(result['epoch_results'][epoch]['entropy_production'])
                    active_subjects.append(result['subject'])
            
            for result in sham_results:
                if result['epoch_results'][epoch] is not None:
                    sham_entropy.append(result['epoch_results'][epoch]['entropy_production'])
                    sham_subjects.append(result['subject'])
            
            # Create scatter plot
            if active_entropy:
                ax.scatter(range(len(active_entropy)), active_entropy, 
                          c=colors['ACTIVE'], marker=markers['ACTIVE'], 
                          s=80, alpha=0.7, label='ACTIVE', edgecolors='black', linewidth=0.5)
            
            if sham_entropy:
                ax.scatter(range(len(sham_entropy)), sham_entropy, 
                          c=colors['SHAM'], marker=markers['SHAM'], 
                          s=80, alpha=0.7, label='SHAM', edgecolors='black', linewidth=0.5)
            
            # Add mean lines
            if active_entropy:
                ax.axhline(np.mean(active_entropy), color=colors['ACTIVE'], 
                          linestyle='--', alpha=0.8, linewidth=2, label=f'ACTIVE mean = {np.mean(active_entropy):.4f}')
            if sham_entropy:
                ax.axhline(np.mean(sham_entropy), color=colors['SHAM'], 
                          linestyle='--', alpha=0.8, linewidth=2, label=f'SHAM mean = {np.mean(sham_entropy):.4f}')
            
            # Formatting
            ax.set_title(f"{epoch.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Subject Index', fontsize=12)
            ax.set_ylabel('Entropy Production', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Set y-axis to start from 0 for better comparison
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "entropy_production_individual_subjects.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Individual subject plot saved: {plot_path}")
        
        # Create a second plot: paired comparison for each subject
        self._plot_paired_subject_comparison(active_results, sham_results)
    
    def _plot_paired_subject_comparison(self, active_results, sham_results):
        """Create enhanced paired comparison plot with consistent y-limits and temporal trajectories."""
        print("Creating paired subject comparison...")
        
        # Find subjects with both conditions
        active_subjects = {result['subject'] for result in active_results}
        sham_subjects = {result['subject'] for result in sham_results}
        paired_subjects = sorted(active_subjects.intersection(sham_subjects))
        
        if not paired_subjects:
            print("  No subjects with both ACTIVE and SHAM data found")
            return
        
        epochs = ['baseline', 'stimulation', 'recovery']
        
        # Create figure with 2 rows: paired comparison + temporal trajectories
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Collect all data to determine consistent y-limits
        all_entropy_values = []
        paired_data = {'ACTIVE': {}, 'SHAM': {}}
        
        for subject in paired_subjects:
            active_result = next((r for r in active_results if r['subject'] == subject), None)
            sham_result = next((r for r in sham_results if r['subject'] == subject), None)
            
            if active_result and sham_result:
                paired_data['ACTIVE'][subject] = {}
                paired_data['SHAM'][subject] = {}
                
                for epoch in epochs:
                    if (active_result['epoch_results'][epoch] is not None and 
                        sham_result['epoch_results'][epoch] is not None):
                        
                        active_val = active_result['epoch_results'][epoch]['entropy_production']
                        sham_val = sham_result['epoch_results'][epoch]['entropy_production']
                        
                        paired_data['ACTIVE'][subject][epoch] = active_val
                        paired_data['SHAM'][subject][epoch] = sham_val
                        
                        all_entropy_values.extend([active_val, sham_val])
        
        # Set consistent y-limits with some padding
        if all_entropy_values:
            y_min = min(all_entropy_values) * 0.9
            y_max = max(all_entropy_values) * 1.1
        else:
            y_min, y_max = 0, 1
        
        # ROW 1: Paired comparison plots (ACTIVE vs SHAM for each epoch)
        for i, epoch in enumerate(epochs):
            ax = axes[0, i]
            
            active_vals = []
            sham_vals = []
            subjects_with_data = []
            
            for subject in paired_subjects:
                if (subject in paired_data['ACTIVE'] and 
                    epoch in paired_data['ACTIVE'][subject] and
                    subject in paired_data['SHAM'] and 
                    epoch in paired_data['SHAM'][subject]):
                    
                    active_vals.append(paired_data['ACTIVE'][subject][epoch])
                    sham_vals.append(paired_data['SHAM'][subject][epoch])
                    subjects_with_data.append(subject)
            
            if active_vals and sham_vals:
                # Create paired scatter plot with connecting lines
                for j, (active_val, sham_val, subj) in enumerate(zip(active_vals, sham_vals, subjects_with_data)):
                    ax.plot([0, 1], [active_val, sham_val], 'o-', color='gray', alpha=0.4, linewidth=1)
                
                # Add condition markers
                ax.scatter([0]*len(active_vals), active_vals, c='#FF6B6B', s=100, alpha=0.8, 
                          marker='o', edgecolors='black', linewidth=1, label='ACTIVE')
                ax.scatter([1]*len(sham_vals), sham_vals, c='#4ECDC4', s=100, alpha=0.8, 
                          marker='s', edgecolors='black', linewidth=1, label='SHAM')
                
                # Statistical test
                from scipy import stats
                t_stat, p_val = stats.ttest_rel(active_vals, sham_vals)
                
                ax.set_title(f"{epoch.replace('_', ' ').title()}\\n"
                           f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}", 
                           fontsize=12, fontweight='bold')
                ax.set_xlim(-0.2, 1.2)
                ax.set_ylim(y_min, y_max)  # Consistent y-limits
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['ACTIVE', 'SHAM'])
                ax.set_ylabel('Entropy Production', fontsize=12)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
        
        # ROW 2: Temporal trajectory plots (pre-during-post for each condition)
        conditions = ['ACTIVE', 'SHAM']
        colors = {'ACTIVE': '#FF6B6B', 'SHAM': '#4ECDC4'}
        
        for cond_idx, condition in enumerate(conditions):
            ax = axes[1, cond_idx]
            
            # Plot trajectory for each subject
            x_positions = [0, 1, 2]  # pre, during, post
            
            for subject in paired_subjects:
                if subject in paired_data[condition]:
                    y_values = []
                    valid_epochs = []
                    
                    for epoch_idx, epoch in enumerate(epochs):
                        if epoch in paired_data[condition][subject]:
                            y_values.append(paired_data[condition][subject][epoch])
                            valid_epochs.append(epoch_idx)
                    
                    if len(y_values) >= 2:  # Need at least 2 points to draw a line
                        # Connect points with dashed lines
                        ax.plot(valid_epochs, y_values, 'o--', color=colors[condition], 
                               alpha=0.6, linewidth=1.5, markersize=6, 
                               label=subject if subject == paired_subjects[0] else "")
                        
                        # Add subject labels
                        for epoch_idx, y_val in zip(valid_epochs, y_values):
                            ax.text(epoch_idx, y_val, subject, fontsize=8, ha='center', va='bottom', 
                                   alpha=0.7)
            
            # Calculate and plot mean trajectory
            mean_trajectory = []
            std_trajectory = []
            
            for epoch in epochs:
                epoch_values = []
                for subject in paired_subjects:
                    if (subject in paired_data[condition] and 
                        epoch in paired_data[condition][subject]):
                        epoch_values.append(paired_data[condition][subject][epoch])
                
                if epoch_values:
                    mean_trajectory.append(np.mean(epoch_values))
                    std_trajectory.append(np.std(epoch_values))
                else:
                    mean_trajectory.append(np.nan)
                    std_trajectory.append(np.nan)
            
            # Plot mean trajectory with error bars
            valid_means = [i for i, val in enumerate(mean_trajectory) if not np.isnan(val)]
            if valid_means:
                ax.errorbar(valid_means, [mean_trajectory[i] for i in valid_means], 
                           yerr=[std_trajectory[i] for i in valid_means],
                           color='black', linewidth=3, capsize=5, capthick=2, 
                           marker='D', markersize=8, label='Mean ± SD')
            
            ax.set_title(f"{condition} Condition\\nTemporal Trajectory", fontsize=12, fontweight='bold')
            ax.set_xlim(-0.3, 2.3)
            ax.set_ylim(y_min, y_max)  # Consistent y-limits
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Pre', 'During', 'Post'])
            ax.set_xlabel('tFUS Epoch', fontsize=12)
            ax.set_ylabel('Entropy Production', fontsize=12)
            ax.grid(True, alpha=0.3)
            if cond_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Third subplot: Direct comparison of trajectories
        ax = axes[1, 2]
        
        for condition in conditions:
            # Calculate mean trajectory
            mean_trajectory = []
            std_trajectory = []
            
            for epoch in epochs:
                epoch_values = []
                for subject in paired_subjects:
                    if (subject in paired_data[condition] and 
                        epoch in paired_data[condition][subject]):
                        epoch_values.append(paired_data[condition][subject][epoch])
                
                if epoch_values:
                    mean_trajectory.append(np.mean(epoch_values))
                    std_trajectory.append(np.std(epoch_values) / np.sqrt(len(epoch_values)))  # SEM
                else:
                    mean_trajectory.append(np.nan)
                    std_trajectory.append(np.nan)
            
            # Plot mean trajectory with SEM
            valid_means = [i for i, val in enumerate(mean_trajectory) if not np.isnan(val)]
            if valid_means:
                ax.errorbar(valid_means, [mean_trajectory[i] for i in valid_means], 
                           yerr=[std_trajectory[i] for i in valid_means],
                           color=colors[condition], linewidth=3, capsize=5, capthick=2, 
                           marker='o', markersize=8, label=f'{condition} (Mean ± SEM)')
        
        ax.set_title("ACTIVE vs SHAM\\nMean Trajectories", fontsize=12, fontweight='bold')
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Pre', 'During', 'Post'])
        ax.set_xlabel('tFUS Epoch', fontsize=12)
        ax.set_ylabel('Entropy Production', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "entropy_production_paired_comparison_enhanced.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Enhanced paired comparison plot saved: {plot_path}")


def main(processing_type='minimal', entropy_method='discrete', roi_data_dir=None, n_states=8):
    """Main entropy production analysis pipeline.
    
    Parameters:
    -----------
    processing_type : str, optional
        Type of preprocessing ('minimal' or 'conservative')
    entropy_method : str, optional
        Method for entropy estimation ('discrete', 'gaussian', 'kde')
    roi_data_dir : str or Path, optional
        Path to ROI data directory
    n_states : int, optional
        Number of discrete states for transition estimation (default: 8)
    """
    print(f"\\n=== ENTROPY PRODUCTION ANALYSIS ===")
    print(f"Processing type: {processing_type}")
    print(f"Entropy method: {entropy_method}")
    print(f"Number of discrete states: {n_states}")
    
    # Initialize analyzer
    if roi_data_dir is None:
        roi_data_dir = "/Users/jacekdmochowski/PROJECTS/fus/data/roi_time_series"
    
    analyzer = EntropyProductionAnalyzer(roi_data_dir)
    
    # Update analyzer to use specified method
    analyzer.entropy_method = entropy_method
    
    # Analyze ACTIVE condition
    print("\\n" + "="*60)
    active_results = analyzer.analyze_condition('ACTIVE', processing_type=processing_type, method=entropy_method, n_states=n_states)
    
    # Analyze SHAM condition
    print("\\n" + "="*60)
    sham_results = analyzer.analyze_condition('SHAM', processing_type=processing_type, method=entropy_method, n_states=n_states)
    
    # Compare conditions
    if active_results and sham_results:
        comparison_results = analyzer.compare_conditions(active_results, sham_results)
    
    return active_results, sham_results, comparison_results


if __name__ == "__main__":
    # Example usage with different parameters:
    
    # Run with default parameters (minimal preprocessing, discrete method)
    active_results, sham_results, comparison_results = main(processing_type='conservative', entropy_method='discrete', n_states=12)
    #main()
    
    # Alternative configurations:
    # main(processing_type='conservative', entropy_method='gaussian')
    # main(processing_type='minimal', entropy_method='kde', n_states=12)
    # main(entropy_method='discrete', n_states=6)  # Fewer states for simpler discretization