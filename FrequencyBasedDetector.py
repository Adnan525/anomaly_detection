import numpy as np
from collections import defaultdict, Counter

class FrequencyBasedDetector:
    def __init__(self):
        """
        This detector will use frequency patterns we discover to identify anomalies
        """
        self.normal_profiles = {}
        self.thresholds = {}
        
    def train(self, normal_sessions):
        """
        Build frequency profiles from normal sessions only
        """
        print("Training frequency-based detector...")
        
        # Build template frequency profile
        template_frequencies = defaultdict(list)
        session_lengths = []
        session_diversities = []
        
        for session in normal_sessions:
            session_template_counts = Counter(session['templates'])
            session_lengths.append(session['length'])
            session_diversities.append(len(set(session['templates'])))
            
            # Track how often each template appears per session
            for template, count in session_template_counts.items():
                template_frequencies[template].append(count)
        
        # Create statistical profiles for each template
        for template, counts in template_frequencies.items():
            self.normal_profiles[template] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'max': np.max(counts),
                'sessions_with_template': len(counts) / len(normal_sessions)
            }
        
        # Create profiles for session-level features
        self.normal_profiles['session_length'] = {
            'mean': np.mean(session_lengths),
            'std': np.std(session_lengths),
            'percentile_95': np.percentile(session_lengths, 95)
        }
        
        self.normal_profiles['session_diversity'] = {
            'mean': np.mean(session_diversities), 
            'std': np.std(session_diversities),
            'percentile_95': np.percentile(session_diversities, 95)
        }
        
        print(f"Built profiles for {len(template_frequencies)} unique templates")
        
    def detect_anomaly(self, test_session):
        """
        Use our frequency profiles to score how anomalous a session looks
        """
        anomaly_score = 0
        anomaly_reasons = []
        
        # Check session length
        length_z_score = abs(test_session['length'] - self.normal_profiles['session_length']['mean']) / max(self.normal_profiles['session_length']['std'], 1)
        if length_z_score > 2:  # More than 2 standard deviations
            anomaly_score += length_z_score
            anomaly_reasons.append(f"Unusual session length: {test_session['length']}")
        
        # Check template frequencies
        session_template_counts = Counter(test_session['templates'])
        
        for template, count in session_template_counts.items():
            if template in self.normal_profiles:
                profile = self.normal_profiles[template]
                expected_mean = profile['mean']
                expected_std = max(profile['std'], 0.1)  # Avoid division by zero
                
                z_score = abs(count - expected_mean) / expected_std
                if z_score > 2:
                    anomaly_score += z_score
                    anomaly_reasons.append(f"Template '{template}' appears {count} times (expected ~{expected_mean:.1f})")
            else:
                # This template never appeared in normal training data
                anomaly_score += 5  # High penalty for unknown templates
                anomaly_reasons.append(f"Unknown template: '{template}'")
        
        return anomaly_score, anomaly_reasons