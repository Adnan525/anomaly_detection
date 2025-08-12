import json
import os
import numpy as np
from collections import Counter
import math

class FeatureEngineer:
    """
    Converts semi-structured session features into ML-ready structured features.
    Takes session_features.json and creates a clean numeric dataset.
    """
    
    def __init__(self):
        self.session_features = {}
        self.metadata = {}
        self.ml_features = {}
        self.feature_names = []
        
    def load_session_features(self, session_features_file):
        """Load session features from JSON file."""
        print(f"Loading session features from {session_features_file}")
        
        with open(session_features_file, 'r') as f:
            data = json.load(f)
            
        self.session_features = data['session_features']
        self.metadata = data['metadata']
        
        print(f"Loaded {len(self.session_features)} sessions for feature engineering")
        return self
        
    def engineer_features(self):
        """Convert all semi-structured features to structured ML-ready features."""
        print("Engineering structured features from semi-structured data...")
        
        total_sessions = len(self.session_features)
        
        for i, (session_id, features) in enumerate(self.session_features.items()):
            if i % 100 == 0:
                print(f"  Processed {i}/{total_sessions} sessions...")
                
            # Extract structured features for this session
            ml_features = self._extract_ml_features(features)
            self.ml_features[session_id] = ml_features
            
        # Create feature names list (from first session)
        if self.ml_features:
            first_session = list(self.ml_features.values())[0]
            self.feature_names = list(first_session.keys())
            
        print(f"Feature engineering complete!")
        print(f"Generated {len(self.feature_names)} structured features per session")
        
    def _extract_ml_features(self, session_features):
        """Extract all ML-ready features from one session."""
        ml_features = {}
        
        # 1. Already structured features (pass through)
        ml_features.update(self._get_structured_features(session_features))
        
        # 2. Template features (from semi-structured)
        ml_features.update(self._extract_template_features(session_features))
        
        # 3. Sequence features (from semi-structured)
        ml_features.update(self._extract_sequence_features(session_features))
        
        # 4. Temporal features (from semi-structured)
        ml_features.update(self._extract_temporal_features(session_features))
        
        # 5. Log level features (from semi-structured)
        ml_features.update(self._extract_log_level_features(session_features))
        
        # 6. Component features (already structured vector)
        ml_features.update(self._extract_component_features(session_features))
        
        return ml_features
        
    def _get_structured_features(self, features):
        """Extract already structured numeric features."""
        structured = {
            'session_length': int(features['session_length']),
            'session_duration': float(features['session_duration']),
            'unique_templates': int(features['unique_templates']),
            'unique_components': int(features['unique_components']),
            'unique_threads': int(features['unique_threads'])
        }
        return structured
        
    def _extract_template_features(self, features):
        """Convert template_frequency dict to structured features."""
        template_freq = features['template_frequency']
        session_length = features['session_length']
        
        if not template_freq or session_length == 0:
            return {
                'template_diversity_score': 0.0,
                'most_frequent_template_ratio': 0.0,
                'template_entropy': 0.0,
                'template_gini_coefficient': 0.0,
                'rare_template_count': 0
            }
            
        # Template diversity
        template_diversity = len(template_freq) / session_length
        
        # Most frequent template ratio
        max_freq = max(template_freq.values())
        most_frequent_ratio = max_freq / session_length
        
        # Template entropy (information diversity)
        template_entropy = self._calculate_entropy(list(template_freq.values()))
        
        # Gini coefficient (inequality measure)
        gini_coeff = self._calculate_gini_coefficient(list(template_freq.values()))
        
        # Rare templates (appearing only once)
        rare_template_count = sum(1 for freq in template_freq.values() if freq == 1)
        
        return {
            'template_diversity_score': float(template_diversity),
            'most_frequent_template_ratio': float(most_frequent_ratio),
            'template_entropy': float(template_entropy),
            'template_gini_coefficient': float(gini_coeff),
            'rare_template_count': int(rare_template_count)
        }
        
    def _extract_sequence_features(self, features):
        """Convert template_sequence array to structured features."""
        sequence = features['template_sequence']
        
        if len(sequence) < 2:
            return {
                'sequence_complexity': 0,
                'transition_entropy': 0.0,
                'repeated_transitions': 0,
                'sequence_periodicity': 0.0,
                'longest_repeated_pattern': 0
            }
            
        # Unique transitions count
        transitions = list(zip(sequence[:-1], sequence[1:]))
        sequence_complexity = len(set(transitions))
        
        # Transition entropy
        transition_counts = list(Counter(transitions).values())
        transition_entropy = self._calculate_entropy(transition_counts)
        
        # Repeated transitions
        repeated_transitions = len(transitions) - len(set(transitions))
        
        # Sequence periodicity (how repetitive is the pattern)
        periodicity = self._calculate_periodicity(sequence)
        
        # Longest repeated consecutive pattern
        longest_pattern = self._find_longest_repeated_pattern(sequence)
        
        return {
            'sequence_complexity': int(sequence_complexity),
            'transition_entropy': float(transition_entropy),
            'repeated_transitions': int(repeated_transitions),
            'sequence_periodicity': float(periodicity),
            'longest_repeated_pattern': int(longest_pattern)
        }
        
    def _extract_temporal_features(self, features):
        """Convert time_gaps array to structured features."""
        time_gaps = features['time_gaps']
        
        if not time_gaps:
            return {
                'mean_time_gap': 0.0,
                'std_time_gap': 0.0,
                'max_time_gap': 0.0,
                'min_time_gap': 0.0,
                'time_gap_variance': 0.0,
                'time_gap_skewness': 0.0,
                'long_gap_count': 0
            }
            
        gaps = np.array(time_gaps)
        
        # Basic statistics
        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        max_gap = float(np.max(gaps))
        min_gap = float(np.min(gaps))
        variance_gap = float(np.var(gaps))
        
        # Skewness (asymmetry of distribution)
        skewness = float(self._calculate_skewness(gaps))
        
        # Count of unusually long gaps (> 2 standard deviations)
        if std_gap > 0:
            threshold = mean_gap + 2 * std_gap
            long_gap_count = int(np.sum(gaps > threshold))
        else:
            long_gap_count = 0
            
        return {
            'mean_time_gap': mean_gap,
            'std_time_gap': std_gap,
            'max_time_gap': max_gap,
            'min_time_gap': min_gap,
            'time_gap_variance': variance_gap,
            'time_gap_skewness': skewness,
            'long_gap_count': long_gap_count
        }
        
    def _extract_log_level_features(self, features):
        """Convert log_levels dict to structured features."""
        log_levels = features['log_levels']
        session_length = features['session_length']
        
        # Standard log levels to look for
        standard_levels = ['INFO', 'WARN', 'ERROR', 'DEBUG', 'FATAL']
        
        level_features = {}
        
        # Count and ratio for each standard level
        for level in standard_levels:
            count = log_levels.get(level, 0)
            ratio = count / session_length if session_length > 0 else 0.0
            
            level_features[f'{level.lower()}_count'] = int(count)
            level_features[f'{level.lower()}_ratio'] = float(ratio)
            
        # Log level diversity
        level_features['log_level_diversity'] = len(log_levels)
        
        # Error indicators (WARN + ERROR + FATAL)
        error_count = log_levels.get('WARN', 0) + log_levels.get('ERROR', 0) + log_levels.get('FATAL', 0)
        level_features['error_indicator_count'] = int(error_count)
        level_features['error_indicator_ratio'] = float(error_count / session_length) if session_length > 0 else 0.0
        
        return level_features
        
    def _extract_component_features(self, features):
        """Convert component_frequency vector to named features."""
        component_freq = features['component_frequency']
        component_mapping = self.metadata['component_mapping']
        
        # Create reverse mapping (index -> component name)
        reverse_mapping = {idx: comp for comp, idx in component_mapping.items()}
        
        component_features = {}
        
        # Named component frequencies
        for idx, freq in enumerate(component_freq):
            if idx in reverse_mapping:
                comp_name = reverse_mapping[idx]
                # Clean component name for feature name
                clean_name = comp_name.replace('.', '_').replace('$', '_').lower()
                component_features[f'comp_{clean_name}_freq'] = int(freq)
                
        # Component diversity metrics
        active_components = sum(1 for freq in component_freq if freq > 0)
        total_component_freq = sum(component_freq)
        
        component_features['active_component_count'] = int(active_components)
        component_features['component_concentration'] = float(max(component_freq) / total_component_freq) if total_component_freq > 0 else 0.0
        
        return component_features
        
    # Helper methods for calculations
    def _calculate_entropy(self, values):
        """Calculate Shannon entropy."""
        if not values or sum(values) == 0:
            return 0.0
            
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy
        
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient (inequality measure)."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        if cumsum[-1] == 0:
            return 0.0
            
        gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_values))) / (n * cumsum[-1])
        return max(0.0, gini)
        
    def _calculate_periodicity(self, sequence):
        """Calculate how periodic/repetitive a sequence is."""
        if len(sequence) < 4:
            return 0.0
            
        # Look for repeating patterns of length 2, 3, 4
        max_periodicity = 0.0
        
        for pattern_length in range(2, min(len(sequence) // 2 + 1, 5)):
            pattern_matches = 0
            total_patterns = len(sequence) - pattern_length + 1
            
            for i in range(len(sequence) - 2 * pattern_length + 1):
                pattern1 = sequence[i:i + pattern_length]
                pattern2 = sequence[i + pattern_length:i + 2 * pattern_length]
                
                if pattern1 == pattern2:
                    pattern_matches += 1
                    
            periodicity = pattern_matches / total_patterns if total_patterns > 0 else 0.0
            max_periodicity = max(max_periodicity, periodicity)
            
        return max_periodicity
        
    def _find_longest_repeated_pattern(self, sequence):
        """Find the length of the longest immediately repeated pattern."""
        if len(sequence) < 2:
            return 0
            
        max_length = 0
        
        for i in range(len(sequence) - 1):
            current_length = 0
            j = i
            
            while j + 1 < len(sequence) and sequence[j] == sequence[j + 1]:
                current_length += 1
                j += 1
                
            max_length = max(max_length, current_length)
            
        return max_length
        
    def _calculate_skewness(self, values):
        """Calculate skewness of a distribution."""
        if len(values) < 3:
            return 0.0
            
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
            
        skewness = np.mean(((values - mean_val) / std_val) ** 3)
        return skewness
        
    def save_ml_dataset(self, output_file="results/ml_ready_dataset.json"):
        """Save ML-ready dataset to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare ML-ready dataset
        ml_dataset = {
            'metadata': {
                'total_sessions': len(self.ml_features),
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'original_component_mapping': self.metadata.get('component_mapping', {}),
                'feature_categories': {
                    'structured_features': ['session_length', 'session_duration', 'unique_templates', 'unique_components', 'unique_threads'],
                    'template_features': [f for f in self.feature_names if 'template' in f],
                    'sequence_features': [f for f in self.feature_names if 'sequence' in f or 'transition' in f],
                    'temporal_features': [f for f in self.feature_names if 'time_gap' in f],
                    'log_level_features': [f for f in self.feature_names if any(level in f for level in ['info', 'warn', 'error', 'debug', 'fatal'])],
                    'component_features': [f for f in self.feature_names if f.startswith('comp_')]
                }
            },
            'ml_features': self.ml_features
        }
        
        with open(output_file, 'w') as f:
            json.dump(ml_dataset, f, indent=2)
            
        print(f"ML-ready dataset saved to {output_file}")
        
    def get_feature_summary(self):
        """Generate summary of engineered features."""
        if not self.ml_features:
            return {}
            
        # Sample one session to get feature statistics
        sample_session = list(self.ml_features.values())[0]
        
        feature_categories = {}
        for feature_name in sample_session.keys():
            if 'template' in feature_name:
                category = 'template_features'
            elif 'sequence' in feature_name or 'transition' in feature_name:
                category = 'sequence_features'
            elif 'time_gap' in feature_name:
                category = 'temporal_features'
            elif any(level in feature_name for level in ['info', 'warn', 'error', 'debug', 'fatal']):
                category = 'log_level_features'
            elif feature_name.startswith('comp_'):
                category = 'component_features'
            else:
                category = 'basic_features'
                
            if category not in feature_categories:
                feature_categories[category] = []
            feature_categories[category].append(feature_name)
            
        summary = {
            'total_sessions_processed': len(self.ml_features),
            'total_features_per_session': len(self.feature_names),
            'feature_categories': feature_categories,
            'feature_category_counts': {cat: len(features) for cat, features in feature_categories.items()}
        }
        
        return summary
        
    def show_sample_features(self, n=2):
        """Display sample ML features for inspection."""
        print(f"\nSample ML-Ready Features (showing {n} sessions):")
        print("=" * 60)
        
        sample_sessions = list(self.ml_features.items())[:n]
        
        for i, (session_id, features) in enumerate(sample_sessions):
            print(f"\nSession {i+1}: {session_id}")
            print(f"Features ({len(features)} total):")
            
            # Group features by category for better display
            categories = {
                'Basic': [f for f in features.keys() if f in ['session_length', 'session_duration', 'unique_templates', 'unique_components', 'unique_threads']],
                'Template': [f for f in features.keys() if 'template' in f],
                'Sequence': [f for f in features.keys() if 'sequence' in f or 'transition' in f],
                'Temporal': [f for f in features.keys() if 'time_gap' in f],
                'Log Level': [f for f in features.keys() if any(level in f for level in ['info', 'warn', 'error'])],
                'Component': [f for f in features.keys() if f.startswith('comp_')][:5]  # Show first 5 only
            }
            
            for category, feature_list in categories.items():
                if feature_list:
                    print(f"  {category}: {dict((f, features[f]) for f in feature_list[:5])}")

# ====================================================================================================================

def main():
    """Main function to engineer ML-ready features."""
    engineer = FeatureEngineer()
    
    # Load session features and engineer ML features
    engineer.load_session_features("results/session_features.json")
    engineer.engineer_features()
    
    # Show sample features
    engineer.show_sample_features(2)
    
    # Show feature summary
    summary = engineer.get_feature_summary()
    print(f"\nFeature Engineering Summary:")
    print("=" * 40)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save ML dataset
    engineer.save_ml_dataset("results/ml_ready_dataset.json")
    
    print(f"\nFeature engineering complete!")
    print(f"ML-ready dataset saved with {len(engineer.feature_names)} features per session")
    print(f"Ready for anomaly detection model training!")

if __name__ == "__main__":
    main()