import json
import os
from datetime import datetime
from collections import Counter

class SessionAnalyser:
    """
    Simple and comprehensive session feature extractor for anomaly detection research.
    Processes sessions.json and extracts rich features for each session.
    """
    
    def __init__(self):
        self.sessions_data = {}
        self.session_features = {}
        self.component_mapping = {}  # For numeric encoding
        
    def load_sessions(self, sessions_file_path):
        """Load sessions from JSON file."""
        print(f"Loading sessions from {sessions_file_path}")
        
        with open(sessions_file_path, 'r') as f:
            self.sessions_data = json.load(f)
            
        print(f"Loaded {len(self.sessions_data)} sessions")
        return self
    
    def analyse_all_sessions(self):
        """Extract features from all sessions."""
        print("Analysing session features...")
        
        # First pass: create component mapping for numeric encoding
        self._create_component_mapping()
        
        # Second pass: extract features for each session
        total_sessions = len(self.sessions_data)
        
        for i, (session_id, logs) in enumerate(self.sessions_data.items()):
            if i % 100 == 0:
                print(f"  Processed {i}/{total_sessions} sessions...")
                
            self.session_features[session_id] = self._extract_session_features(logs)
            
        print(f"Feature extraction complete for {len(self.session_features)} sessions")
        
    def _create_component_mapping(self):
        """Create numeric mapping for components across all sessions."""
        all_components = set()
        
        for logs in self.sessions_data.values():
            for log in logs:
                all_components.add(log['component'])
                
        # Create numeric mapping
        self.component_mapping = {comp: idx for idx, comp in enumerate(sorted(all_components))}
        print(f"Found {len(self.component_mapping)} unique components")
        
    def _extract_session_features(self, logs):
        """Extract comprehensive features from a single session."""
        if not logs:
            return None
            
        # Sort logs by line number to maintain sequence
        sorted_logs = sorted(logs, key=lambda x: x['line_number'])
        
        # 1. Template frequency
        template_frequency = dict(Counter(log['template_id'] for log in sorted_logs))
        
        # 2. Template transitions (sequence)
        template_sequence = [log['template_id'] for log in sorted_logs]
        
        # 3. Time gaps calculation
        time_gaps = self._calculate_time_gaps(sorted_logs)
        
        # 4. Session duration and length
        session_length = len(sorted_logs)
        session_duration = self._calculate_duration(sorted_logs)
        
        # 5. All original messages for NLP
        all_messages = [log['original_message'] for log in sorted_logs]
        
        # 6. Component frequency (numeric vector)
        component_frequency = self._encode_components(sorted_logs)
        
        # 7. Log levels distribution
        log_levels = dict(Counter(log['level'] for log in sorted_logs))
        
        # 8. Thread IDs
        thread_ids = [log['thread_id'] for log in sorted_logs]
        
        return {
            'template_frequency': template_frequency,
            'template_sequence': template_sequence,
            'time_gaps': time_gaps,
            'session_length': session_length,
            'session_duration': session_duration,
            'all_messages': all_messages,
            'component_frequency': component_frequency,
            'log_levels': log_levels,
            'thread_ids': thread_ids,
            # Additional metadata
            'first_timestamp': sorted_logs[0]['timestamp'],
            'last_timestamp': sorted_logs[-1]['timestamp'],
            'unique_templates': len(set(template_sequence)),
            'unique_components': len(set(log['component'] for log in sorted_logs)),
            'unique_threads': len(set(thread_ids))
        }
    
    def _calculate_time_gaps(self, sorted_logs):
        """Calculate time gaps between consecutive log entries."""
        if len(sorted_logs) < 2:
            return []
            
        time_gaps = []
        
        for i in range(1, len(sorted_logs)):
            try:
                # Parse timestamps (format: "081109 203518")
                prev_time = self._parse_timestamp(sorted_logs[i-1]['timestamp'])
                curr_time = self._parse_timestamp(sorted_logs[i]['timestamp'])
                
                gap = (curr_time - prev_time).total_seconds()
                time_gaps.append(gap)
                
            except Exception:
                # If timestamp parsing fails, use 0 gap
                time_gaps.append(0.0)
                
        return time_gaps
    
    def _parse_timestamp(self, timestamp_str):
        """Parse HDFS timestamp format: '081109 203518'"""
        try:
            # Assume year 2008 for the logs (common for HDFS dataset)
            date_part, time_part = timestamp_str.split()
            
            # Parse date: MMDDYY format (assuming 08 prefix means 2008)
            month = int(date_part[:2])
            day = int(date_part[2:4])
            year = 2000 + int(date_part[4:6])
            
            # Parse time: HHMMSS format
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            second = int(time_part[4:6])
            
            return datetime(year, month, day, hour, minute, second)
            
        except Exception:
            # Return epoch if parsing fails
            return datetime(1970, 1, 1)
    
    def _calculate_duration(self, sorted_logs):
        """Calculate total session duration in seconds."""
        if len(sorted_logs) < 2:
            return 0.0
            
        try:
            first_time = self._parse_timestamp(sorted_logs[0]['timestamp'])
            last_time = self._parse_timestamp(sorted_logs[-1]['timestamp'])
            
            return (last_time - first_time).total_seconds()
            
        except Exception:
            return 0.0
    
    def _encode_components(self, sorted_logs):
        """Encode components as frequency vector using numeric mapping."""
        component_counts = Counter(log['component'] for log in sorted_logs)
        
        # Create frequency vector
        frequency_vector = [0] * len(self.component_mapping)
        
        for component, count in component_counts.items():
            if component in self.component_mapping:
                idx = self.component_mapping[component]
                frequency_vector[idx] = count
                
        return frequency_vector
    
    def save_features(self, output_file="results/session_features.json"):
        """Save extracted features to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data with metadata
        output_data = {
            'metadata': {
                'total_sessions': len(self.session_features),
                'component_mapping': self.component_mapping,
                'features_extracted': [
                    'template_frequency', 'template_sequence', 'time_gaps',
                    'session_length', 'session_duration', 'all_messages',
                    'component_frequency', 'log_levels', 'thread_ids'
                ]
            },
            'session_features': self.session_features
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Session features saved to {output_file}")
        
    def get_summary_statistics(self):
        """Generate summary statistics of extracted features."""
        if not self.session_features:
            return {}
            
        # Collect statistics
        session_lengths = []
        session_durations = []
        unique_templates_per_session = []
        unique_components_per_session = []
        
        for features in self.session_features.values():
            session_lengths.append(features['session_length'])
            session_durations.append(features['session_duration'])
            unique_templates_per_session.append(features['unique_templates'])
            unique_components_per_session.append(features['unique_components'])
        
        import statistics
        
        summary = {
            'total_sessions_analysed': len(self.session_features),
            'session_length_stats': {
                'min': min(session_lengths),
                'max': max(session_lengths),
                'mean': round(statistics.mean(session_lengths), 2),
                'median': statistics.median(session_lengths)
            },
            'session_duration_stats': {
                'min_seconds': min(session_durations),
                'max_seconds': max(session_durations),
                'mean_seconds': round(statistics.mean(session_durations), 2),
                'median_seconds': statistics.median(session_durations)
            },
            'template_diversity_stats': {
                'min_unique_templates': min(unique_templates_per_session),
                'max_unique_templates': max(unique_templates_per_session),
                'mean_unique_templates': round(statistics.mean(unique_templates_per_session), 2)
            },
            'component_diversity_stats': {
                'min_unique_components': min(unique_components_per_session),
                'max_unique_components': max(unique_components_per_session),
                'mean_unique_components': round(statistics.mean(unique_components_per_session), 2)
            },
            'total_unique_components_across_dataset': len(self.component_mapping)
        }
        
        return summary
    
    def show_sample_features(self, n=3):
        """Display sample session features for inspection."""
        print(f"\nSample Session Features (showing {n} sessions):")
        print("=" * 60)
        
        sample_sessions = list(self.session_features.items())[:n]
        
        for i, (session_id, features) in enumerate(sample_sessions):
            print(f"\nSession {i+1}: {session_id}")
            print(f"  Length: {features['session_length']} logs")
            print(f"  Duration: {features['session_duration']:.2f} seconds")
            print(f"  Unique templates: {features['unique_templates']}")
            print(f"  Template sequence: {features['template_sequence'][:10]}{'...' if len(features['template_sequence']) > 10 else ''}")
            print(f"  Time gaps (first 5): {features['time_gaps'][:5]}")
            print(f"  Log levels: {features['log_levels']}")
            print(f"  Unique components: {features['unique_components']}")

def main():
    """Main function to analyse sessions and extract features."""
    analyser = SessionAnalyser()
    
    # Load and analyse sessions
    analyser.load_sessions("results/sessions.json")
    analyser.analyse_all_sessions()
    
    # Show sample features
    analyser.show_sample_features(3)
    
    # Show summary statistics
    summary = analyser.get_summary_statistics()
    print(f"\nSummary Statistics:")
    print("=" * 30)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save features
    analyser.save_features("results/session_features.json")
    
    print(f"\nFeature extraction complete!")


if __name__ == "__main__":
    main()