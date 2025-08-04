import numpy as np
from collections import Counter

class HDFSFrequencyAnalyzer:
    def __init__(self):
        """
        This analyzer will help us understand frequency patterns in HDFS logs
        and compare normal vs anomalous sessions
        """
        self.normal_sessions = []
        self.anomalous_sessions = []
        self.template_categories = {}
        
    def load_sessions(self, analyzer):
        """
        Separate normal and anomalous sessions for comparison
        """
        for session_id, session_logs in analyzer.sessions.items():
            session_templates = [log['log_template'] for log in session_logs]
            
            if analyzer.session_labels.get(session_id) == 'Normal':
                self.normal_sessions.append({
                    'id': session_id,
                    'templates': session_templates,
                    'length': len(session_templates)
                })
            else:
                self.anomalous_sessions.append({
                    'id': session_id, 
                    'templates': session_templates,
                    'length': len(session_templates)
                })
        
        print(f"Loaded {len(self.normal_sessions)} normal and {len(self.anomalous_sessions)} anomalous sessions")
    
    def analyze_frequency_patterns(self):
        """
        This is where we'll discover the key differences between normal and anomalous sessions
        Let's look at several different frequency-based features
        """
        print("=== Frequency Pattern Analysis ===\n")
        
        # 1. Session length distribution
        normal_lengths = [session['length'] for session in self.normal_sessions]
        anomalous_lengths = [session['length'] for session in self.anomalous_sessions]
        
        print(f"Session Length Statistics:")
        print(f"  Normal sessions - Mean: {np.mean(normal_lengths):.1f}, Std: {np.std(normal_lengths):.1f}")
        print(f"  Anomalous sessions - Mean: {np.mean(anomalous_lengths):.1f}, Std: {np.std(anomalous_lengths):.1f}")
        
        # 2. Template frequency analysis
        self._analyze_template_frequencies()
        
        # 3. Template diversity analysis
        self._analyze_template_diversity()
        
    def _analyze_template_frequencies(self):
        """
        Compare how often different templates appear in normal vs anomalous sessions
        """
        print(f"\n=== Template Frequency Analysis ===")
        
        # Count template occurrences in normal sessions
        normal_template_counts = Counter()
        for session in self.normal_sessions:
            normal_template_counts.update(session['templates'])
        
        # Count template occurrences in anomalous sessions  
        anomalous_template_counts = Counter()
        for session in self.anomalous_sessions:
            anomalous_template_counts.update(session['templates'])
        
        # Find templates that appear much more frequently in anomalous sessions
        print("Templates that might indicate anomalies:")
        for template in anomalous_template_counts:
            anomaly_freq = anomalous_template_counts[template] / len(self.anomalous_sessions)
            normal_freq = normal_template_counts.get(template, 0) / len(self.normal_sessions)
            
            # If this template appears much more often in anomalous sessions
            if anomaly_freq > normal_freq * 2 and anomalous_template_counts[template] > 5:
                print(f"  '{template}' - Normal: {normal_freq:.3f}, Anomaly: {anomaly_freq:.3f}")
    
    def _analyze_template_diversity(self):
        """
        Look at how many unique templates appear in each session type
        """
        print(f"\n=== Template Diversity Analysis ===")
        
        normal_diversities = [len(set(session['templates'])) for session in self.normal_sessions]
        anomalous_diversities = [len(set(session['templates'])) for session in self.anomalous_sessions]
        
        print(f"Unique templates per session:")
        print(f"  Normal sessions - Mean: {np.mean(normal_diversities):.1f}, Std: {np.std(normal_diversities):.1f}")
        print(f"  Anomalous sessions - Mean: {np.mean(anomalous_diversities):.1f}, Std: {np.std(anomalous_diversities):.1f}")