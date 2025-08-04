import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
import time
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

class LogDatasetAnalyser:
    """
    Enhanced log analysis class with intelligent sampling capabilities.
    This demonstrates how to work efficiently with large datasets during research development.
    """

    def __init__(self, dataset_name: str = "HDFS"):
        self.dataset_name = dataset_name
        self.raw_log_count = 0
        self.total_lines_in_file = 0  # Track the full dataset size
        self.sampling_ratio = 1.0     # Track what portion we actually processed
        self.sessions = {}
        self.session_labels = {}
        self.templates = {}
        self.template_stats = {}
        self.parsed_logs = []
        

    def load_hdfs_data(self, 
                       log_file_path: str, 
                       label_file_path: str = None, 
                       max_lines: int = None):
        """
        Load and parse HDFS logs with optional line limiting for efficient research development.
        
        The max_lines parameter allows you to work with a manageable subset of the data
        while developing your algorithms, then scale up to the full dataset when ready.
        """
        print(f"Loading HDFS data from {log_file_path}")
        
        if max_lines:
            print(f"  Limiting analysis to first {max_lines:,} lines for efficient development")
            
        # Parse logs with drain3 (with optional line limit)
        self._parse_with_drain3(log_file_path, max_lines)
        
        # Create sessions from parsed logs
        self._create_sessions()
        
        # Load labels if provided
        if label_file_path and os.path.exists(label_file_path):
            self._load_labels(label_file_path)
            
        print(f"Analysis complete: {len(self.sessions)} sessions with {len(self.templates)} unique templates")
        
        if max_lines and self.total_lines_in_file > max_lines:
            print(f"  Processed {max_lines:,} of {self.total_lines_in_file:,} total lines ({self.sampling_ratio:.1%} sample)")
            print(f"  This sample should capture the majority of template patterns for research development")
            
        return self
        
    def _parse_with_drain3(self, log_file_path, max_lines=None):
        """
        Parse logs using drain3 with optional line limiting.
        
        This method demonstrates how to implement efficient sampling for large-scale
        log analysis research. The key insight is that log data patterns tend to
        repeat, so a representative sample often captures the essential characteristics.
        """
        print("Parsing logs with drain3...")
        
        # Configure drain3 for optimal performance on HDFS logs
        config = TemplateMinerConfig()
        config.drain_sim_th = 0.7    # 70% similarity threshold for grouping messages
        config.drain_depth = 4       # Tree depth for efficient pattern matching
        config.drain_max_children = 100  # Limit branches to prevent memory explosion
        
        template_miner = TemplateMiner(config=config)
        
        # Storage for parsing results
        templates = {}
        template_stats = defaultdict(int)
        parsed_logs = []
        
        print("  Processing log file line by line...")
        
        # First, let's count total lines to understand our sampling ratio
        if max_lines:
            print("    Counting total lines in dataset...")
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.total_lines_in_file = sum(1 for _ in f)
            self.sampling_ratio = min(max_lines / self.total_lines_in_file, 1.0)
            print(f"    Dataset contains {self.total_lines_in_file:,} total lines")
        
        # Now process the lines with our limit
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                # Stop processing if we've reached our limit
                if max_lines and line_num >= max_lines:
                    print(f"    Reached processing limit of {max_lines:,} lines")
                    break
                    
                # Show progress every 50,000 lines for more frequent updates
                if line_num % 50000 == 0 and line_num > 0:
                    print(f"      Processed {line_num:,} lines...")
                    
                # Parse each HDFS log line
                parts = line.strip().split(' ', 4)
                if len(parts) < 5:
                    continue
                    
                try:
                    date, time_part, thread_id, level = parts[:4]
                    remaining = parts[4]
                    
                    # Extract component and message
                    if ':' in remaining:
                        component, message = remaining.split(':', 1)
                        message = message.strip()
                    else:
                        component = "Unknown"
                        message = remaining
                        
                    # Use Drain to extract template patterns
                    result = template_miner.add_log_message(message)
                    template_id = result['cluster_id']
                    template = result['template_mined']
                    
                    # Store results
                    templates[template_id] = template
                    template_stats[template_id] += 1
                    
                    # Create structured log entry
                    parsed_logs.append({
                        'timestamp': f"{date} {time_part}",
                        'thread_id': thread_id,
                        'level': level,
                        'component': component,
                        'content': message,
                        'template_id': template_id,
                        'template': template
                    })
                    
                except Exception:
                    continue  # Skip malformed lines
                    
        # Store results
        self.templates = templates
        self.template_stats = dict(template_stats)
        self.parsed_logs = parsed_logs
        self.raw_log_count = len(parsed_logs)
        
        print(f"  Successfully extracted {len(templates)} unique templates")
        print(f"  Processed {self.raw_log_count:,} log entries")
        print(f"  Achieved {self.raw_log_count / len(templates):.1f}x compression ratio")
        
    def _create_sessions(self):
        """
        Group log entries into sessions for analysis.
        
        This step transforms individual log entries into meaningful sequences
        that represent complete system workflows or time periods.
        """
        print("Creating sessions from parsed logs...")
        
        sessions = {}
        current_session = []
        session_id = 0
        
        for log_entry in self.parsed_logs:
            current_session.append(log_entry)
            
            # Create sessions of manageable size (1000 entries each)
            if len(current_session) >= 1000:
                sessions[f"session_{session_id}"] = current_session
                session_id += 1
                current_session = []
                
        # Include final session if it has entries
        if current_session:
            sessions[f"session_{session_id}"] = current_session
            
        self.sessions = sessions
        print(f"  Created {len(sessions)} sessions for analysis")
        
    def _load_labels(self, label_file_path):
        """Load session labels for supervised evaluation."""
        print("Loading session labels...")
        
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    session_id, label = parts[0].strip(), parts[1].strip()
                    self.session_labels[session_id] = label
                    
        # Report how many labels we have for our sessions
        matching_labels = sum(1 for sid in self.sessions.keys() if sid in self.session_labels)
        print(f"  Found labels for {matching_labels} of our {len(self.sessions)} sessions")
        
    def get_sampling_analysis(self):
        """
        Analyze the characteristics of our sample to understand how representative it is.
        
        This helps you understand whether your sample captures the essential patterns
        needed for developing your frequency-based anomaly detection algorithms.
        """
        analysis = {
            'sampling_info': {
                'total_lines_available': self.total_lines_in_file,
                'lines_processed': self.raw_log_count,
                'sampling_ratio': self.sampling_ratio,
                'estimated_total_templates': len(self.templates) / self.sampling_ratio if self.sampling_ratio > 0 else len(self.templates)
            },
            'template_discovery_curve': self._analyze_template_discovery_rate(),
            'representativeness_metrics': self._calculate_representativeness()
        }
        
        return analysis
        
    def _analyze_template_discovery_rate(self):
        """
        Analyze how quickly we discovered new templates as we processed more data.
        
        This helps determine if we've captured most of the important patterns or if
        processing more data would likely reveal many additional template types.
        """
        # Track template discovery over time
        templates_discovered = []
        unique_templates_seen = set()
        
        # Process logs in order and track when each template was first discovered
        for i, log_entry in enumerate(self.parsed_logs):
            if log_entry['template_id'] not in unique_templates_seen:
                unique_templates_seen.add(log_entry['template_id'])
                templates_discovered.append({
                    'log_position': i,
                    'templates_discovered': len(unique_templates_seen)
                })
                
        return {
            'total_templates_found': len(unique_templates_seen),
            'discovery_curve': templates_discovered[-10:] if templates_discovered else [],  # Last 10 discoveries
            'discovery_rate_slowing': len(templates_discovered) > 10 and 
                                     (templates_discovered[-1]['log_position'] - templates_discovered[-10]['log_position']) > 
                                     (templates_discovered[-10]['log_position'] - templates_discovered[-20]['log_position'] if len(templates_discovered) > 20 else 0)
        }
        
    def _calculate_representativeness(self):
        """
        Calculate metrics that help assess how representative our sample is.
        """
        template_counts = list(self.template_stats.values())
        
        return {
            'template_frequency_distribution': {
                'most_common_frequency': max(template_counts),
                'least_common_frequency': min(template_counts),
                'median_frequency': np.median(template_counts),
                'frequency_std': np.std(template_counts)
            },
            'coverage_analysis': {
                'templates_with_single_occurrence': sum(1 for count in template_counts if count == 1),
                'templates_with_high_frequency': sum(1 for count in template_counts if count > np.percentile(template_counts, 90))
            }
        }

    def get_comprehensive_stats(self):
        """Calculate comprehensive statistics for frequency-based research."""
        
        template_counts = list(self.template_stats.values())
        total_occurrences = sum(template_counts)
        
        session_lengths = [len(session) for session in self.sessions.values()]
        session_diversities = [len(set(entry['template_id'] for entry in session)) 
                              for session in self.sessions.values()]
        
        sorted_template_counts = sorted(template_counts, reverse=True)
        top_10_percent_count = len(sorted_template_counts) // 10
        top_10_percent_coverage = sum(sorted_template_counts[:top_10_percent_count]) / total_occurrences
        
        stats = {
            'data_overview': {
                'total_log_entries': self.raw_log_count,
                'unique_templates': len(self.templates),
                'total_sessions': len(self.sessions),
                'labeled_sessions': len(self.session_labels),
                'compression_ratio': self.raw_log_count / len(self.templates),
                'sampling_ratio': self.sampling_ratio
            },
            'template_distribution': {
                'total_template_occurrences': total_occurrences,
                'most_frequent_template_count': max(template_counts),
                'least_frequent_template_count': min(template_counts),
                'top_10_percent_coverage': top_10_percent_coverage,
                'median_template_frequency': np.median(template_counts),
                'template_frequency_std': np.std(template_counts)
            },
            'session_characteristics': {
                'length_stats': {
                    'mean': np.mean(session_lengths),
                    'std': np.std(session_lengths),
                    'min': np.min(session_lengths),
                    'max': np.max(session_lengths),
                    'percentile_95': np.percentile(session_lengths, 95),
                    'percentile_99': np.percentile(session_lengths, 99)
                },
                'diversity_stats': {
                    'mean_unique_templates_per_session': np.mean(session_diversities),
                    'std_unique_templates_per_session': np.std(session_diversities),
                    'max_unique_templates_per_session': np.max(session_diversities),
                    'min_unique_templates_per_session': np.min(session_diversities)
                }
            }
        }
        
        return stats
        
    def show_template_examples(self, top_n=15):
        """Display the most frequent templates with examples."""
        print(f"\nTop {top_n} Most Frequent Templates:")
        print("=" * 80)
        
        sorted_templates = sorted(self.template_stats.items(), key=lambda x: x[1], reverse=True)
        total_occurrences = sum(self.template_stats.values())
        
        for i, (template_id, count) in enumerate(sorted_templates[:top_n]):
            percentage = (count / total_occurrences) * 100
            template_text = self.templates[template_id]
            
            print(f"{i+1:2d}. ({count:6,} times, {percentage:5.1f}%) {template_text}")
            
            # Show concrete examples
            examples = [entry['content'] for entry in self.parsed_logs 
                       if entry['template_id'] == template_id][:2]  # Just 2 examples to save space
            
            for j, example in enumerate(examples):
                print(f"    Example {j+1}: {example}")
            print()
            
    def get_frequency_research_data(self):
        """Return research-ready data structure."""
        return {
            'sessions': self.sessions,
            'session_labels': self.session_labels,
            'templates': self.templates,
            'template_frequencies': self.template_stats,
            'parsed_logs': self.parsed_logs,
            'sampling_info': {
                'sampling_ratio': self.sampling_ratio,
                'total_available_lines': self.total_lines_in_file
            }
        }

def run_efficient_hdfs_analysis(max_lines=1000000):
    """
    Run HDFS analysis with intelligent sampling for efficient research development.
    
    This approach allows you to develop and test your frequency-based algorithms
    quickly, then scale up to larger datasets once you're confident in your approach.
    """
    print("Starting Efficient HDFS Frequency Analysis")
    print(f"Using intelligent sampling approach: processing first {max_lines:,} lines")
    
    # Load and analyze the sampled data
    analyzer = LogDatasetAnalyser("HDFS")
    analyzer.load_hdfs_data("data/HDFS.log", 
                            label_file_path="data/anomaly_label.csv", 
                            max_lines=max_lines)
    
    # Get comprehensive statistics
    stats = analyzer.get_comprehensive_stats()
    
    # Analyze sampling effectiveness
    sampling_analysis = analyzer.get_sampling_analysis()
    
    # Display key insights
    print("\n" + "="*60)
    print("COMPREHENSIVE STATISTICS SUMMARY")
    print("="*60)
    
    print(f"\nSampling Analysis:")
    sampling_info = sampling_analysis['sampling_info']
    print(f"  Processed {sampling_info['lines_processed']:,} of {sampling_info['total_lines_available']:,} available lines")
    print(f"  Sampling ratio: {sampling_info['sampling_ratio']:.1%}")
    print(f"  Templates discovered: {stats['data_overview']['unique_templates']:,}")
    print(f"  Estimated total templates in full dataset: {sampling_info['estimated_total_templates']:.0f}")
    
    print(f"\nData Overview:")
    print(f"  Total log entries processed: {stats['data_overview']['total_log_entries']:,}")
    print(f"  Unique templates found: {stats['data_overview']['unique_templates']:,}")
    print(f"  Sessions created: {stats['data_overview']['total_sessions']:,}")
    print(f"  Compression ratio: {stats['data_overview']['compression_ratio']:.1f}x")
    
    print(f"\nTemplate Distribution Insights:")
    print(f"  Top 10% of templates cover {stats['template_distribution']['top_10_percent_coverage']:.1%} of all occurrences")
    print(f"  Most frequent template appears {stats['template_distribution']['most_frequent_template_count']:,} times")
    print(f"  Median template frequency: {stats['template_distribution']['median_template_frequency']:.0f}")
    
    # Show template examples
    analyzer.show_template_examples(top_n=15)
    
    # Prepare research data
    research_data = analyzer.get_frequency_research_data()
    
    print(f"\nReady for frequency-based anomaly detection research!")
    print(f"Your sample contains {len(research_data['sessions'])} sessions for algorithm development")
    print(f"Once you've developed your approach, you can scale up to the full dataset")
    
    return analyzer, stats, research_data

if __name__ == "__main__":
    # Process just the first million lines for efficient development
    analyzer, stats, research_data = run_efficient_hdfs_analysis(max_lines=1000000)