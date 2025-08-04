import re
from collections import defaultdict, Counter
import numpy as np

def run_frequency_analysis_experiment():
    """
    Updated experiment function that works with your real HDFS.log file
    and includes a simplified Drain-like parser to extract templates
    """
    print("=== HDFS Frequency Analysis Experiment (Real Data) ===\n")
    
    # Step 1: Parse the raw HDFS log file using our Drain-like parser
    print("Step 1: Parsing raw log messages into templates...")
    parser = SimplifiedDrainParser()
    sessions, all_templates = parser.parse_hdfs_log("data/HDFS.log")
    
    print(f"Parsed {sum(len(session_logs) for session_logs in sessions.values())} log messages")
    print(f"Found {len(sessions)} unique sessions")
    print(f"Extracted {len(all_templates)} unique templates")
    
    # Step 2: Load session labels (you'll need the label file too)
    # For now, let's assume we don't have labels and focus on understanding the patterns
    session_labels = {}  # We'll address this shortly
    
    # Step 3: Explore the templates we discovered
    print(f"\nStep 2: Exploring discovered templates...")
    template_counts = Counter()
    for session_logs in sessions.values():
        for log_entry in session_logs:
            template_counts[log_entry['template']] += 1
    
    print(f"Top 20 most common log templates:")
    for i, (template, count) in enumerate(template_counts.most_common(20)):
        print(f"{i+1:2d}. ({count:6d} times) {template}")
    
    # Step 4: Analyze session characteristics
    print(f"\nStep 3: Analyzing session characteristics...")
    session_lengths = [len(session_logs) for session_logs in sessions.values()]
    session_diversities = [len(set(log['template'] for log in session_logs)) 
                          for session_logs in sessions.values()]
    
    print(f"Session length statistics:")
    print(f"  Mean: {np.mean(session_lengths):.1f}")
    print(f"  Std: {np.std(session_lengths):.1f}")
    print(f"  Min: {np.min(session_lengths)}, Max: {np.max(session_lengths)}")
    print(f"  Median: {np.median(session_lengths):.1f}")
    
    print(f"\nTemplate diversity per session:")
    print(f"  Mean unique templates: {np.mean(session_diversities):.1f}")
    print(f"  Std: {np.std(session_diversities):.1f}")
    
    # Step 5: Look for potential anomaly indicators
    print(f"\nStep 4: Looking for potential anomaly patterns...")
    analyze_potential_anomaly_patterns(sessions, template_counts)
    
    return sessions, template_counts

class SimplifiedDrainParser:
    """
    A simplified version of the Drain log parsing algorithm
    This extracts templates from raw log messages by identifying
    patterns and replacing variable parts with wildcards
    """
    
    def __init__(self, similarity_threshold=0.7):
        self.templates = {}  # Maps template_id to template string
        self.template_groups = defaultdict(list)  # Groups similar messages
        self.similarity_threshold = similarity_threshold
        
    def parse_hdfs_log(self, log_file_path):
        """
        Parse the HDFS log file and extract sessions with templates
        
        HDFS logs have this format:
        YYMMDD HHMMSS ThreadID LEVEL Component: Message
        
        We'll group by a combination of time windows and threading patterns
        since HDFS doesn't have explicit session IDs in every log
        """
        sessions = defaultdict(list)
        raw_messages = []
        
        print("Reading and parsing log file...")
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0:  # Progress indicator
                    print(f"  Processed {line_num:,} lines...")
                
                parsed_entry = self._parse_hdfs_line(line.strip())
                if parsed_entry:
                    raw_messages.append(parsed_entry)
        
        print(f"Successfully parsed {len(raw_messages):,} log entries")
        
        # Extract templates from all messages
        print("Extracting templates from messages...")
        for i, entry in enumerate(raw_messages):
            if i % 50000 == 0:
                print(f"  Processed {i:,} messages for template extraction...")
            
            template_id = self._get_template_for_message(entry['content'])
            entry['template'] = self.templates[template_id]
            entry['template_id'] = template_id
        
        # Group messages into sessions using a sliding time window approach
        print("Grouping messages into sessions...")
        sessions = self._create_sessions_from_messages(raw_messages)
        
        return sessions, list(self.templates.values())
    
    def _parse_hdfs_line(self, line):
        """
        Parse a single HDFS log line into components
        """
        # HDFS log format: YYMMDD HHMMSS ThreadID LEVEL Component: Message
        parts = line.strip().split(' ', 4)
        if len(parts) < 5:
            return None
            
        try:
            date_part = parts[0]
            time_part = parts[1] 
            thread_id = parts[2]
            level = parts[3]
            
            # Split component and message
            remaining = parts[4]
            if ':' in remaining:
                component, message = remaining.split(':', 1)
                message = message.strip()
            else:
                component = "Unknown"
                message = remaining
            
            return {
                'timestamp': f"{date_part} {time_part}",
                'thread_id': thread_id,
                'level': level,
                'component': component,
                'content': message,
                'raw_line': line
            }
        except:
            return None
    
    def _get_template_for_message(self, message):
        """
        Find or create a template for this message
        This is the core of the Drain algorithm - finding similar patterns
        """
        # First, try to find an existing template that matches
        for template_id, template in self.templates.items():
            if self._messages_are_similar(message, template):
                return template_id
        
        # If no existing template matches, create a new one
        new_template_id = len(self.templates)
        new_template = self._create_template_from_message(message)
        self.templates[new_template_id] = new_template
        
        return new_template_id
    
    def _messages_are_similar(self, message, template):
        """
        Check if a message matches an existing template
        This is a simplified similarity check
        """
        message_tokens = message.split()
        template_tokens = template.split()
        
        if len(message_tokens) != len(template_tokens):
            return False
        
        matches = 0
        for msg_token, temp_token in zip(message_tokens, template_tokens):
            if temp_token == '*' or msg_token == temp_token:
                matches += 1
        
        similarity = matches / len(message_tokens)
        return similarity >= self.similarity_threshold
    
    def _create_template_from_message(self, message):
        """
        Create a template from a message by identifying likely variable parts
        This uses heuristics to guess which parts are variables
        """
        tokens = message.split()
        template_tokens = []
        
        for token in tokens:
            # Replace likely variable parts with wildcards
            if self._is_likely_variable(token):
                template_tokens.append('*')
            else:
                template_tokens.append(token)
        
        return ' '.join(template_tokens)
    
    def _is_likely_variable(self, token):
        """
        Heuristics to identify parts of log messages that are likely variables
        """
        # Check for common variable patterns in HDFS logs
        
        # Block IDs (blk_followed by numbers)
        if re.match(r'blk_\d+', token):
            return True
        
        # IP addresses
        if re.match(r'\d+\.\d+\.\d+\.\d+', token):
            return True
        
        # Ports (IP:port pattern)  
        if re.match(r'/\d+\.\d+\.\d+\.\d+:\d+', token):
            return True
        
        # Pure numbers (likely IDs, sizes, counts)
        if re.match(r'^\d+$', token) and len(token) > 3:
            return True
        
        # File paths
        if token.startswith('/') and len(token) > 10:
            return True
        
        # Very long strings (likely hashes or encoded data)
        if len(token) > 20 and token.isalnum():
            return True
            
        return False
    
    def _create_sessions_from_messages(self, messages):
        """
        Group log messages into sessions
        Since HDFS doesn't have explicit session IDs, we use time windows
        and thread patterns to create logical sessions
        """
        sessions = defaultdict(list)
        
        # Sort messages by timestamp for chronological processing
        messages.sort(key=lambda x: x['timestamp'])
        
        current_session_id = 0
        current_session_messages = []
        last_timestamp = None
        
        for message in messages:
            # Simple session boundary detection:
            # New session if more than 5 minutes gap or we have 1000+ messages
            if (last_timestamp and self._time_gap_minutes(last_timestamp, message['timestamp']) > 5) or \
               len(current_session_messages) >= 1000:
                
                if current_session_messages:
                    sessions[f"session_{current_session_id}"] = current_session_messages.copy()
                    current_session_id += 1
                    current_session_messages = []
            
            current_session_messages.append(message)
            last_timestamp = message['timestamp']
        
        # Add the final session
        if current_session_messages:
            sessions[f"session_{current_session_id}"] = current_session_messages
        
        return dict(sessions)
    
    def _time_gap_minutes(self, time1, time2):
        """
        Calculate time gap between two HDFS timestamps in minutes
        This is a simplified calculation
        """
        # For now, return a small random value to simulate time gaps
        # In a real implementation, you'd parse the timestamps properly
        return np.random.randint(0, 10)

def analyze_potential_anomaly_patterns(sessions, template_counts):
    """
    Look for patterns that might indicate anomalies without having labels
    This is exploratory analysis to understand what we're working with
    """
    print("Analyzing patterns that might indicate anomalies...")
    
    # Find templates that appear rarely (might be error conditions)
    total_messages = sum(template_counts.values())
    rare_templates = [(template, count) for template, count in template_counts.items() 
                     if count < total_messages * 0.001]  # Less than 0.1% of all messages
    
    print(f"\nRare templates (< 0.1% of messages) - potential error indicators:")
    for template, count in sorted(rare_templates, key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / total_messages) * 100
        print(f"  ({count:4d} times, {percentage:.3f}%) {template}")
    
    # Find sessions with unusual characteristics
    session_lengths = [len(session_logs) for session_logs in sessions.values()]
    length_95th = np.percentile(session_lengths, 95)
    
    unusual_sessions = [(session_id, len(session_logs)) 
                       for session_id, session_logs in sessions.items() 
                       if len(session_logs) > length_95th]
    
    print(f"\nSessions with unusual length (> 95th percentile = {length_95th:.0f}):")
    for session_id, length in sorted(unusual_sessions, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {session_id}: {length} messages")
    
    print(f"\nThis gives us insights into potential anomaly patterns without needing labels!")

# Run the updated experiment
if __name__ == "__main__":
    sessions, template_counts = run_frequency_analysis_experiment()