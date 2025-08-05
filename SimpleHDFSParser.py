import pandas as pd
import os
import json
import re
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

class SimpleHDFSParser:
    """
    Simple HDFS log parser using Drain3 for template extraction.
    Focuses on parsing first 1M lines and saving results for inspection.
    """
    
    def __init__(self):
        self.parsed_logs = []
        self.templates = {}
        self.lines_processed = 0
        
    def parse_logs(self, log_file_path, max_lines=1000000):
        """Parse HDFS logs using Drain3 and extract templates."""
        print(f"Parsing first {max_lines:,} lines from {log_file_path}")
        
        # Configure Drain3
        config = TemplateMinerConfig()
        config.drain_sim_th = 0.7
        config.drain_depth = 4
        config.drain_max_children = 100
        
        template_miner = TemplateMiner(config=config)
        
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if line_num >= max_lines:
                    print(f"Reached limit of {max_lines:,} lines")
                    break
                    
                if line_num % int(max_lines/4) == 0:
                    print(f"  Processed {line_num:,} lines...")
                
                # Parse HDFS log line format: Date Time PID Level Component: Message
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
                    
                    # Extract session ID (block ID) from message
                    session_id = self._extract_session_id(message)
                    
                    # Get template from Drain3
                    result = template_miner.add_log_message(message)
                    template_id = result['cluster_id']
                    template = result['template_mined']
                    
                    # Store template
                    self.templates[template_id] = template
                    
                    # Store parsed log entry
                    self.parsed_logs.append({
                        'line_number': int(line_num),
                        'timestamp': f"{date} {time_part}",
                        'thread_id': thread_id,
                        'level': level,
                        'component': component,
                        'session_id': session_id,
                        'original_message': message,
                        'template_id': int(template_id),
                        'template': template
                    })
                    
                except Exception as e:
                    # Skip malformed lines
                    continue
                    
        self.lines_processed = len(self.parsed_logs)
        print(f"Successfully parsed {self.lines_processed:,} log entries")
        print(f"Found {len(self.templates)} unique templates")
        
        # Show session statistics
        session_ids = [log['session_id'] for log in self.parsed_logs if log['session_id']]
        unique_sessions = len(set(session_ids))
        print(f"Found {unique_sessions} unique sessions")
        
    def _extract_session_id(self, message):
        """Extract session ID (block ID) from HDFS log message."""
        # Look for block IDs in format: blk_123456789 or blk_-123456789
        block_pattern = r'blk_-?\d+'
        match = re.search(block_pattern, message)
        if match:
            return match.group()
        
        # Look for other session identifiers
        # Sometimes sessions are identified by file paths or other IDs
        # Add more patterns as needed based on your data
        return None
        
    def save_results(self, output_dir="results"):
        """Save parsed results to JSON files for easy inspection."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parsed logs as JSONL (one JSON object per line)
        logs_file = os.path.join(output_dir, "parsed_logs.jsonl")
        with open(logs_file, 'w') as f:
            for log_entry in self.parsed_logs:
                f.write(json.dumps(log_entry) + '\n')
        print(f"Saved parsed logs to {logs_file}")
        
        # Save templates with frequency counts as JSON
        df_logs = pd.DataFrame(self.parsed_logs)
        template_counts = df_logs['template_id'].value_counts()
        
        templates_data = {}
        for template_id, template in self.templates.items():
            frequency = int(template_counts.get(template_id, 0))  # Convert to int
            templates_data[str(template_id)] = {  # Convert key to string
                'template': template,
                'frequency': frequency
            }
        
        # Sort by frequency
        templates_sorted = dict(sorted(templates_data.items(), 
                                     key=lambda x: x[1]['frequency'], 
                                     reverse=True))
        
        templates_file = os.path.join(output_dir, "templates.json")
        with open(templates_file, 'w') as f:
            json.dump(templates_sorted, f, indent=2)
        print(f"Saved templates to {templates_file}")
        
        # Save sessions summary
        session_logs = {}
        for log_entry in self.parsed_logs:
            session_id = log_entry['session_id']
            if session_id:
                if session_id not in session_logs:
                    session_logs[session_id] = []
                session_logs[session_id].append(log_entry)
        
        sessions_file = os.path.join(output_dir, "sessions.json")
        with open(sessions_file, 'w') as f:
            json.dump(session_logs, f, indent=2)
        print(f"Saved sessions to {sessions_file}")
        
        # Save summary statistics
        unique_sessions = len(session_logs)
        session_sizes = [len(logs) for logs in session_logs.values()]
        
        summary = {
            'total_lines_processed': int(self.lines_processed),
            'unique_templates': int(len(self.templates)),
            'unique_sessions': int(unique_sessions),
            'logs_with_session_id': int(sum(1 for log in self.parsed_logs if log['session_id'])),
            'logs_without_session_id': int(sum(1 for log in self.parsed_logs if not log['session_id'])),
            'session_statistics': {
                'min_session_size': int(min(session_sizes)) if session_sizes else 0,
                'max_session_size': int(max(session_sizes)) if session_sizes else 0,
                'avg_session_size': float(round(sum(session_sizes) / len(session_sizes), 2)) if session_sizes else 0.0
            },
            'most_frequent_template': {
                'template_id': list(templates_sorted.keys())[0] if templates_sorted else None,
                'template': list(templates_sorted.values())[0]['template'] if templates_sorted else None,
                'frequency': int(list(templates_sorted.values())[0]['frequency']) if templates_sorted else 0
            },
            'compression_ratio': float(round(self.lines_processed / len(self.templates), 2)) if self.templates else 0.0
        }
        
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")
        
    def show_top_templates(self, n=10):
        """Display the most frequent templates."""
        if not self.parsed_logs:
            print("No data parsed yet. Run parse_logs() first.")
            return
            
        df_logs = pd.DataFrame(self.parsed_logs)
        template_counts = df_logs['template_id'].value_counts()
        
        print(f"\nTop {n} Most Frequent Templates:")
        print("=" * 60)
        
        for i, (template_id, count) in enumerate(template_counts.head(n).items()):
            template = self.templates[template_id]
            percentage = (count / len(self.parsed_logs)) * 100
            print(f"{i+1:2d}. ({count:,} times, {percentage:.1f}%) Template ID: {template_id}")
            print(f"    {template}")
            
            # Show one example
            example = df_logs[df_logs['template_id'] == template_id]['original_message'].iloc[0]
            print(f"    Example: {example}")
            print()

def main():
    """Main function to run the parsing."""
    parser = SimpleHDFSParser()
    
    # Parse the logs
    parser.parse_logs("data/HDFS.log", max_lines=10000)
    
    # Show top templates
    parser.show_top_templates(10)
    
    # Save results
    parser.save_results()
    
    print("\nParsing complete! Check the 'results' folder for:")
    print("  - parsed_logs.jsonl: All parsed log entries (one JSON object per line)")
    print("  - templates.json: Unique templates with frequencies") 
    print("  - sessions.json: Logs grouped by session ID")
    print("  - summary.json: Overview statistics")

if __name__ == "__main__":
    main()