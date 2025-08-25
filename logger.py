import logging
import os
from datetime import datetime
from typing import Optional

class MalwareAnalysisLogger:
    """Professional logging system for Smali malware analysis."""
    
    def __init__(self, name: str = "malware_analysis", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with file and console handlers."""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler (detailed logging)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"malware_analysis_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (info and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log startup
        self.logger.info(f"Malware Analysis Logger initialized. Log file: {log_file}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_analysis_start(self, app_name: str, behaviors: list, smali_folder: str):
        """Log analysis session start."""
        self.info("=" * 60)
        self.info("MALWARE ANALYSIS SESSION STARTED")
        self.info("=" * 60)
        self.info(f"App Name: {app_name}")
        self.info(f"Behaviors to Analyze: {behaviors}")
        self.info(f"Smali Folder: {smali_folder}")
        self.info(f"Analysis Timestamp: {datetime.now().isoformat()}")
        self.info("=" * 60)
    
    def log_analysis_end(self, results_summary: dict):
        """Log analysis session end."""
        self.info("=" * 60)
        self.info("MALWARE ANALYSIS SESSION COMPLETED")
        self.info("=" * 60)
        self.info(f"Total Classes Analyzed: {results_summary.get('total_classes', 0)}")
        self.info(f"Relevant Classes Found: {results_summary.get('relevant_classes', 0)}")
        self.info(f"Total Methods Analyzed: {results_summary.get('total_methods', 0)}")
        self.info(f"Analysis Duration: {results_summary.get('duration', 'N/A')}")
        self.info("=" * 60)
    
    def log_file_processing(self, file_path: str, status: str, details: Optional[str] = None):
        """Log file processing status."""
        if status == "success":
            self.debug(f"✓ Processed: {file_path}")
        elif status == "skipped":
            self.debug(f"⚠ Skipped: {file_path} - {details}")
        elif status == "error":
            self.error(f"✗ Error processing: {file_path} - {details}")
        elif status == "synthetic":
            self.debug(f"🔧 Synthetic class: {file_path}")
    
    def log_behavior_analysis(self, behavior_id: int, behavior_name: str, results: dict):
        """Log behavior analysis results."""
        relevant_classes = len(results.get('class_results', []))
        self.info(f"Behavior {behavior_id} ({behavior_name}): {relevant_classes} relevant classes found")
        
        if relevant_classes > 0:
            for class_result in results['class_results']:
                class_name = class_result.get('class_name', 'Unknown')
                similarity_score = class_result.get('similarity_score', 0)
                involved_methods = len(class_result.get('involved_methods', []))
                self.info(f"  └─ {class_name} (score: {similarity_score:.3f}, methods: {involved_methods})")

# Global logger instance
logger = MalwareAnalysisLogger() 