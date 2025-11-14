#!/usr/bin/env python3
"""
Main application for Smali Malware Analysis RAG System
Using LangChain v0.1.16 for retrieval-augmented generation
"""

import asyncio
import os
import sys
import argparse
import time
from typing import List, Dict
from tqdm import tqdm

from config import CONFIG, BEHAVIOR_DESCRIPTIONS
from smali_loader import SmaliFolderLoader
from retrieval_engine import MalwareRetrievalEngine
from report_generator import ReportGenerator
from logger import logger

class SmaliMalwareAnalyzer:
    """Main class for Smali malware analysis using RAG."""
    
    def __init__(self, smali_folder: str, output_dir: str = None):
        self.smali_folder = smali_folder
        self.output_dir = output_dir or CONFIG["output"]["output_dir"]
        self.loader = None
        self.retrieval_engine = None
        self.report_generator = ReportGenerator(self.output_dir)
        self.start_time = None
        self.document_count = 0  # Store document count to avoid reloading
        
        # Validate OpenAI API key
        if not CONFIG["openai"]["api_key"]:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
    
    async def setup_system(self, force_rebuild: bool = False):
        """Set up the RAG system (load documents and create vector store)."""
        logger.info("Setting up Smali Malware Analysis RAG System...")
        
        # Check if vector store already exists
        vectorstore_path = CONFIG["vectorstore"]["persist_directory"]
        if os.path.exists(vectorstore_path) and not force_rebuild:
            logger.info(f"Loading existing vector store from {vectorstore_path}")
            self.retrieval_engine = MalwareRetrievalEngine(vectorstore_path)
        else:
            logger.info("Creating new vector store...")
            await self._create_vectorstore()
    
    async def _create_vectorstore(self):
        """Create vector store from Smali files."""
        logger.info(f"Loading Smali files from: {self.smali_folder}")
        
        # Load Smali documents
        self.loader = SmaliFolderLoader(self.smali_folder)
        documents = await self.loader.load()
        
        if not documents:
            raise ValueError(f"No Smali files found in {self.smali_folder}")
        
        # Store document count to avoid reloading later
        self.document_count = len(documents)
        logger.info(f"Loaded {self.document_count} Smali classes")
        
        # Create retrieval engine and vector store
        self.retrieval_engine = MalwareRetrievalEngine()
        self.retrieval_engine.create_vectorstore(documents)
        
        logger.info("Vector store created successfully!")
    
    async def analyze_behaviors(self, behavior_ids: List[int], app_name: str = None) -> Dict:
        """Analyze specified behaviors for the Smali files."""
        if not self.retrieval_engine:
            raise ValueError("System not set up. Call setup_system() first.")
        
        if not app_name:
            app_name = os.path.basename(self.smali_folder)
        
        self.start_time = time.time()
        
        # Log analysis session start
        logger.log_analysis_start(app_name, behavior_ids, self.smali_folder)
        
        logger.info(f"Analyzing behaviors for: {app_name}")
        logger.info(f"Behaviors to analyze: {behavior_ids}")
        
        behavior_results = []
        total_relevant_classes = 0
        total_methods_analyzed = 0
        
        filtered_behavior_ids = []
        for behavior_id in tqdm(behavior_ids, desc="Analyzing behaviors"):
            if behavior_id not in BEHAVIOR_DESCRIPTIONS:
                logger.warning(f"Behavior ID {behavior_id} not found. Skipping.")
                continue
            filtered_behavior_ids.append(behavior_id)
            
        tasks = [
            self.retrieval_engine.retrieve_classes_for_behavior(behavior_id)
            for behavior_id in filtered_behavior_ids
        ]
        classes_results = await asyncio.gather(*tasks, return_exceptions=True)

        tasks = [
            self._analyze_methods(behavior_id, class_results)
            for behavior_id, class_results in zip(filtered_behavior_ids, classes_results)
            if not isinstance(class_results, Exception)
        ]
        behavior_analysis_results = await asyncio.gather(*tasks, return_exceptions=True)        
        for behavior_result, num_methods_involved in behavior_analysis_results:
            if isinstance(behavior_result, Exception):
                logger.error(f"Error analyzing behavior {behavior_id}: {behavior_result}")
                continue

            behavior_results.append(behavior_result)
            total_relevant_classes += len(behavior_result["class_results"])
            total_methods_analyzed += num_methods_involved
        
        # Calculate analysis duration
        duration = time.time() - self.start_time
        
        # Log analysis session end
        results_summary = {
            "total_classes": self.document_count,
            "relevant_classes": total_relevant_classes,
            "total_methods": total_methods_analyzed,
            "duration": f"{duration:.2f} seconds"
        }
        logger.log_analysis_end(results_summary)
        
        # Generate report
        report = self.report_generator.generate_behavior_report(app_name, behavior_results)
        
        return report
    
    async def _analyze_methods(self, behavior_id: int, class_results: List[Dict]) -> List[Dict]:
        # Retrieve relevant classes
        total_methods_analyzed = 0
        behavior_name = BEHAVIOR_DESCRIPTIONS[behavior_id]
        logger.info(f"Analyzing Behavior {behavior_id}: {behavior_name}")
        
        if not class_results:
            logger.info(f"No relevant classes found for behavior {behavior_id}")
            return {
                "behavior_id": behavior_id,
                "class_results": []
            }, 0
        
        logger.info(f"Found {len(class_results)} relevant classes")
        
        tasks = [
            self.retrieval_engine.analyze_methods_in_class(class_result, behavior_id)
            for class_result in class_results
        ]
        classes_involved_methods = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze methods in each class
        for class_result, involved_methods in zip(class_results, classes_involved_methods):
            if isinstance(involved_methods, Exception):
                logger.error(f"Error analyzing methods in class {class_result['class_name']}: {involved_methods}")
                continue
            logger.debug(f"Analyzing methods in {class_result['class_name']}")
            class_result['involved_methods'] = involved_methods
            total_methods_analyzed += len(involved_methods)
            logger.debug(f"Found {len(involved_methods)} involved methods")
        
        # Log behavior analysis results
        logger.log_behavior_analysis(behavior_id, behavior_name, {
            "class_results": class_results
        })
        
        return {
            "behavior_id": behavior_id,
            "class_results": class_results
        }, total_methods_analyzed
        
    
    def save_results(self, report: Dict, save_summary: bool = True):
        """Save analysis results to files."""
        # Save JSON report
        json_path = self.report_generator.save_report(report)
        logger.info(f"Report saved to: {json_path}")
        
        # Save summary report
        if save_summary:
            summary_path = self.report_generator.save_summary_report(report)
            logger.info(f"Summary report saved to: {summary_path}")
        
        # Print summary to console
        self.report_generator.print_analysis_summary(report)
        
        return json_path

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smali Malware Analysis RAG System")
    parser.add_argument("smali_folder", help="Path to folder containing Smali files")
    parser.add_argument("--behaviors", nargs="+", type=int, required=True,
                       help="List of behavior IDs to analyze (1-12)")
    parser.add_argument("--app-name", help="Name of the app being analyzed")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild of vector store")
    parser.add_argument("--no-summary", action="store_true",
                       help="Don't generate summary report")
    
    args = parser.parse_args()
    
    # Validate behavior IDs
    for behavior_id in args.behaviors:
        if behavior_id not in BEHAVIOR_DESCRIPTIONS:
            logger.error(f"Invalid behavior ID {behavior_id}. Valid IDs: {list(BEHAVIOR_DESCRIPTIONS.keys())}")
            sys.exit(1)
    
    try:
        # Initialize analyzer
        analyzer = SmaliMalwareAnalyzer(args.smali_folder, args.output_dir)
        
        # Setup system
        await analyzer.setup_system(force_rebuild=args.force_rebuild)
        
        # Analyze behaviors
        report = await analyzer.analyze_behaviors(args.behaviors, args.app_name)
        
        # Save results
        analyzer.save_results(report, save_summary=not args.no_summary)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
