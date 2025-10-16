import json
import os
from typing import List, Dict
from datetime import datetime
from config import CONFIG, BEHAVIOR_DESCRIPTIONS

class ReportGenerator:
    """Generator for malware analysis reports."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or CONFIG["output"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_behavior_report(self, app_name: str, behavior_results: List[Dict]) -> Dict:
        """Generate a comprehensive report for all analyzed behaviors."""
        report = {
            "app_name": app_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_behaviors_analyzed": len(behavior_results),
            "behaviors": []
        }
        
        for behavior_result in behavior_results:
            behavior_report = self._format_behavior_result(behavior_result)
            report["behaviors"].append(behavior_report)
        
        return report
    
    def _format_behavior_result(self, behavior_result: Dict) -> Dict:
        """Format a single behavior analysis result."""
        behavior_id = behavior_result["behavior_id"]
        
        formatted_result = {
            "behavior_id": behavior_id,
            "behavior_description": BEHAVIOR_DESCRIPTIONS[behavior_id],
            "relevant_classes": []
        }
        
        for class_result in behavior_result["class_results"]:
            formatted_class = {
                "class_signature": class_result["class_signature"],
                "class_explanation": class_result["explanation"],
                "vector_similarity_score": class_result.get("vector_similarity_score", class_result.get("similarity_score", 0.0)),
                "llm_relevance_score": class_result.get("llm_relevance_score", 0.0),
                "involved_methods": []
            }
            
            for method_result in class_result["involved_methods"]:
                formatted_method = {
                    "method_signature": method_result["method_signature"],
                    "role_explanation": method_result["role_explanation"],
                    "relevance_score": method_result["relevance_score"]
                }
                formatted_class["involved_methods"].append(formatted_method)
            
            formatted_result["relevant_classes"].append(formatted_class)
        
        return formatted_result
    
    def save_report(self, report: Dict, filename: str = None) -> str:
        """Save the report to a file."""
        if not filename:
            app_name = report["app_name"].replace(".apk", "").replace(".", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"malware_analysis_{app_name}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to: {filepath}")
        return filepath
    
    def generate_summary_report(self, report: Dict) -> str:
        """Generate a human-readable summary of the analysis."""
        summary_parts = []
        
        summary_parts.append(f"# Malware Analysis Report: {report['app_name']}")
        summary_parts.append(f"**Analysis Date:** {report['analysis_timestamp']}")
        summary_parts.append(f"**Behaviors Analyzed:** {report['total_behaviors_analyzed']}")
        summary_parts.append("")
        
        for behavior in report["behaviors"]:
            summary_parts.append(f"## Behavior {behavior['behavior_id']}: {behavior['behavior_description']}")
            
            if not behavior["relevant_classes"]:
                summary_parts.append("No relevant classes found.")
            else:
                summary_parts.append(f"**Relevant Classes:** {len(behavior['relevant_classes'])}")
                
                for i, class_result in enumerate(behavior["relevant_classes"], 1):
                    summary_parts.append(f"")
                    summary_parts.append(f"### {i}. {class_result['class_signature']}")
                    summary_parts.append(f"**Vector Similarity Score:** {class_result['vector_similarity_score']:.3f}")
                    summary_parts.append(f"**LLM Relevance Score:** {class_result['llm_relevance_score']:.3f}")
                    summary_parts.append(f"**Explanation:** {class_result['class_explanation']}")
                    
                    if class_result["involved_methods"]:
                        summary_parts.append(f"**Involved Methods:** {len(class_result['involved_methods'])}")
                        
                        for method in class_result["involved_methods"]:
                            summary_parts.append(f"- **{method['method_signature']}**")
                            summary_parts.append(f"  - Relevance: {method['relevance_score']:.3f}")
                            summary_parts.append(f"  - Role: {method['role_explanation']}")
            
            summary_parts.append("")
            summary_parts.append("---")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def save_summary_report(self, report: Dict, filename: str = None) -> str:
        """Save a human-readable summary report."""
        if not filename:
            app_name = report["app_name"].replace(".apk", "").replace(".", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"malware_analysis_summary_{app_name}_{timestamp}.md"
        
        filepath = os.path.join(self.output_dir, filename)
        summary_content = self.generate_summary_report(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"Summary report saved to: {filepath}")
        return filepath
    
    def print_analysis_summary(self, report: Dict):
        """Print a brief summary of the analysis to console."""
        print(f"\n{'='*60}")
        print(f"MALWARE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"App: {report['app_name']}")
        print(f"Analysis Date: {report['analysis_timestamp']}")
        print(f"Behaviors Analyzed: {report['total_behaviors_analyzed']}")
        print(f"{'='*60}")
        
        total_classes = 0
        total_methods = 0
        
        for behavior in report["behaviors"]:
            print(f"\nBehavior {behavior['behavior_id']}: {behavior['behavior_description']}")
            print(f"  Relevant Classes: {len(behavior['relevant_classes'])}")
            
            for class_result in behavior["relevant_classes"]:
                total_classes += 1
                print(f"    - {class_result['class_signature']} (Vector: {class_result['vector_similarity_score']:.3f}, LLM: {class_result['llm_relevance_score']:.3f})")
                print(f"      Methods: {len(class_result['involved_methods'])}")
                total_methods += len(class_result['involved_methods'])
        
        print(f"\n{'='*60}")
        print(f"TOTAL FINDINGS:")
        print(f"  Classes: {total_classes}")
        print(f"  Methods: {total_methods}")
        print(f"{'='*60}") 