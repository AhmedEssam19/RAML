#!/usr/bin/env python3
"""
Evaluation Script for Malware Detection System
Computes precision, recall, and F1 score at both class and method levels
by comparing system predictions against ground truth.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, List
from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for a given level (class or method)"""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    def compute_metrics(self):
        """Calculate precision, recall, and F1 score"""
        # Precision = TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0
        
        # Recall = TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0


@dataclass
class BehaviorMetrics:
    """Stores metrics for a specific behavior"""
    behavior_id: int
    behavior_name: str
    class_metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    method_metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)


def extract_ground_truth(ground_truth_data: Dict) -> Dict[int, Dict]:
    """
    Extract ground truth classes and methods for each behavior
    
    Returns:
        Dict mapping behavior_id to {
            'classes': Set[class_name],
            'methods': Set[(class_name, method_signature)]
        }
    """
    behavior_map = {}
    
    for behavior in ground_truth_data.get('groundtruth', []):
        behavior_id = behavior['behavior_id']
        behavior_name = behavior['behavior_name']
        class_name = behavior['class_name']
        
        if behavior_id not in behavior_map:
            behavior_map[behavior_id] = {
                'name': behavior_name,
                'classes': set(),
                'methods': set()
            }
        
        # Add class
        behavior_map[behavior_id]['classes'].add(class_name)
        
        # Add methods
        if 'methods' in behavior:
            for method in behavior['methods']:
                method_sig = method['signature']
                behavior_map[behavior_id]['methods'].add((class_name, method_sig))
        
        # Handle method_groups (for behaviors with multiple method groups)
        if 'method_groups' in behavior:
            for group in behavior['method_groups']:
                for method in group:
                    method_sig = method['signature']
                    behavior_map[behavior_id]['methods'].add((class_name, method_sig))
    
    return behavior_map


def extract_predictions(prediction_data: Dict) -> Dict[int, Dict]:
    """
    Extract predicted classes and methods for each behavior
    
    Returns:
        Dict mapping behavior_id to {
            'classes': Set[class_name],
            'methods': Set[(class_name, method_signature)]
        }
    """
    behavior_map = {}
    
    for behavior in prediction_data.get('behaviors', []):
        behavior_id = behavior['behavior_id']
        
        if behavior_id not in behavior_map:
            behavior_map[behavior_id] = {
                'classes': set(),
                'methods': set()
            }
        
        # Extract classes and methods from relevant_classes
        for class_info in behavior.get('relevant_classes', []):
            class_sig = class_info['class_signature']
            
            # Add class
            behavior_map[behavior_id]['classes'].add(class_sig)
            
            # Add methods
            for method in class_info.get('involved_methods', []):
                method_sig = method['method_signature']
                behavior_map[behavior_id]['methods'].add((class_sig, method_sig))
    
    return behavior_map


def evaluate_behavior(
    behavior_id: int,
    ground_truth: Dict,
    predictions: Dict
) -> BehaviorMetrics:
    """Evaluate predictions for a single behavior"""
    
    gt_classes = ground_truth.get('classes', set())
    gt_methods = ground_truth.get('methods', set())
    pred_classes = predictions.get('classes', set())
    pred_methods = predictions.get('methods', set())
    
    behavior_name = ground_truth.get('name', f'Behavior {behavior_id}')
    metrics = BehaviorMetrics(behavior_id=behavior_id, behavior_name=behavior_name)
    
    # Class-level evaluation
    class_tp = len(gt_classes & pred_classes)  # Intersection (true positives)
    class_fp = len(pred_classes - gt_classes)  # Predicted but not in ground truth
    class_fn = len(gt_classes - pred_classes)  # In ground truth but not predicted
    
    metrics.class_metrics.true_positives = class_tp
    metrics.class_metrics.false_positives = class_fp
    metrics.class_metrics.false_negatives = class_fn
    metrics.class_metrics.compute_metrics()
    
    # Method-level evaluation
    method_tp = len(gt_methods & pred_methods)
    method_fp = len(pred_methods - gt_methods)
    method_fn = len(gt_methods - pred_methods)
    
    metrics.method_metrics.true_positives = method_tp
    metrics.method_metrics.false_positives = method_fp
    metrics.method_metrics.false_negatives = method_fn
    metrics.method_metrics.compute_metrics()
    
    return metrics


def compute_overall_metrics(behavior_metrics_list: List[BehaviorMetrics]) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
    """Compute overall metrics across all behaviors"""
    
    overall_class = EvaluationMetrics()
    overall_method = EvaluationMetrics()
    
    # Aggregate counts
    for bm in behavior_metrics_list:
        overall_class.true_positives += bm.class_metrics.true_positives
        overall_class.false_positives += bm.class_metrics.false_positives
        overall_class.false_negatives += bm.class_metrics.false_negatives
        
        overall_method.true_positives += bm.method_metrics.true_positives
        overall_method.false_positives += bm.method_metrics.false_positives
        overall_method.false_negatives += bm.method_metrics.false_negatives
    
    # Compute overall metrics
    overall_class.compute_metrics()
    overall_method.compute_metrics()
    
    return overall_class, overall_method


def print_results(behavior_metrics_list: List[BehaviorMetrics], overall_class: EvaluationMetrics, overall_method: EvaluationMetrics):
    """Print evaluation results in a formatted table"""
    
    print("\n" + "="*100)
    print("MALWARE DETECTION EVALUATION RESULTS")
    print("="*100)
    
    # Per-behavior results
    for bm in behavior_metrics_list:
        print(f"\n{'─'*100}")
        print(f"BEHAVIOR {bm.behavior_id}: {bm.behavior_name}")
        print(f"{'─'*100}")
        
        # Class-level metrics
        print(f"\n  CLASS-LEVEL METRICS:")
        print(f"    True Positives:  {bm.class_metrics.true_positives}")
        print(f"    False Positives: {bm.class_metrics.false_positives}")
        print(f"    False Negatives: {bm.class_metrics.false_negatives}")
        print(f"    Precision:       {bm.class_metrics.precision:.4f}")
        print(f"    Recall:          {bm.class_metrics.recall:.4f}")
        print(f"    F1 Score:        {bm.class_metrics.f1_score:.4f}")
        
        # Method-level metrics
        print(f"\n  METHOD-LEVEL METRICS:")
        print(f"    True Positives:  {bm.method_metrics.true_positives}")
        print(f"    False Positives: {bm.method_metrics.false_positives}")
        print(f"    False Negatives: {bm.method_metrics.false_negatives}")
        print(f"    Precision:       {bm.method_metrics.precision:.4f}")
        print(f"    Recall:          {bm.method_metrics.recall:.4f}")
        print(f"    F1 Score:        {bm.method_metrics.f1_score:.4f}")
    
    # Overall results
    print(f"\n{'='*100}")
    print("OVERALL RESULTS (Across All Behaviors)")
    print(f"{'='*100}")
    
    print(f"\n  CLASS-LEVEL OVERALL:")
    print(f"    True Positives:  {overall_class.true_positives}")
    print(f"    False Positives: {overall_class.false_positives}")
    print(f"    False Negatives: {overall_class.false_negatives}")
    print(f"    Precision:       {overall_class.precision:.4f}")
    print(f"    Recall:          {overall_class.recall:.4f}")
    print(f"    F1 Score:        {overall_class.f1_score:.4f}")
    
    print(f"\n  METHOD-LEVEL OVERALL:")
    print(f"    True Positives:  {overall_method.true_positives}")
    print(f"    False Positives: {overall_method.false_positives}")
    print(f"    False Negatives: {overall_method.false_negatives}")
    print(f"    Precision:       {overall_method.precision:.4f}")
    print(f"    Recall:          {overall_method.recall:.4f}")
    print(f"    F1 Score:        {overall_method.f1_score:.4f}")
    
    print(f"\n{'='*100}\n")


def save_results_json(behavior_metrics_list: List[BehaviorMetrics], overall_class: EvaluationMetrics, overall_method: EvaluationMetrics, output_path: Path):
    """Save evaluation results to a JSON file"""
    
    results = {
        "per_behavior_results": [],
        "overall_results": {
            "class_level": {
                "true_positives": overall_class.true_positives,
                "false_positives": overall_class.false_positives,
                "false_negatives": overall_class.false_negatives,
                "precision": overall_class.precision,
                "recall": overall_class.recall,
                "f1_score": overall_class.f1_score
            },
            "method_level": {
                "true_positives": overall_method.true_positives,
                "false_positives": overall_method.false_positives,
                "false_negatives": overall_method.false_negatives,
                "precision": overall_method.precision,
                "recall": overall_method.recall,
                "f1_score": overall_method.f1_score
            }
        }
    }
    
    for bm in behavior_metrics_list:
        results["per_behavior_results"].append({
            "behavior_id": bm.behavior_id,
            "behavior_name": bm.behavior_name,
            "class_level": {
                "true_positives": bm.class_metrics.true_positives,
                "false_positives": bm.class_metrics.false_positives,
                "false_negatives": bm.class_metrics.false_negatives,
                "precision": bm.class_metrics.precision,
                "recall": bm.class_metrics.recall,
                "f1_score": bm.class_metrics.f1_score
            },
            "method_level": {
                "true_positives": bm.method_metrics.true_positives,
                "false_positives": bm.method_metrics.false_positives,
                "false_negatives": bm.method_metrics.false_negatives,
                "precision": bm.method_metrics.precision,
                "recall": bm.method_metrics.recall,
                "f1_score": bm.method_metrics.f1_score
            }
        })
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate malware detection predictions against ground truth"
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Path to save evaluation results JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth from: {args.ground_truth}")
    with open(args.ground_truth, 'r') as f:
        ground_truth_data = json.load(f)
    
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        prediction_data = json.load(f)
    
    # Extract data
    gt_behaviors = extract_ground_truth(ground_truth_data)
    pred_behaviors = extract_predictions(prediction_data)
    
    # Get all behavior IDs
    all_behavior_ids = set(gt_behaviors.keys()) | set(pred_behaviors.keys())
    
    # Evaluate each behavior
    behavior_metrics_list = []
    for behavior_id in sorted(all_behavior_ids):
        gt = gt_behaviors.get(behavior_id, {'classes': set(), 'methods': set(), 'name': f'Behavior {behavior_id}'})
        pred = pred_behaviors.get(behavior_id, {'classes': set(), 'methods': set()})
        
        metrics = evaluate_behavior(behavior_id, gt, pred)
        behavior_metrics_list.append(metrics)
    
    # Compute overall metrics
    overall_class, overall_method = compute_overall_metrics(behavior_metrics_list)
    
    # Print results
    print_results(behavior_metrics_list, overall_class, overall_method)
    
    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        save_results_json(behavior_metrics_list, overall_class, overall_method, output_path)


if __name__ == "__main__":
    main()
