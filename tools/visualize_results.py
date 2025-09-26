# ------------------------------------------------------------------------------
# Visualization script for Litchi evaluation results
# ------------------------------------------------------------------------------

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.font_manager import FontProperties

# Set matplotlib to use a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(results_file):
    """Load evaluation results from JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def plot_pixel_accuracy(results, output_dir):
    """Plot Pixel Accuracy"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    pixel_acc = results['pixel_accuracy']
    
    bars = ax.bar(['Pixel Accuracy'], [pixel_acc], color='skyblue', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Pixel Accuracy')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pixel_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_pixel_accuracy(results, output_dir):
    """Plot Class Pixel Accuracy"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    class_names = results['class_names']
    class_acc = results['class_pixel_accuracy']
    
    bars = ax.bar(class_names, class_acc, color=['lightcoral', 'lightgreen', 'lightblue'], alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Class Pixel Accuracy')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_pixel_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_mpa(results, output_dir):
    """Plot Mean Pixel Accuracy (MPA)"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    mpa = results['mean_pixel_accuracy']
    
    bars = ax.bar(['Mean Pixel Accuracy (MPA)'], [mpa], color='orange', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Mean Pixel Accuracy (MPA)')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mpa.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_iou(results, output_dir):
    """Plot IoU for each class"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    class_names = results['class_names']
    iou = results['iou']
    
    bars = ax.bar(class_names, iou, color=['lightcoral', 'lightgreen', 'lightblue'], alpha=0.8)
    ax.set_ylabel('IoU')
    ax.set_title('Intersection over Union (IoU) per Class')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_miou(results, output_dir):
    """Plot Mean IoU (MIoU)"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    miou = results['miou']
    
    bars = ax.bar(['Mean IoU (MIoU)'], [miou], color='purple', alpha=0.8)
    ax.set_ylabel('IoU')
    ax.set_title('Mean Intersection over Union (MIoU)')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'miou.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_recall(results, output_dir):
    """Plot Recall for each class"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    class_names = results['class_names']
    recall = results['recall']
    
    bars = ax.bar(class_names, recall, color=['salmon', 'lightseagreen', 'plum'], alpha=0.8)
    ax.set_ylabel('Recall')
    ax.set_title('Recall per Class')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision(results, output_dir):
    """Plot Precision for each class"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    class_names = results['class_names']
    precision = results['precision']
    
    bars = ax.bar(class_names, precision, color=['gold', 'mediumaquamarine', 'mediumpurple'], alpha=0.8)
    ax.set_ylabel('Precision')
    ax.set_title('Precision per Class')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fscore(results, output_dir):
    """Plot F-score for each class"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    class_names = results['class_names']
    fscore = results['fscore']
    
    bars = ax.bar(class_names, fscore, color=['khaki', 'lightsteelblue', 'mistyrose'], alpha=0.8)
    ax.set_ylabel('F-score')
    ax.set_title('F-score per Class')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fscore.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameters(results, output_dir):
    """Plot Model Parameters"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    params = results['model_parameters_M']
    
    bars = ax.bar(['Model Parameters (M)'], [params], color='teal', alpha=0.8)
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Model Parameters')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameters.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(results, output_dir):
    """Plot Confusion Matrix"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    cm = np.array(results['confusion_matrix'])
    class_names = results['class_names']
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Normalized Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_metrics(results, output_dir):
    """Plot comprehensive metrics comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    class_names = results['class_names']
    
    # Plot 1: Precision, Recall, F-score comparison
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = results['precision']
    recall = results['recall']
    fscore = results['fscore']
    
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, fscore, width, label='F-score', alpha=0.8)
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, F-score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Plot 2: IoU and Class Pixel Accuracy comparison
    iou = results['iou']
    class_acc = results['class_pixel_accuracy']
    
    ax2.bar(x - width/2, iou, width, label='IoU', alpha=0.8)
    ax2.bar(x + width/2, class_acc, width, label='Class Pixel Accuracy', alpha=0.8)
    
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Score')
    ax2.set_title('IoU vs Class Pixel Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Plot 3: Overall metrics
    overall_metrics = ['Pixel Accuracy', 'MPA', 'MIoU']
    overall_values = [results['pixel_accuracy'], results['mean_pixel_accuracy'], results['miou']]
    
    bars3 = ax3.bar(overall_metrics, overall_values, color=['skyblue', 'orange', 'purple'], alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Overall Performance Metrics')
    ax3.set_ylim(0, 1)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Plot 4: Model info
    model_info = ['Parameters (M)', 'FPS']
    model_values = [results['model_parameters_M'], results['fps']]
    
    # Normalize FPS to 0-1 scale for visualization (assuming max FPS of 100)
    normalized_fps = min(results['fps'] / 100.0, 1.0)
    model_values_normalized = [results['model_parameters_M'] / 50.0, normalized_fps]  # Assuming max 50M params
    
    bars4 = ax4.bar(model_info, model_values_normalized, color=['teal', 'coral'], alpha=0.8)
    ax4.set_ylabel('Normalized Score')
    ax4.set_title('Model Efficiency')
    ax4.set_ylim(0, 1)
    
    # Add actual values as text
    ax4.text(0, model_values_normalized[0] + 0.05, f'{model_values[0]:.2f}M', ha='center')
    ax4.text(1, model_values_normalized[1] + 0.05, f'{model_values[1]:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(results_file, output_dir):
    """Generate all visualization plots"""
    # Load results
    results = load_results(results_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualization plots...")
    
    # Generate individual plots
    plot_pixel_accuracy(results, output_dir)
    print("✓ Pixel Accuracy plot generated")
    
    plot_class_pixel_accuracy(results, output_dir)
    print("✓ Class Pixel Accuracy plot generated")
    
    plot_mpa(results, output_dir)
    print("✓ MPA plot generated")
    
    plot_iou(results, output_dir)
    print("✓ IoU plot generated")
    
    plot_miou(results, output_dir)
    print("✓ MIoU plot generated")
    
    plot_recall(results, output_dir)
    print("✓ Recall plot generated")
    
    plot_precision(results, output_dir)
    print("✓ Precision plot generated")
    
    plot_fscore(results, output_dir)
    print("✓ F-score plot generated")
    
    plot_parameters(results, output_dir)
    print("✓ Parameters plot generated")
    
    plot_confusion_matrix(results, output_dir)
    print("✓ Confusion Matrix plot generated")
    
    plot_comprehensive_metrics(results, output_dir)
    print("✓ Comprehensive metrics plot generated")
    
    print(f"\nAll plots saved to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--results-file',
                        help='evaluation results JSON file',
                        default='../output/litchi/evaluation/evaluation_results.json',
                        type=str)
    parser.add_argument('--output-dir',
                        help='output directory for plots',
                        default='../output/litchi/evaluation/plots',
                        type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        print("Please run the evaluation script first.")
        return
    
    generate_all_plots(args.results_file, args.output_dir)

if __name__ == '__main__':
    main()