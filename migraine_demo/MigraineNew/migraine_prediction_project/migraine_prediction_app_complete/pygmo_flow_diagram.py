import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects
import io
import os

def create_flow_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set background color
    ax.set_facecolor('#f9f9f9')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    
    # Define colors
    colors = {
        'data': '#3498db',
        'expert': '#2ecc71',
        'gating': '#9b59b6',
        'pygmo': '#e74c3c',
        'output': '#f39c12',
        'arrow': '#34495e',
        'box': '#ecf0f1',
        'text': '#2c3e50'
    }
    
    # Define shadow effect for boxes
    shadow_effect = [path_effects.SimpleLineShadow(offset=(2, -2), alpha=0.3),
                     path_effects.Normal()]
    
    # Helper function to create a box with text
    def create_box(x, y, width, height, text, color, fontsize=10, alpha=0.9):
        rect = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, 
                         edgecolor=colors['arrow'], linewidth=1)
        rect.set_path_effects(shadow_effect)
        ax.add_patch(rect)
        
        text_obj = ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                 fontsize=fontsize, color=colors['text'], fontweight='bold', wrap=True)
        
        return rect
    
    # Helper function to create an arrow
    def create_arrow(start, end, color=colors['arrow'], width=0.5, style='arc3,rad=0.1'):
        arrow = FancyArrowPatch(start, end, connectionstyle=f'{style}', 
                               arrowstyle='-|>', color=color, linewidth=width, 
                               mutation_scale=20)
        ax.add_patch(arrow)
        return arrow
    
    # Create data boxes
    data_box = create_box(5, 55, 20, 10, 'Input Data\n(Sleep, Weather,\nStress/Diet, Physio)', colors['data'])
    
    # Create expert model boxes
    sleep_expert = create_box(5, 40, 15, 8, 'Sleep Expert', colors['expert'])
    weather_expert = create_box(25, 40, 15, 8, 'Weather Expert', colors['expert'])
    stress_diet_expert = create_box(45, 40, 15, 8, 'Stress/Diet Expert', colors['expert'])
    physio_expert = create_box(65, 40, 15, 8, 'Physio Expert', colors['expert'])
    
    # Create gating network box
    gating_box = create_box(35, 25, 30, 8, 'Gating Network', colors['gating'])
    
    # Create PyGMO optimization boxes
    pygmo_box = create_box(75, 25, 20, 20, 'PyGMO Optimization', colors['pygmo'], fontsize=12)
    
    # Create optimization phases inside PyGMO box
    phase1_box = create_box(77, 40, 16, 3, 'Phase 1: Expert Optimization', colors['box'], fontsize=8)
    phase2_box = create_box(77, 35, 16, 3, 'Phase 2: Gating Optimization', colors['box'], fontsize=8)
    phase3_box = create_box(77, 30, 16, 3, 'Phase 3: End-to-End Optimization', colors['box'], fontsize=8)
    
    # Create output boxes
    fusion_box = create_box(35, 10, 30, 8, 'Fusion Mechanism', colors['output'])
    prediction_box = create_box(35, 2, 30, 5, 'Migraine Prediction', colors['output'])
    
    # Create arrows from data to experts
    create_arrow((15, 55), (12, 48))
    create_arrow((15, 55), (32, 48))
    create_arrow((15, 55), (52, 48))
    create_arrow((15, 55), (72, 48))
    
    # Create arrows from experts to gating
    create_arrow((12, 40), (40, 33))
    create_arrow((32, 40), (45, 33))
    create_arrow((52, 40), (50, 33))
    create_arrow((72, 40), (60, 33))
    
    # Create arrows from gating to fusion
    create_arrow((50, 25), (50, 18))
    
    # Create arrow from fusion to prediction
    create_arrow((50, 10), (50, 7))
    
    # Create arrows from PyGMO to experts and gating
    create_arrow((85, 40), (20, 44), style='arc3,rad=-0.2')
    create_arrow((85, 40), (40, 44), style='arc3,rad=-0.1')
    create_arrow((85, 40), (60, 44), style='arc3,rad=-0.1')
    create_arrow((85, 40), (80, 44), style='arc3,rad=-0.1')
    
    create_arrow((85, 35), (50, 29), style='arc3,rad=-0.1')
    
    # Create arrow for end-to-end optimization
    create_arrow((85, 30), (50, 15), style='arc3,rad=-0.1')
    
    # Add title
    ax.text(50, 67, 'PyGMO Optimization Flow in Migraine Prediction Model', 
            ha='center', fontsize=16, fontweight='bold', color=colors['text'])
    
    # Add explanatory text
    explanation_text = """
    1. Expert Optimization: PyGMO optimizes each expert model's hyperparameters independently
    2. Gating Optimization: PyGMO optimizes the weights assigned to each expert (Sleep: 35%, Weather: 15%, etc.)
    3. End-to-End Optimization: PyGMO performs multi-objective optimization for AUC and latency
    """
    
    ax.text(50, 62, explanation_text, ha='center', fontsize=10, color=colors['text'])
    
    # Save the diagram
    output_dir = '/home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/static'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as SVG
    svg_path = os.path.join(output_dir, 'pygmo_flow_diagram.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Save as PNG for backup
    png_path = os.path.join(output_dir, 'pygmo_flow_diagram.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.close()
    
    return svg_path, png_path

if __name__ == "__main__":
    svg_path, png_path = create_flow_diagram()
    print(f"Flow diagram created and saved to {svg_path} and {png_path}")
