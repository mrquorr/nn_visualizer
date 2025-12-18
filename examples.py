#!/usr/bin/env python3
"""
Example usage of the Neural Network Visualizer
Shows how to visualize different network architectures
"""

import torch
import torch.nn as nn
from nn_visualizerv3 import NeuralNetworkVisualizer
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


# Example 1: Simple XOR Network
def example_xor_network():
    """Visualize a small network for XOR problem"""
    print("\n=== Example 1: XOR Network ===")
    
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Tanh(),
        nn.Linear(3, 1)
    )
    
    # XOR input
    input_data = torch.tensor([1.0, 0.0])
    target = torch.tensor([1.0])
    
    win = NeuralNetworkVisualizer(model, input_data, target)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


# Example 2: Digit Classification Network
def example_digit_network():
    """Visualize a simple digit classification network"""
    print("\n=== Example 2: Digit Classification Network ===")
    
    # Simple network for 8x8 pixel images
    model = nn.Sequential(
        nn.Linear(8, 6),  # 8 pixel input
        nn.ReLU(),
        nn.Linear(6, 4),  # Hidden layer
        nn.ReLU(),
        nn.Linear(4, 3)   # 3 class output
    )
    
    # Random input representing flattened pixels
    input_data = torch.randn(8)
    target = torch.tensor([1.0, 0.0, 0.0])  # One-hot encoded
    
    win = NeuralNetworkVisualizer(model, input_data, target)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


# Example 3: Regression Network
def example_regression_network():
    """Visualize a regression network"""
    print("\n=== Example 3: Regression Network ===")
    
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1)
    )
    
    input_data = torch.tensor([0.5, -0.3, 0.8, -0.1])
    target = torch.tensor([0.75])
    
    win = NeuralNetworkVisualizer(model, input_data, target)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


# Example 4: Custom Network Class
class CustomNetwork(nn.Module):
    """Example of using a custom nn.Module"""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def example_custom_network():
    """Visualize a custom network class"""
    print("\n=== Example 4: Custom Network Class ===")
    
    model = CustomNetwork()
    
    input_data = torch.tensor([1.0, 0.5, -0.2, 0.8, -0.5])
    target = torch.tensor([0.8, 0.3, 0.1])
    
    win = NeuralNetworkVisualizer(model, input_data, target)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


# Example 5: Pre-trained Network
def example_pretrained_network():
    """Visualize a network with pre-trained weights"""
    print("\n=== Example 5: Pre-trained Network ===")
    
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Set specific weights for demonstration
    with torch.no_grad():
        model[0].weight.data = torch.tensor([
            [0.5, -0.3, 0.2],
            [0.8, 0.1, -0.4],
            [-0.2, 0.6, 0.3],
            [0.4, -0.5, 0.7],
            [-0.6, 0.2, -0.1]
        ])
        model[0].bias.data = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2])
        
        model[2].weight.data = torch.tensor([
            [0.3, -0.4, 0.5, -0.2, 0.6],
            [-0.5, 0.3, -0.1, 0.4, -0.3]
        ])
        model[2].bias.data = torch.tensor([0.2, -0.1])
    
    input_data = torch.tensor([1.0, 0.5, -0.3])
    target = torch.tensor([0.8, 0.2])
    
    win = NeuralNetworkVisualizer(model, input_data, target)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


def main():
    """Main function - select which example to run"""
    examples = {
        '1': ('XOR Network (2→3→1)', example_xor_network),
        '2': ('Digit Classification (8→6→4→3)', example_digit_network),
        '3': ('Regression Network (4→8→1)', example_regression_network),
        '4': ('Custom Network Class (5→6→3)', example_custom_network),
        '5': ('Pre-trained Network (3→5→2)', example_pretrained_network),
    }
    
    print("\nNeural Network Visualizer - Examples")
    print("=" * 50)
    print("Select an example to visualize:")
    print()
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    print("\n  q. Quit")
    print("=" * 50)
    
    choice = input("\nEnter your choice: ").strip()
    
    if choice in examples:
        _, func = examples[choice]
        try:
            func()
        except ValueError as e:
            print(f"\nError: {e}")
            print("The network has too many neurons (max 40).")
            return 1
        except Exception as e:
            print(f"\nError: {e}")
            return 1
    elif choice.lower() == 'q':
        print("Goodbye!")
        return 0
    else:
        print("Invalid choice!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
