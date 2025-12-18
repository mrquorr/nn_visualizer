#!/usr/bin/env python3
"""
Neural Network Visualization Tool
Visualizes forward and backward propagation through PyTorch networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, Pango
import cairo
import math
from typing import List, Dict, Tuple, Optional
import numpy as np


class NetworkAnalyzer:
    """Analyzes a PyTorch network and captures forward/backward propagation"""
    
    def __init__(self, model: nn.Module, input_data: torch.Tensor, target: torch.Tensor):
        self.model = model
        self.input_data = input_data
        self.target = target
        self.layers_info = []
        self.total_neurons = 0
        
        # Storage for intermediate values
        self.activations = {}
        self.gradients = {}
        self.weights = {}
        self.biases = {}
        self.weight_grads = {}
        self.bias_grads = {}
        
        # Analyze network structure
        self._analyze_network()
        
    def _analyze_network(self):
        """Analyze network structure and validate neuron count"""
        self.layers_info = []
        self.total_neurons = 0
        
        # Get input size
        if len(self.input_data.shape) == 1:
            input_size = self.input_data.shape[0]
        else:
            input_size = self.input_data.shape[1]
        
        self.total_neurons += input_size
        self.layers_info.append({
            'type': 'input',
            'size': input_size,
            'name': 'Input'
        })
        
        # Analyze each layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                out_features = module.out_features
                self.total_neurons += out_features
                self.layers_info.append({
                    'type': 'linear',
                    'size': out_features,
                    'name': name if name else 'Linear',
                    'module': module
                })
        
        # Validate neuron count
        if self.total_neurons > 40:
            raise ValueError(f"Network has {self.total_neurons} neurons, maximum is 40")
    
    def run_forward_backward(self):
        """Run forward and backward pass, capturing all intermediate values"""
        # Register hooks
        hooks = []
        
        def forward_hook(module, input, output):
            name = self._get_module_name(module)
            self.activations[name] = output.detach().clone()
        
        def backward_hook(module, grad_input, grad_output):
            name = self._get_module_name(module)
            self.gradients[name] = grad_output[0].detach().clone()
        
        # Register hooks for all Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_full_backward_hook(backward_hook))
        
        # Store initial weights and biases
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.weights[name] = module.weight.detach().clone()
                if module.bias is not None:
                    self.biases[name] = module.bias.detach().clone()
                    print(self.biases[name])
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(self.input_data)
        
        # Calculate loss
        criterion = nn.MSELoss()
        loss = criterion(output, self.target)
        
        # Backward pass
        loss.backward()
        
        # Store gradients
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    self.weight_grads[name] = module.weight.grad.detach().clone()
                if module.bias is not None and module.bias.grad is not None:
                    self.bias_grads[name] = module.bias.grad.detach().clone()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Store input
        self.activations['input'] = self.input_data.detach().clone()
        
        return loss.item()
    
    def _get_module_name(self, module):
        """Get the name of a module"""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name if name else 'Linear'
        return 'Unknown'
    
    def create_visualization_states(self) -> List[Dict]:
        """Create a list of visualization states for each step"""
        states = []
        
        # State 0: Input layer
        states.append({
            'step': 0,
            'type': 'input',
            'title': 'Input Layer',
            'description': 'Initial input values before processing',
            'values': self.activations['input'],
            'layer_idx': 0
        })
        
        # Forward propagation states
        layer_idx = 1
        for layer_info in self.layers_info[1:]:
            if layer_info['type'] == 'linear':
                name = layer_info['name']
                module = layer_info['module']
                
                # Pre-activation state
                states.append({
                    'step': len(states),
                    'type': 'forward_pre',
                    'title': f'Layer {layer_idx}: Weight Application',
                    'description': f'Computing z = Wx + b for {name}. Weight values shown on connections.',
                    'values': self.activations[name],
                    'weights': self.weights[name],
                    'biases': self.biases.get(name),
                    'prev_values': states[-1]['values'],
                    'layer_idx': layer_idx
                })
                
                # Post-activation state
                states.append({
                    'step': len(states),
                    'type': 'forward_post',
                    'title': f'Layer {layer_idx}: Activation Output',
                    'description': f'Output activations from {name}. Weight values shown on connections.',
                    'values': self.activations[name],
                    'layer_idx': layer_idx
                })
                
                layer_idx += 1
        
        # Backward propagation states
        layer_idx = len(self.layers_info) - 1
        for layer_info in reversed(self.layers_info[1:]):
            if layer_info['type'] == 'linear':
                name = layer_info['name']
                
                # Gradient at output
                if name in self.gradients:
                    states.append({
                        'step': len(states),
                        'type': 'backward_output',
                        'title': f'Layer {layer_idx}: Output Gradients',
                        'description': f'Gradients flowing back to {name}',
                        'gradients': self.gradients[name],
                        'layer_idx': layer_idx
                    })
                
                # Weight gradients
                if name in self.weight_grads:
                    states.append({
                        'step': len(states),
                        'type': 'backward_weights',
                        'title': f'Layer {layer_idx}: Weight Gradients',
                        'description': f'Gradients for weights in {name}',
                        'weight_grads': self.weight_grads[name],
                        'bias_grads': self.bias_grads.get(name),
                        'weights': self.weights[name],
                        'biases': self.biases.get(name),
                        'layer_idx': layer_idx
                    })
                
                layer_idx -= 1
        
        return states


class NeuralNetworkVisualizer(Gtk.Window):
    """GTK Window for visualizing neural network propagation"""
    
    def __init__(self, model: nn.Module, input_data: torch.Tensor, target: torch.Tensor):
        super().__init__(title="Neural Network Visualizer")
        
        self.set_default_size(1400, 900)
        self.set_position(Gtk.WindowPosition.CENTER)
        
        # Analyze network
        self.analyzer = NetworkAnalyzer(model, input_data, target)
        loss = self.analyzer.run_forward_backward()
        self.states = self.analyzer.create_visualization_states()
        self.current_state = 0
        
        # Setup UI
        self._setup_ui(loss)
        
        # Initialize formula display
        self._update_formula_display(self.states[0])
        
        # Connect keyboard events
        self.connect("key-press-event", self._on_key_press)
        
    def _setup_ui(self, loss: float):
        """Setup the user interface"""
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
        
        # Info bar at top
        info_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        info_box.set_margin_start(10)
        info_box.set_margin_end(10)
        info_box.set_margin_top(10)
        info_box.set_margin_bottom(10)
        
        self.title_label = Gtk.Label()
        self.title_label.set_markup(f"<b><span size='14000'>{self.states[0]['title']}</span></b>")
        info_box.pack_start(self.title_label, True, True, 0)
        
        self.step_label = Gtk.Label()
        self.step_label.set_text(f"Step: 0 / {len(self.states)-1}")
        info_box.pack_end(self.step_label, False, False, 0)
        
        loss_label = Gtk.Label()
        loss_label.set_text(f"Loss: {loss:.6f}")
        info_box.pack_end(loss_label, False, False, 0)
        
        vbox.pack_start(info_box, False, False, 0)
        
        # Description label
        self.desc_label = Gtk.Label()
        self.desc_label.set_text(self.states[0]['description'])
        self.desc_label.set_margin_start(10)
        self.desc_label.set_margin_end(10)
        vbox.pack_start(self.desc_label, False, False, 0)
        
        # Legend
        legend_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=15)
        legend_box.set_margin_start(10)
        legend_box.set_margin_end(10)
        legend_box.set_margin_bottom(5)
        legend_box.set_halign(Gtk.Align.CENTER)
        
        legend_label = Gtk.Label()
        legend_label.set_markup("<b>Legend:</b>")
        legend_box.pack_start(legend_label, False, False, 0)
        
        # Color indicators
        colors = [
            ("◉", "0.6 0.8 1.0", "Outbound (sending)"),
            ("◉", "0.6 1.0 0.6", "Inbound (receiving)"),
            ("◉", "0.85 0.85 0.85", "Inactive"),
            ("◉", "1.0 0.6 0.6", "Gradient flow")
        ]
        
        for symbol, color, desc in colors:
            item_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
            color_label = Gtk.Label()
            # Parse RGB string
            r, g, b = map(float, color.split())
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
            color_label.set_markup(f"<span foreground='{color_hex}' size='14000'><b>{symbol}</b></span>")
            item_box.pack_start(color_label, False, False, 0)
            
            desc_label = Gtk.Label()
            desc_label.set_text(desc)
            item_box.pack_start(desc_label, False, False, 0)
            
            legend_box.pack_start(item_box, False, False, 0)
        
        # Add note about weight values
        weight_note = Gtk.Label()
        weight_note.set_markup("<i>• Weight values shown on active connections</i>")
        legend_box.pack_end(weight_note, False, False, 10)
        
        vbox.pack_start(legend_box, False, False, 0)
        
        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.pack_start(separator, False, False, 0)
        
        # Main content area with formula panel and drawing
        content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        # Drawing area
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect("draw", self._on_draw)
        content_box.pack_start(self.drawing_area, True, True, 0)
        
        # Formula panel on the right
        formula_frame = Gtk.Frame(label="Mathematical Operations")
        formula_frame.set_margin_end(10)
        formula_frame.set_size_request(400, -1)
        
        formula_scroll = Gtk.ScrolledWindow()
        formula_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        
        self.formula_textview = Gtk.TextView()
        self.formula_textview.set_editable(False)
        self.formula_textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.formula_textview.set_margin_start(10)
        self.formula_textview.set_margin_end(10)
        self.formula_textview.set_margin_top(10)
        self.formula_textview.set_margin_bottom(10)
        
        # Set monospace font for better formula display
        self.formula_textview.override_font(Pango.FontDescription("monospace 10"))
        
        self.formula_buffer = self.formula_textview.get_buffer()
        formula_scroll.add(self.formula_textview)
        formula_frame.add(formula_scroll)
        
        content_box.pack_start(formula_frame, False, False, 0)
        
        vbox.pack_start(content_box, True, True, 0)
        
        # Navigation buttons
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        button_box.set_halign(Gtk.Align.CENTER)
        button_box.set_margin_bottom(10)
        
        self.prev_button = Gtk.Button(label="◀ Previous (Left Arrow)")
        self.prev_button.connect("clicked", self._on_prev_clicked)
        self.prev_button.set_sensitive(False)
        button_box.pack_start(self.prev_button, False, False, 0)
        
        self.next_button = Gtk.Button(label="Next (Right Arrow) ▶")
        self.next_button.connect("clicked", self._on_next_clicked)
        button_box.pack_start(self.next_button, False, False, 0)
        
        vbox.pack_start(button_box, False, False, 0)
        
    def _on_key_press(self, widget, event):
        """Handle keyboard events"""
        if event.keyval == Gdk.KEY_Left:
            self._prev_state()
        elif event.keyval == Gdk.KEY_Right:
            self._next_state()
        return True
    
    def _on_prev_clicked(self, button):
        """Handle previous button click"""
        self._prev_state()
    
    def _on_next_clicked(self, button):
        """Handle next button click"""
        self._next_state()
    
    def _prev_state(self):
        """Go to previous state"""
        if self.current_state > 0:
            self.current_state -= 1
            self._update_ui()
    
    def _next_state(self):
        """Go to next state"""
        if self.current_state < len(self.states) - 1:
            self.current_state += 1
            self._update_ui()
    
    def _update_ui(self):
        """Update UI elements"""
        state = self.states[self.current_state]
        self.title_label.set_markup(f"<b><span size='14000'>{state['title']}</span></b>")
        self.desc_label.set_text(state['description'])
        self.step_label.set_text(f"Step: {self.current_state} / {len(self.states)-1}")
        
        self.prev_button.set_sensitive(self.current_state > 0)
        self.next_button.set_sensitive(self.current_state < len(self.states) - 1)
        
        # Update formula display
        self._update_formula_display(state)
        
        self.drawing_area.queue_draw()
    
    def _update_formula_display(self, state):
        """Update the formula text display based on current state"""
        formula_text = self._generate_formula(state)
        self.formula_buffer.set_text(formula_text)
    
    def _generate_formula(self, state):
        """Generate formula text for current state"""
        state_type = state['type']
        
        if state_type == 'input':
            return self._formula_input(state)
        elif state_type == 'forward_pre':
            return self._formula_forward_pre(state)
        elif state_type == 'forward_post':
            return self._formula_forward_post(state)
        elif state_type == 'backward_output':
            return self._formula_backward_output(state)
        elif state_type == 'backward_weights':
            return self._formula_backward_weights(state)
        else:
            return "No formula for this state"
    
    def _formula_input(self, state):
        """Formula for input layer"""
        values = state['values']
        if isinstance(values, torch.Tensor):
            values = values.numpy()
        if len(values.shape) > 1:
            values = values.flatten()
        
        formula = "=" * 60 + "\n"
        formula += "INPUT LAYER\n"
        formula += "=" * 60 + "\n\n"
        formula += "Initial input vector:\n\n"
        formula += "SYMBOLIC:\n"
        formula += "  x = [x₀, x₁, x₂, ...]\n\n"
        formula += "NUMERICAL:\n"
        formula += "  x = ["
        formula += ", ".join([f"{v:.3f}" for v in values])
        formula += "]\n\n"
        formula += "This is the raw input data that will be\n"
        formula += "propagated forward through the network.\n"
        
        return formula
    
    def _formula_forward_pre(self, state):
        """Formula for forward propagation weight application"""
        layer_idx = state['layer_idx']
        values = state['values']
        prev_values = state.get('prev_values')
        weights = state.get('weights')
        biases = state.get('biases')
        
        # Convert to numpy
        if isinstance(values, torch.Tensor):
            values = values.numpy()
        if len(values.shape) > 1:
            values = values.flatten()
        
        if prev_values is not None:
            if isinstance(prev_values, torch.Tensor):
                prev_values = prev_values.numpy()
            if len(prev_values.shape) > 1:
                prev_values = prev_values.flatten()
        
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.numpy()
        
        if biases is not None:
            if isinstance(biases, torch.Tensor):
                biases = biases.numpy()
        
        formula = "=" * 60 + "\n"
        formula += f"LAYER {layer_idx}: WEIGHT APPLICATION\n"
        formula += "=" * 60 + "\n\n"
        formula += "Computing linear transformation:\n\n"
        
        # Symbolic formula
        formula += "SYMBOLIC FORMULA:\n"
        if biases is not None:
            formula += "  z = W × x + b\n\n"
        else:
            formula += "  z = W × x\n\n"
        
        formula += "Where:\n"
        formula += f"  x = input vector (size {len(prev_values)})\n"
        formula += f"  W = weight matrix ({len(values)} × {len(prev_values)})\n"
        if biases is not None:
            formula += f"  b = bias vector (size {len(values)})\n"
        formula += f"  z = output vector (size {len(values)})\n\n"
        
        # Numerical example for first output neuron
        formula += "NUMERICAL CALCULATION:\n\n"
        formula += "For first output neuron (z₀):\n\n"
        
        # Show calculation for first neuron
        formula += "  z₀ = "
        terms = []
        for i, (w, x) in enumerate(zip(weights[0], prev_values)):
            terms.append(f"({w:.3f} × {x:.3f})")
        formula += " + ".join(terms)
        
        if biases is not None:
            formula += f" + {biases[0]:.3f}"
        
        # Calculate the sum
        sum_val = sum(w * x for w, x in zip(weights[0], prev_values))
        if biases is not None:
            sum_val += biases[0]
        
        formula += f"\n     = {sum_val:.6f}\n\n"
        
        # Show all outputs
        formula += "All outputs:\n"
        for i in range(min(len(values), 5)):  # Show first 5
            formula += f"  z{i} = {values[i]:.6f}\n"
        if len(values) > 5:
            formula += f"  ... ({len(values) - 5} more)\n"
        
        return formula
    
    def _formula_forward_post(self, state):
        """Formula for forward propagation after activation"""
        layer_idx = state['layer_idx']
        values = state['values']
        
        if isinstance(values, torch.Tensor):
            values = values.numpy()
        if len(values.shape) > 1:
            values = values.flatten()
        
        formula = "=" * 60 + "\n"
        formula += f"LAYER {layer_idx}: ACTIVATION FUNCTION\n"
        formula += "=" * 60 + "\n\n"
        formula += "Applying non-linear activation:\n\n"
        
        # Symbolic formula
        formula += "SYMBOLIC FORMULA:\n"
        formula += "  a = f(z)\n\n"
        formula += "Where:\n"
        formula += "  z = pre-activation values\n"
        formula += "  f = activation function (ReLU, Sigmoid, etc.)\n"
        formula += "  a = activated outputs\n\n"
        
        # Numerical
        formula += "NUMERICAL VALUES:\n\n"
        formula += "After activation:\n"
        for i in range(min(len(values), 8)):
            formula += f"  a{i} = {values[i]:.6f}\n"
        if len(values) > 8:
            formula += f"  ... ({len(values) - 8} more)\n"
        
        formula += "\nThese activated values become the input\n"
        formula += "to the next layer.\n"
        
        return formula
    
    def _formula_backward_output(self, state):
        """Formula for backward propagation output gradients"""
        layer_idx = state['layer_idx']
        gradients = state['gradients']
        
        if isinstance(gradients, torch.Tensor):
            gradients = gradients.numpy()
        if len(gradients.shape) > 1:
            gradients = gradients.flatten()
        
        formula = "=" * 60 + "\n"
        formula += f"LAYER {layer_idx}: GRADIENT COMPUTATION\n"
        formula += "=" * 60 + "\n\n"
        formula += "Computing gradients via chain rule:\n\n"
        
        # Symbolic formula
        formula += "SYMBOLIC FORMULA:\n"
        formula += "  ∂L/∂a = ∂L/∂z × ∂z/∂a\n\n"
        formula += "Where:\n"
        formula += "  L = loss function\n"
        formula += "  a = activations from this layer\n"
        formula += "  z = weighted sum in next layer\n"
        formula += "  ∂L/∂a = gradient w.r.t. activations\n\n"
        
        # Chain rule expansion
        formula += "Chain rule expansion:\n"
        formula += "  ∂L/∂a[i] = Σⱼ (∂L/∂z[j] × W[j,i])\n\n"
        
        # Numerical
        formula += "NUMERICAL VALUES:\n\n"
        formula += "Gradients flowing to this layer:\n"
        for i in range(min(len(gradients), 8)):
            formula += f"  ∂L/∂a{i} = {gradients[i]:.6f}\n"
        if len(gradients) > 8:
            formula += f"  ... ({len(gradients) - 8} more)\n"
        
        formula += "\nThese gradients will be used to compute\n"
        formula += "weight updates for this layer.\n"
        
        return formula
    
    def _formula_backward_weights(self, state):
        """Formula for weight gradient computation"""
        layer_idx = state['layer_idx']
        weight_grads = state['weight_grads']
        bias_grads = state.get('bias_grads')
        weights = state['weights']
        
        if isinstance(weight_grads, torch.Tensor):
            weight_grads = weight_grads.numpy()
        if isinstance(weights, torch.Tensor):
            weights = weights.numpy()
        if bias_grads is not None and isinstance(bias_grads, torch.Tensor):
            bias_grads = bias_grads.numpy()
        
        formula = "=" * 60 + "\n"
        formula += f"LAYER {layer_idx}: WEIGHT GRADIENTS\n"
        formula += "=" * 60 + "\n\n"
        formula += "Computing gradients for weight updates:\n\n"
        
        # Symbolic formula
        formula += "SYMBOLIC FORMULA:\n"
        formula += "  ∂L/∂W = ∂L/∂z × aᵀ\n"
        if bias_grads is not None:
            formula += "  ∂L/∂b = ∂L/∂z\n"
        formula += "\nWhere:\n"
        formula += "  ∂L/∂W = gradient w.r.t. weights\n"
        formula += "  ∂L/∂z = gradient from next layer\n"
        formula += "  a = activations from previous layer\n"
        if bias_grads is not None:
            formula += "  ∂L/∂b = gradient w.r.t. biases\n"
        formula += "\n"
        
        # Weight update rule
        formula += "WEIGHT UPDATE RULE:\n"
        formula += "  W_new = W_old - η × ∂L/∂W\n"
        if bias_grads is not None:
            formula += "  b_new = b_old - η × ∂L/∂b\n"
        formula += "\nWhere η is the learning rate\n\n"
        
        # Numerical examples
        formula += "NUMERICAL EXAMPLES:\n\n"
        
        # Show a few weight gradients
        rows, cols = weight_grads.shape
        formula += f"Weight gradient matrix ({rows}×{cols}):\n"
        for i in range(min(3, rows)):
            formula += "  ["
            for j in range(min(4, cols)):
                formula += f"{weight_grads[i, j]:7.4f}"
            if cols > 4:
                formula += " ..."
            formula += "]\n"
        if rows > 3:
            formula += f"  ... ({rows - 3} more rows)\n"
        
        formula += "\n"
        
        # Show bias gradients
        if bias_grads is not None:
            formula += "Bias gradients:\n"
            for i in range(min(8, len(bias_grads))):
                formula += f"  ∂L/∂b{i} = {bias_grads[i]:.6f}\n"
            if len(bias_grads) > 8:
                formula += f"  ... ({len(bias_grads) - 8} more)\n"
            formula += "\n"
        
        # Example update for one weight
        formula += "Example update for W[0,0]:\n"
        formula += f"  W[0,0] = {weights[0, 0]:.6f}\n"
        formula += f"  ∂L/∂W[0,0] = {weight_grads[0, 0]:.6f}\n"
        formula += f"  W_new[0,0] = {weights[0, 0]:.6f} - η × {weight_grads[0, 0]:.6f}\n"
        formula += "  (assuming learning rate η)\n"
        
        return formula
    
    def _on_draw(self, widget, cr):
        """Draw the visualization"""
        state = self.states[self.current_state]
        
        # Get drawing area size
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        
        # White background
        cr.set_source_rgb(1, 1, 1)
        cr.paint()
        
        # Draw based on state type
        if state['type'] in ['input', 'forward_pre', 'forward_post']:
            self._draw_full_network_forward(cr, width, height, state)
        elif state['type'] == 'backward_output':
            self._draw_full_network_backward(cr, width, height, state)
        elif state['type'] == 'backward_weights':
            self._draw_backward_weights(cr, width, height, state)
    
    def _draw_full_network_forward(self, cr, width, height, state):
        """Draw the entire network with current propagation step highlighted"""
        # Get all layer information
        all_layers = []
        
        # Add input layer
        input_values = self.analyzer.activations['input'].numpy()
        if len(input_values.shape) > 1:
            input_values = input_values.flatten()
        all_layers.append({
            'name': 'Input',
            'values': input_values,
            'layer_idx': 0,
            'type': 'input'
        })
        
        # Add all Linear layers
        for idx, layer_info in enumerate(self.analyzer.layers_info[1:], 1):
            if layer_info['type'] == 'linear':
                name = layer_info['name']
                values = self.analyzer.activations[name].numpy()
                if len(values.shape) > 1:
                    values = values.flatten()
                all_layers.append({
                    'name': name,
                    'values': values,
                    'layer_idx': idx,
                    'type': 'linear',
                    'module': layer_info['module']
                })
        
        # Determine active layers for current state
        current_layer = state.get('layer_idx', 0)
        
        # Calculate layout
        n_layers = len(all_layers)
        layer_spacing = (width - 100) / (n_layers - 1) if n_layers > 1 else width / 2
        
        # Find max neurons in any layer for vertical spacing
        max_neurons = max(len(layer['values']) for layer in all_layers)
        neuron_radius = min(25, (height - 100) / (max_neurons * 2.5))
        
        # Store layer positions for drawing connections
        layer_positions = []
        
        # Draw connections first (behind neurons)
        for i in range(len(all_layers) - 1):
            layer = all_layers[i]
            next_layer = all_layers[i + 1]
            
            if next_layer['type'] != 'linear':
                continue
            
            # Get weights
            weights = self.analyzer.weights[next_layer['name']].numpy()
            
            # Determine if this connection is active
            is_active = False
            if state['type'] == 'forward_pre' and current_layer == i + 1:
                is_active = True
            elif state['type'] == 'forward_post' and current_layer == i + 1:
                is_active = True
            elif state['type'] == 'input' and i == 0:
                is_active = False
            
            # Calculate positions
            layer_x = 50 + i * layer_spacing
            next_layer_x = 50 + (i + 1) * layer_spacing
            
            layer_spacing_y = min(80, (height - 100) / len(layer['values']))
            next_layer_spacing_y = min(80, (height - 100) / len(next_layer['values']))
            
            layer_start_y = (height - (len(layer['values']) - 1) * layer_spacing_y) / 2
            next_layer_start_y = (height - (len(next_layer['values']) - 1) * next_layer_spacing_y) / 2
            
            # Draw connections
            for j in range(len(next_layer['values'])):
                for k in range(len(layer['values'])):
                    weight = weights[j, k]
                    
                    layer_y = layer_start_y + k * layer_spacing_y
                    next_layer_y = next_layer_start_y + j * next_layer_spacing_y
                    
                    if is_active:
                        # Active connection - colored by weight
                        if weight > 0:
                            cr.set_source_rgba(0, 0, 1, min(abs(weight) * 0.8, 0.8))
                        else:
                            cr.set_source_rgba(1, 0, 0, min(abs(weight) * 0.8, 0.8))
                        cr.set_line_width(max(0.5, min(abs(weight) * 2, 3)))
                    else:
                        # Inactive connection - light gray
                        cr.set_source_rgba(0.7, 0.7, 0.7, 0.3)
                        cr.set_line_width(0.5)
                    
                    cr.move_to(layer_x + neuron_radius, layer_y)
                    cr.line_to(next_layer_x - neuron_radius, next_layer_y)
                    cr.stroke()
                    
                    # Draw weight value on active connections
                    if is_active:
                        # Only show weight values if not too many connections
                        total_connections = len(next_layer['values']) * len(layer['values'])
                        show_weight_labels = total_connections <= 20
                        
                        if show_weight_labels:
                            # Calculate midpoint of connection
                            mid_x = (layer_x + neuron_radius + next_layer_x - neuron_radius) / 2
                            mid_y = (layer_y + next_layer_y) / 2
                            
                            # Draw weight value
                            cr.set_source_rgb(0, 0, 0)
                            cr.set_font_size(8)
                            weight_text = f"{weight:.2f}"
                            extents = cr.text_extents(weight_text)
                            
                            # Offset slightly to avoid line overlap
                            offset_y = 10 if j % 2 == 0 else -5
                            
                            # Draw background rectangle for readability
                            padding = 2
                            cr.set_source_rgba(1, 1, 1, 0.9)
                            cr.rectangle(mid_x - extents.width / 2 - padding,
                                       mid_y + offset_y - extents.height - padding,
                                       extents.width + 2 * padding,
                                       extents.height + 2 * padding)
                            cr.fill()
                            
                            # Draw text
                            cr.set_source_rgb(0, 0, 0)
                            cr.move_to(mid_x - extents.width / 2, mid_y + offset_y)
                            cr.show_text(weight_text)
        
        # Draw neurons
        for i, layer in enumerate(all_layers):
            layer_x = 50 + i * layer_spacing
            
            n_neurons = len(layer['values'])
            spacing_y = min(80, (height - 100) / n_neurons)
            start_y = (height - (n_neurons - 1) * spacing_y) / 2
            
            # Determine layer color
            if state['type'] == 'input':
                if i == 0:
                    color_mode = 'outbound'  # Light blue
                else:
                    color_mode = 'inactive'  # Gray
            elif state['type'] == 'forward_pre':
                if i == current_layer - 1:
                    color_mode = 'outbound'  # Light blue (sending)
                elif i == current_layer:
                    color_mode = 'inbound'  # Light green (receiving)
                else:
                    color_mode = 'inactive'  # Gray
            elif state['type'] == 'forward_post':
                if i == current_layer:
                    color_mode = 'inbound'  # Light green (just received)
                elif i < current_layer:
                    color_mode = 'processed'  # Darker (already processed)
                else:
                    color_mode = 'inactive'  # Gray (not yet processed)
            else:
                color_mode = 'inactive'
            
            # Draw neurons in this layer
            for j, value in enumerate(layer['values']):
                y = start_y + j * spacing_y
                
                label = f"x{j}" if i == 0 else f"a{i}.{j}"
                self._draw_neuron_colored(cr, layer_x, y, neuron_radius, value, label, color_mode)
            
            # Draw layer label
            cr.set_source_rgb(0, 0, 0)
            cr.set_font_size(12)
            layer_label = f"Layer {i}" if i > 0 else "Input"
            extents = cr.text_extents(layer_label)
            cr.move_to(layer_x - extents.width / 2, height - 20)
            cr.show_text(layer_label)
            
            # Draw bias values for linear layers
            if layer['type'] == 'linear' and i > 0:
                biases = self.analyzer.biases.get(layer['name'])
                if biases is not None:
                    biases = biases.numpy()
                    # Draw bias annotation
                    cr.set_font_size(9)
                    cr.set_source_rgb(0.4, 0.4, 0.4)
                    for j, bias in enumerate(biases):
                        y = start_y + j * spacing_y
                        cr.move_to(layer_x + neuron_radius + 5, y + 15)
                        if color_mode != 'inactive':
                            cr.show_text(f"b={bias:.2f}")
    
    def _draw_full_network_backward(self, cr, width, height, state):
        """Draw the entire network with backward propagation highlighted"""
        # Get all layer information
        all_layers = []
        
        # Add input layer
        input_values = self.analyzer.activations['input'].numpy()
        if len(input_values.shape) > 1:
            input_values = input_values.flatten()
        all_layers.append({
            'name': 'Input',
            'values': input_values,
            'layer_idx': 0,
            'type': 'input'
        })
        
        # Add all Linear layers
        for idx, layer_info in enumerate(self.analyzer.layers_info[1:], 1):
            if layer_info['type'] == 'linear':
                name = layer_info['name']
                values = self.analyzer.activations[name].numpy()
                if len(values.shape) > 1:
                    values = values.flatten()
                all_layers.append({
                    'name': name,
                    'values': values,
                    'layer_idx': idx,
                    'type': 'linear',
                    'module': layer_info['module']
                })
        
        # Determine active layers for current state
        current_layer = state.get('layer_idx', len(all_layers) - 1)
        
        # Get gradients for current layer
        layer_name = all_layers[current_layer]['name'] if current_layer > 0 else None
        gradients = None
        if layer_name and layer_name in self.analyzer.gradients:
            gradients = self.analyzer.gradients[layer_name].numpy()
            if len(gradients.shape) > 1:
                gradients = gradients.flatten()
        
        # Calculate layout
        n_layers = len(all_layers)
        layer_spacing = (width - 100) / (n_layers - 1) if n_layers > 1 else width / 2
        
        # Find max neurons in any layer for vertical spacing
        max_neurons = max(len(layer['values']) for layer in all_layers)
        neuron_radius = min(25, (height - 100) / (max_neurons * 2.5))
        
        # Draw connections (light gray)
        for i in range(len(all_layers) - 1):
            layer = all_layers[i]
            next_layer = all_layers[i + 1]
            
            if next_layer['type'] != 'linear':
                continue
            
            weights = self.analyzer.weights[next_layer['name']].numpy()
            
            layer_x = 50 + i * layer_spacing
            next_layer_x = 50 + (i + 1) * layer_spacing
            
            layer_spacing_y = min(80, (height - 100) / len(layer['values']))
            next_layer_spacing_y = min(80, (height - 100) / len(next_layer['values']))
            
            layer_start_y = (height - (len(layer['values']) - 1) * layer_spacing_y) / 2
            next_layer_start_y = (height - (len(next_layer['values']) - 1) * next_layer_spacing_y) / 2
            
            # Highlight connections involved in gradient flow
            is_gradient_path = (i + 1 == current_layer or i == current_layer)
            
            for j in range(len(next_layer['values'])):
                for k in range(len(layer['values'])):
                    weight = weights[j, k]
                    
                    layer_y = layer_start_y + k * layer_spacing_y
                    next_layer_y = next_layer_start_y + j * next_layer_spacing_y
                    
                    if is_gradient_path:
                        # Show gradient flow
                        cr.set_source_rgba(1, 0.3, 0.3, 0.6)
                        cr.set_line_width(1.5)
                    else:
                        cr.set_source_rgba(0.7, 0.7, 0.7, 0.2)
                        cr.set_line_width(0.5)
                    
                    cr.move_to(layer_x + neuron_radius, layer_y)
                    cr.line_to(next_layer_x - neuron_radius, next_layer_y)
                    cr.stroke()
        
        # Draw neurons
        for i, layer in enumerate(all_layers):
            layer_x = 50 + i * layer_spacing
            
            n_neurons = len(layer['values'])
            spacing_y = min(80, (height - 100) / n_neurons)
            start_y = (height - (n_neurons - 1) * spacing_y) / 2
            
            # Determine if this layer is involved in current backward step
            if i == current_layer:
                color_mode = 'gradient_active'
            elif i > current_layer:
                color_mode = 'gradient_done'
            else:
                color_mode = 'inactive'
            
            # Draw neurons in this layer
            for j, value in enumerate(layer['values']):
                y = start_y + j * spacing_y
                
                # Show gradient value if available
                if i == current_layer and gradients is not None and j < len(gradients):
                    grad_val = gradients[j]
                    label = f"∂L/∂a={grad_val:.3f}"
                    self._draw_neuron_with_gradient(cr, layer_x, y, neuron_radius, value, label, grad_val)
                else:
                    label = f"x{j}" if i == 0 else f"a{i}.{j}"
                    self._draw_neuron_colored(cr, layer_x, y, neuron_radius, value, label, color_mode)
            
            # Draw layer label
            cr.set_source_rgb(0, 0, 0)
            cr.set_font_size(12)
            layer_label = f"Layer {i}" if i > 0 else "Input"
            extents = cr.text_extents(layer_label)
            cr.move_to(layer_x - extents.width / 2, height - 20)
            cr.show_text(layer_label)
    
    def _draw_input_layer(self, cr, width, height, state):
        """Draw input layer visualization (DEPRECATED - using full network view)"""
        values = state['values'].numpy() if isinstance(state['values'], torch.Tensor) else state['values']
        if len(values.shape) > 1:
            values = values.flatten()
        
        n_neurons = len(values)
        neuron_radius = 25
        
        # Calculate positions
        spacing = min(80, (height - 100) / n_neurons)
        start_y = (height - (n_neurons - 1) * spacing) / 2
        x = width / 2
        
        # Draw neurons
        for i, val in enumerate(values):
            y = start_y + i * spacing
            self._draw_neuron(cr, x, y, neuron_radius, val, f"x{i}", is_input=True)
    
    def _draw_forward_propagation(self, cr, width, height, state):
        """Draw forward propagation visualization (DEPRECATED - using full network view)"""
        values = state['values'].numpy() if isinstance(state['values'], torch.Tensor) else state['values']
        if len(values.shape) > 1:
            values = values.flatten()
        
        prev_values = None
        weights = None
        biases = None
        
        if 'prev_values' in state:
            prev_values = state['prev_values'].numpy() if isinstance(state['prev_values'], torch.Tensor) else state['prev_values']
            if len(prev_values.shape) > 1:
                prev_values = prev_values.flatten()
        
        if 'weights' in state:
            weights = state['weights'].numpy() if isinstance(state['weights'], torch.Tensor) else state['weights']
        
        if 'biases' in state:
            biases = state['biases']
            if biases is not None:
                biases = biases.numpy() if isinstance(biases, torch.Tensor) else biases
        
        # Draw two layers
        self._draw_two_layers(cr, width, height, prev_values, values, weights, biases)
    
    def _draw_backward_output(self, cr, width, height, state):
        """Draw backward propagation output gradients (DEPRECATED - using full network view)"""
        gradients = state['gradients'].numpy() if isinstance(state['gradients'], torch.Tensor) else state['gradients']
        if len(gradients.shape) > 1:
            gradients = gradients.flatten()
        
        n_neurons = len(gradients)
        neuron_radius = 25
        
        # Calculate positions
        spacing = min(80, (height - 100) / n_neurons)
        start_y = (height - (n_neurons - 1) * spacing) / 2
        x = width / 2
        
        # Draw neurons with gradients
        for i, grad in enumerate(gradients):
            y = start_y + i * spacing
            self._draw_neuron(cr, x, y, neuron_radius, grad, f"∂L/∂a{i}", is_gradient=True)
    
    def _draw_backward_weights(self, cr, width, height, state):
        """Draw weight gradients"""
        weight_grads = state['weight_grads'].numpy() if isinstance(state['weight_grads'], torch.Tensor) else state['weight_grads']
        weights = state['weights'].numpy() if isinstance(state['weights'], torch.Tensor) else state['weights']
        
        bias_grads = None
        if 'bias_grads' in state and state['bias_grads'] is not None:
            bias_grads = state['bias_grads'].numpy() if isinstance(state['bias_grads'], torch.Tensor) else state['bias_grads']
        
        # Draw weight matrix with gradients
        self._draw_weight_gradients(cr, width, height, weights, weight_grads, bias_grads)
    
    def _draw_two_layers(self, cr, width, height, prev_values, values, weights, biases):
        """Draw two connected layers with weights"""
        neuron_radius = 20
        
        n_prev = len(prev_values) if prev_values is not None else 0
        n_curr = len(values)
        
        # Calculate positions
        prev_spacing = min(60, (height - 100) / max(n_prev, 1))
        curr_spacing = min(60, (height - 100) / n_curr)
        
        prev_start_y = (height - (n_prev - 1) * prev_spacing) / 2 if n_prev > 0 else height / 2
        curr_start_y = (height - (n_curr - 1) * curr_spacing) / 2
        
        prev_x = width * 0.3
        curr_x = width * 0.7
        
        # Draw connections first (if weights available)
        if weights is not None and prev_values is not None:
            for i in range(n_curr):
                for j in range(n_prev):
                    weight = weights[i, j]
                    
                    # Color based on weight value
                    if weight > 0:
                        cr.set_source_rgba(0, 0, 1, min(abs(weight), 1))
                    else:
                        cr.set_source_rgba(1, 0, 0, min(abs(weight), 1))
                    
                    cr.set_line_width(max(0.5, min(abs(weight) * 2, 3)))
                    
                    prev_y = prev_start_y + j * prev_spacing
                    curr_y = curr_start_y + i * curr_spacing
                    
                    cr.move_to(prev_x + neuron_radius, prev_y)
                    cr.line_to(curr_x - neuron_radius, curr_y)
                    cr.stroke()
        
        # Draw previous layer neurons
        if prev_values is not None:
            for i, val in enumerate(prev_values):
                y = prev_start_y + i * prev_spacing
                self._draw_neuron(cr, prev_x, y, neuron_radius, val, f"a{i}")
        
        # Draw current layer neurons
        for i, val in enumerate(values):
            y = curr_start_y + i * curr_spacing
            label = f"z{i}" if biases is not None else f"a{i}"
            self._draw_neuron(cr, curr_x, y, neuron_radius, val, label)
            
            # Draw bias
            if biases is not None:
                bias = biases[i]
                cr.set_source_rgb(0, 0, 0)
                cr.set_font_size(10)
                cr.move_to(curr_x + neuron_radius + 5, y + 15)
                cr.show_text(f"b={bias:.3f}")
    
    def _draw_weight_gradients(self, cr, width, height, weights, weight_grads, bias_grads):
        """Draw weight matrix with gradient overlays"""
        rows, cols = weights.shape
        
        # Calculate cell size
        max_cell_size = min(50, (width - 200) / cols, (height - 200) / rows)
        cell_size = max_cell_size
        
        # Calculate starting position (centered)
        total_width = cols * cell_size
        total_height = rows * cell_size
        start_x = (width - total_width) / 2
        start_y = (height - total_height) / 2
        
        # Find max absolute values for normalization
        max_weight = np.abs(weights).max()
        max_grad = np.abs(weight_grads).max()
        
        # Draw cells
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * cell_size
                y = start_y + i * cell_size
                
                weight = weights[i, j]
                grad = weight_grads[i, j]
                
                # Draw weight cell
                normalized_weight = weight / max_weight if max_weight > 0 else 0
                if weight > 0:
                    cr.set_source_rgb(0.7 + 0.3 * normalized_weight, 0.7, 0.7)
                else:
                    cr.set_source_rgb(0.7, 0.7, 0.7 - 0.3 * abs(normalized_weight))
                
                cr.rectangle(x, y, cell_size, cell_size)
                cr.fill()
                
                # Draw border
                cr.set_source_rgb(0.3, 0.3, 0.3)
                cr.set_line_width(1)
                cr.rectangle(x, y, cell_size, cell_size)
                cr.stroke()
                
                # Draw weight value
                cr.set_source_rgb(0, 0, 0)
                cr.set_font_size(9)
                text = f"w={weight:.2f}"
                extents = cr.text_extents(text)
                cr.move_to(x + (cell_size - extents.width) / 2, y + cell_size / 2 - 5)
                cr.show_text(text)
                
                # Draw gradient value
                cr.set_source_rgb(1, 0, 0)
                text = f"∂={grad:.3f}"
                extents = cr.text_extents(text)
                cr.move_to(x + (cell_size - extents.width) / 2, y + cell_size / 2 + 10)
                cr.show_text(text)
        
        # Draw bias gradients
        if bias_grads is not None:
            bias_x = start_x + total_width + 20
            for i, grad in enumerate(bias_grads):
                y = start_y + i * cell_size
                
                cr.set_source_rgb(0.9, 0.9, 0.7)
                cr.rectangle(bias_x, y, cell_size, cell_size)
                cr.fill()
                
                cr.set_source_rgb(0.3, 0.3, 0.3)
                cr.set_line_width(1)
                cr.rectangle(bias_x, y, cell_size, cell_size)
                cr.stroke()
                
                cr.set_source_rgb(1, 0, 0)
                cr.set_font_size(9)
                text = f"∂b={grad:.3f}"
                extents = cr.text_extents(text)
                cr.move_to(bias_x + (cell_size - extents.width) / 2, y + cell_size / 2)
                cr.show_text(text)
    
    def _draw_neuron_colored(self, cr, x, y, radius, value, label, color_mode):
        """Draw a neuron with specific color mode
        
        color_mode can be:
        - 'outbound': Light blue (sending signal)
        - 'inbound': Light green (receiving signal)
        - 'inactive': Gray (not involved in current step)
        - 'processed': Darker shade (already processed)
        - 'gradient_active': Red (gradient flowing through)
        - 'gradient_done': Pink (gradient already passed)
        """
        # Set fill color based on mode
        if color_mode == 'outbound':
            cr.set_source_rgb(0.6, 0.8, 1.0)  # Light blue
        elif color_mode == 'inbound':
            cr.set_source_rgb(0.6, 1.0, 0.6)  # Light green
        elif color_mode == 'inactive':
            cr.set_source_rgb(0.85, 0.85, 0.85)  # Gray
        elif color_mode == 'processed':
            cr.set_source_rgb(0.7, 0.85, 0.95)  # Darker blue-ish
        elif color_mode == 'gradient_active':
            cr.set_source_rgb(1.0, 0.6, 0.6)  # Light red
        elif color_mode == 'gradient_done':
            cr.set_source_rgb(1.0, 0.8, 0.8)  # Light pink
        else:
            cr.set_source_rgb(0.9, 0.9, 0.9)
        
        # Draw circle
        cr.arc(x, y, radius, 0, 2 * math.pi)
        cr.fill()
        
        # Draw border with emphasis for active states
        if color_mode in ['outbound', 'inbound', 'gradient_active']:
            cr.set_source_rgb(0, 0, 0)
            cr.set_line_width(3)
        else:
            cr.set_source_rgb(0.4, 0.4, 0.4)
            cr.set_line_width(1.5)
        cr.arc(x, y, radius, 0, 2 * math.pi)
        cr.stroke()
        
        # Draw label
        cr.set_source_rgb(0, 0, 0)
        cr.set_font_size(9)
        extents = cr.text_extents(label)
        cr.move_to(x - extents.width / 2, y - radius - 5)
        cr.show_text(label)
        
        # Draw value (only show if not inactive)
        if color_mode != 'inactive':
            cr.set_font_size(10)
            value_text = f"{value:.3f}"
            extents = cr.text_extents(value_text)
            cr.move_to(x - extents.width / 2, y + 4)
            cr.show_text(value_text)
    
    def _draw_neuron_with_gradient(self, cr, x, y, radius, value, label, gradient):
        """Draw a neuron with gradient information"""
        # Color based on gradient magnitude
        intensity = min(abs(gradient), 1)
        cr.set_source_rgb(1, 1 - intensity * 0.4, 1 - intensity * 0.4)
        
        # Draw circle
        cr.arc(x, y, radius, 0, 2 * math.pi)
        cr.fill()
        
        # Draw border
        cr.set_source_rgb(0.8, 0, 0)
        cr.set_line_width(3)
        cr.arc(x, y, radius, 0, 2 * math.pi)
        cr.stroke()
        
        # Draw label above
        cr.set_source_rgb(0, 0, 0)
        cr.set_font_size(8)
        extents = cr.text_extents(label)
        cr.move_to(x - extents.width / 2, y - radius - 5)
        cr.show_text(label)
    
    def _draw_neuron(self, cr, x, y, radius, value, label, is_input=False, is_gradient=False):
        """Draw a single neuron (legacy method for backward compatibility)"""
        # Color based on value
        if is_gradient:
            # Red for gradients
            intensity = min(abs(value), 1)
            cr.set_source_rgb(1, 1 - intensity, 1 - intensity)
        elif is_input:
            # Blue for inputs
            cr.set_source_rgb(0.8, 0.9, 1)
        else:
            # Green gradient for activations
            normalized = 1 / (1 + math.exp(-value))  # Sigmoid normalization
            cr.set_source_rgb(1 - normalized, 1, 1 - normalized)
        
        # Draw circle
        cr.arc(x, y, radius, 0, 2 * math.pi)
        cr.fill()
        
        # Draw border
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(2)
        cr.arc(x, y, radius, 0, 2 * math.pi)
        cr.stroke()
        
        # Draw label
        cr.set_font_size(10)
        extents = cr.text_extents(label)
        cr.move_to(x - extents.width / 2, y - radius - 5)
        cr.show_text(label)
        
        # Draw value
        cr.set_font_size(11)
        value_text = f"{value:.3f}"
        extents = cr.text_extents(value_text)
        cr.move_to(x - extents.width / 2, y + 5)
        cr.show_text(value_text)


def create_sample_network():
    """Create a sample neural network for demonstration"""
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    )
    return model


def main():
    """Main function"""
    print("Neural Network Visualizer")
    print("=" * 50)
    
    # Create a sample network
    model = create_sample_network()
    
    # Create sample input and target
    input_data = torch.tensor([1.0, 0.5, -0.3])
    target = torch.tensor([1.0, 0.0])
    
    print(f"Network: {model}")
    print(f"Input: {input_data}")
    print(f"Target: {target}")
    print("\nUse LEFT and RIGHT arrow keys to navigate through propagation steps")
    print("=" * 50)
    
    # Create visualizer
    try:
        win = NeuralNetworkVisualizer(model, input_data, target)
        win.connect("destroy", Gtk.main_quit)
        win.show_all()
        Gtk.main()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
