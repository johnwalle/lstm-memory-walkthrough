import numpy as np
import matplotlib.pyplot as plt



# ------------------------------
# Extended LSTM Cell Implementation
# ------------------------------
class ExtendedLSTM:
    def __init__(self, input_size=1, hidden_size=1, seed=42):
        np.random.seed(seed)  # For reproducibility
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight initialization (small random values)
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_f = np.zeros((hidden_size, 1))
        
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_i = np.zeros((hidden_size, 1))
        
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_o = np.zeros((hidden_size, 1))
        
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_c = np.zeros((hidden_size, 1))
    
    # Sigmoid activation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Tanh activation
    def tanh(self, x):
        return np.tanh(x)
    
    # Forward step for one input
    def step(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))  # Combine input and previous hidden state
        
        # Forget gate
        f = self.sigmoid(np.dot(self.W_f, concat) + self.b_f)
        # Input gate
        i = self.sigmoid(np.dot(self.W_i, concat) + self.b_i)
        # Candidate cell state
        c_bar = self.tanh(np.dot(self.W_c, concat) + self.b_c)
        # New cell state
        c = f * c_prev + i * c_bar
        # Output gate
        o = self.sigmoid(np.dot(self.W_o, concat) + self.b_o)
        # Hidden state
        h = o * self.tanh(c)
        
        # Return all values for analysis
        return h, c, f, i, o, c_bar

# ------------------------------
# Run LSTM on an input sequence
# ------------------------------
inputs = [5, 1, 9, 2]  # Example sequence
lstm = ExtendedLSTM(hidden_size=1)

# Initialize hidden and cell states
h = np.zeros((1,1))
c = np.zeros((1,1))

# Lists to store all intermediate values for analysis
hidden_states, cell_states = [], []
forget_gates, input_gates, output_gates, candidate_cells = [], [], [], []

print("Input Sequence:", inputs)
print("-" * 80)

# Step through the sequence
for t, x in enumerate(inputs, 1):
    x_array = np.array([[x]])
    h, c, f, i, o, c_bar = lstm.step(x_array, h, c)
    
    # Store values for analysis
    hidden_states.append(h.item())
    cell_states.append(c.item())
    forget_gates.append(f.item())
    input_gates.append(i.item())
    output_gates.append(o.item())
    candidate_cells.append(c_bar.item())
    
    # Print all gate and state values
    print(f"Step {t} | Input: {x}")
    print(f"  Hidden (h): {h.item():.4f} | Cell (c): {c.item():.4f}")
    print(f"  Forget gate (f): {f.item():.4f} | Input gate (i): {i.item():.4f}")
    print(f"  Output gate (o): {o.item():.4f} | Candidate (c_bar): {c_bar.item():.4f}")
    print("-" * 80)

# ------------------------------
# Plot hidden and cell states
# ------------------------------
plt.figure(figsize=(10,5))
plt.plot(hidden_states, label='Hidden State (h)', marker='o')
plt.plot(cell_states, label='Cell State (c)', marker='x')
plt.title("Hidden and Cell States Over Time")
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.legend()
plt.grid(True)
plt.show()
