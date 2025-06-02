"""
Physics-Informed Neural Network (PINN) for solving Burgers' equation.

This script implements a PINN to solve the Burgers' equation:
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

with initial and boundary conditions.
"""

import os
import time
import psutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

class BurgersPINN:
    def __init__(self, layers=[2, 20, 20, 20, 1], lb=[0, 0], ub=[1, 1], nu=0.01/np.pi):
        """
        Initialize the PINN for Burgers' equation.
        
        Args:
            layers: Neural network architecture (input dim, hidden layers, output dim)
            lb: Lower bounds of the domain [x_min, t_min]
            ub: Upper bounds of the domain [x_max, t_max]
            nu: Viscosity parameter
        """
        self.layers = layers
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.nu = nu
        
        # Initialize neural network
        self.model = self.build_model(layers)
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def build_model(self, layers):
        """
        Build the neural network model.
        
        Args:
            layers: Neural network architecture
            
        Returns:
            model: TensorFlow model
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        
        for width in layers[1:-1]:
            model.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer=tf.keras.initializers.GlorotNormal()
            ))
            
        model.add(tf.keras.layers.Dense(
            layers[-1], activation=None,
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        ))
        
        return model
    
    def net_u(self, x, t):
        """
        The neural network that approximates the solution u(x,t).
        
        Args:
            x: Spatial coordinate tensor
            t: Time coordinate tensor
            
        Returns:
            u: Approximated solution
        """
        # Normalize the inputs
        X = tf.concat([x, t], 1)
        X_normalized = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        
        # Forward pass through the network
        u = self.model(X_normalized)
        
        return u
    
    def net_f(self, x, t):
        """
        The residual of the PDE.
        
        Args:
            x: Spatial coordinate tensor
            t: Time coordinate tensor
            
        Returns:
            f: PDE residual
        """
        # Use GradientTape to compute derivatives
        with tf.GradientTape(persistent=True) as tape:
            # Variables need to be watched
            tape.watch(x)
            tape.watch(t)
            
            # Compute u(x,t)
            u = self.net_u(x, t)
            
            # First derivatives
            u_x = tape.gradient(u, x)
            u_t = tape.gradient(u, t)
        
        # Second derivative
        u_xx = tape.gradient(u_x, x)
        
        # Clean up the tape
        del tape
        
        # Burgers' equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
        f = u_t + u * u_x - self.nu * u_xx
        
        return f
    
    @tf.function
    def loss_fn(self, x_u, t_u, u, x_f, t_f):
        """
        Compute the loss function.
        
        Args:
            x_u: x-coordinates of training points
            t_u: t-coordinates of training points
            u: u values at training points
            x_f: x-coordinates of collocation points
            t_f: t-coordinates of collocation points
            
        Returns:
            loss: Total loss value
        """
        # Data loss
        u_pred = self.net_u(x_u, t_u)
        loss_u = tf.reduce_mean(tf.square(u - u_pred))
        
        # Physics loss
        f_pred = self.net_f(x_f, t_f)
        loss_f = tf.reduce_mean(tf.square(f_pred))
        
        # Total loss
        loss = loss_u + loss_f
        
        return loss, loss_u, loss_f
    
    @tf.function
    def train_step(self, x_u, t_u, u, x_f, t_f):
        """
        Perform one training step.
        
        Args:
            x_u: x-coordinates of training points
            t_u: t-coordinates of training points
            u: u values at training points
            x_f: x-coordinates of collocation points
            t_f: t-coordinates of collocation points
            
        Returns:
            loss: Total loss value
            loss_u: Data loss value
            loss_f: Physics loss value
        """
        with tf.GradientTape() as tape:
            loss, loss_u, loss_f = self.loss_fn(x_u, t_u, u, x_f, t_f)
            
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, loss_u, loss_f
    
    def train(self, X_u, u, X_f, epochs=10000, batch_size=32):
        """
        Train the PINN model.
        
        Args:
            X_u: Training data points for u(x,t) [x, t]
            u: Training data values for u(x,t)
            X_f: Collocation points for enforcing the PDE
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            loss_history: List of loss values during training
        """
        # Extract the coordinates
        x_u = X_u[:, 0:1]
        t_u = X_u[:, 1:2]
        
        x_f = X_f[:, 0:1]
        t_f = X_f[:, 1:2]
        
        # Convert to TensorFlow tensors
        x_u = tf.convert_to_tensor(x_u, dtype=tf.float32)
        t_u = tf.convert_to_tensor(t_u, dtype=tf.float32)
        u = tf.convert_to_tensor(u, dtype=tf.float32)
        x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
        t_f = tf.convert_to_tensor(t_f, dtype=tf.float32)
        
        # Number of training and collocation points
        N_u = X_u.shape[0]
        N_f = X_f.shape[0]
        
        # Create TF datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_u, t_u, u))
        train_dataset = train_dataset.shuffle(buffer_size=N_u).batch(batch_size)
        
        collocation_dataset = tf.data.Dataset.from_tensor_slices((x_f, t_f))
        collocation_dataset = collocation_dataset.shuffle(buffer_size=N_f).batch(batch_size)
        
        # Training loop
        loss_history = []
        start_time = time.time()
        
        for epoch in range(epochs):
            # Reset metrics
            epoch_loss = 0
            epoch_loss_u = 0
            epoch_loss_f = 0
            num_batches = 0
            
            # Create a new iterator for each epoch
            collocation_iter = iter(collocation_dataset.repeat())
            
            # Mini-batch training
            for x_u_batch, t_u_batch, u_batch in train_dataset:
                # Get a batch of collocation points
                x_f_batch, t_f_batch = next(collocation_iter)
                
                # Train on the batch
                loss, loss_u, loss_f = self.train_step(x_u_batch, t_u_batch, u_batch, x_f_batch, t_f_batch)
                
                # Accumulate metrics
                epoch_loss += loss
                epoch_loss_u += loss_u
                epoch_loss_f += loss_f
                num_batches += 1
            
            # Compute average loss for the epoch
            epoch_loss /= num_batches
            epoch_loss_u /= num_batches
            epoch_loss_f /= num_batches
            
            # Append to history
            loss_history.append(epoch_loss.numpy())
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                print(f'Epoch: {epoch}, Loss: {epoch_loss:.6e}, Data Loss: {epoch_loss_u:.6e}, Physics Loss: {epoch_loss_f:.6e}, Time: {elapsed:.2f} sec')
                start_time = time.time()
        
        return loss_history
    
    def predict(self, X):
        """
        Predict the solution at given points.
        
        Args:
            X: Points at which to predict [x, t]
            
        Returns:
            u: Predicted solution values
        """
        x = tf.convert_to_tensor(X[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(X[:, 1:2], dtype=tf.float32)
        
        u = self.net_u(x, t)
        
        return u.numpy()

def exact_solution(x, t, nu=0.01/np.pi):
    """
    Compute the exact solution of Burgers' equation with the given initial condition.
    
    Args:
        x: Spatial coordinates
        t: Time coordinates
        nu: Viscosity parameter
        
    Returns:
        u: Exact solution
    """
    # This is a simplified exact solution for a specific initial condition
    # For a more general solution, one would need to solve the Cole-Hopf transformation
    
    # Example: u(x,0) = -sin(pi*x)
    return -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)

def generate_training_data(n_u=100, n_f=10000, nu=0.01/np.pi):
    """
    Generate training data for the PINN.
    
    Args:
        n_u: Number of training points for u(x,t)
        n_f: Number of collocation points for enforcing the PDE
        nu: Viscosity parameter
        
    Returns:
        X_u: Training data points for u(x,t) [x, t]
        u: Training data values for u(x,t)
        X_f: Collocation points for enforcing the PDE
    """
    # Domain bounds
    lb = np.array([0.0, 0.0])  # Lower bounds [x_min, t_min]
    ub = np.array([1.0, 1.0])  # Upper bounds [x_max, t_max]
    
    # Initial condition: u(x,0) = -sin(pi*x)
    x_ic = np.linspace(lb[0], ub[0], n_u//2)[:, None]
    t_ic = np.zeros_like(x_ic)
    u_ic = exact_solution(x_ic, t_ic, nu)
    
    # Boundary conditions: u(0,t) = u(1,t) = 0
    t_bc = np.linspace(lb[1], ub[1], n_u//2)[:, None]
    x_bc_0 = np.zeros_like(t_bc)
    x_bc_1 = np.ones_like(t_bc)
    u_bc_0 = exact_solution(x_bc_0, t_bc, nu)
    u_bc_1 = exact_solution(x_bc_1, t_bc, nu)
    
    # Combine initial and boundary conditions
    X_u = np.vstack([
        np.hstack([x_ic, t_ic]),
        np.hstack([x_bc_0, t_bc]),
        np.hstack([x_bc_1, t_bc])
    ])
    u = np.vstack([u_ic, u_bc_0, u_bc_1])
    
    # Collocation points for enforcing the PDE
    x_f = lb[0] + (ub[0] - lb[0]) * np.random.random(n_f)[:, None]
    t_f = lb[1] + (ub[1] - lb[1]) * np.random.random(n_f)[:, None]
    X_f = np.hstack([x_f, t_f])
    
    return X_u, u, X_f

def compute_metrics(model, nu=0.01/np.pi, nx=100, nt=100):
    """
    Compute metrics for the PINN model.
    
    Args:
        model: Trained PINN model
        nu: Viscosity parameter
        nx: Number of points in x-direction for evaluation
        nt: Number of points in t-direction for evaluation
        
    Returns:
        mse: Mean squared error compared to exact solution
        runtime: Runtime in seconds
        memory: Memory usage in GB
    """
    # Domain for evaluation
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    # Reshape for prediction
    X_star = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    
    # Measure runtime
    start_time = time.time()
    u_pred = model.predict(X_star)
    runtime = time.time() - start_time
    
    # Compute exact solution
    u_exact = exact_solution(X_star[:, 0:1], X_star[:, 1:2], nu)
    
    # Compute MSE
    mse = np.mean((u_exact - u_pred)**2)
    
    # Measure memory usage
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 ** 3)  # in GB
    
    return mse, runtime, memory

def plot_solution(model, nu=0.01/np.pi, nx=100, nt=100):
    """
    Plot the PINN solution and compare with the exact solution.
    
    Args:
        model: Trained PINN model
        nu: Viscosity parameter
        nx: Number of points in x-direction for plotting
        nt: Number of points in t-direction for plotting
    """
    # Domain for plotting
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    # Reshape for prediction
    X_star = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    
    # Predict using the PINN model
    u_pred = model.predict(X_star)
    
    # Compute exact solution
    u_exact = exact_solution(X_star[:, 0:1], X_star[:, 1:2], nu)
    
    # Reshape for plotting
    U_pred = u_pred.reshape(nt, nx)
    U_exact = u_exact.reshape(nt, nx)
    
    # Compute error
    error = np.abs(U_exact - U_pred)
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # Plot PINN solution
    ax1 = fig.add_subplot(131)
    h1 = ax1.imshow(U_pred, interpolation='nearest', cmap='jet', 
                   extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$t$')
    ax1.set_title('PINN Solution')
    fig.colorbar(h1, ax=ax1)
    
    # Plot exact solution
    ax2 = fig.add_subplot(132)
    h2 = ax2.imshow(U_exact, interpolation='nearest', cmap='jet', 
                   extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$t$')
    ax2.set_title('Exact Solution')
    fig.colorbar(h2, ax=ax2)
    
    # Plot error
    ax3 = fig.add_subplot(133)
    h3 = ax3.imshow(error, interpolation='nearest', cmap='jet', 
                   extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$t$')
    ax3.set_title('Absolute Error')
    fig.colorbar(h3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('burgers_pinn_solution.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_with_fem_hmc(pinn_mse, pinn_runtime, pinn_memory):
    """
    Compare PINN metrics with FEM and HMC.
    
    Args:
        pinn_mse: MSE of the PINN solution
        pinn_runtime: Runtime of the PINN solution in seconds
        pinn_memory: Memory usage of the PINN solution in GB
    """
    # FEM and HMC metrics from Table 1
    methods = ['FEM', 'HMC', 'PINN']
    mse = [3.2e-5, 1.1e-3, pinn_mse]
    runtime = [0.5, 4.7, pinn_runtime / 3600]  # Convert seconds to hours
    memory = [2.1, 8.3, pinn_memory]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(methods))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    ax.bar(r1, mse, width=barWidth, label='MSE', color='blue')
    ax.bar(r2, runtime, width=barWidth, label='Runtime (hours)', color='green')
    ax.bar(r3, memory, width=barWidth, label='Memory (GB)', color='red')
    
    # Add xticks on the middle of the group bars
    ax.set_xlabel('Method', fontweight='bold', fontsize=12)
    ax.set_xticks([r + barWidth for r in range(len(methods))])
    ax.set_xticklabels(methods)
    
    # Create legend & show graphic
    ax.set_yscale('log')
    ax.set_ylabel('Value (log scale)', fontsize=12)
    ax.set_title('Comparison of Methods: MSE, Runtime, and Memory Usage', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the PINN for Burgers' equation.
    """
    # Set viscosity parameter
    nu = 0.01/np.pi
    
    # Generate training data
    X_u, u, X_f = generate_training_data(n_u=200, n_f=10000, nu=nu)
    
    # Create and train the PINN model
    model = BurgersPINN(layers=[2, 20, 20, 20, 1], nu=nu)
    
    # Start timing
    start_time = time.time()
    
    # Train the model
    loss_history = model.train(X_u, u, X_f, ep
(Content truncated due to size limit. Use line ranges to read in chunks)