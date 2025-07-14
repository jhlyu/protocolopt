import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import gc
from collections import defaultdict
import os
import json
import math
import yaml
import opt_kyle as bit_flip

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

simulation_params = config["simulation_params"]
protocol_params = config["protocol_params"]
training_params = config["training_params"]

save_dir = config['save_directory'] 
plot_dir = save_dir + 'plots/' 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


protocol_params['b_endpoints'] = torch.tensor(protocol_params['b_endpoints'], device='cpu', dtype=torch.float32)
protocol_params['a_endpoints'] = torch.tensor(protocol_params['a_endpoints'], device='cpu', dtype=torch.float32)
simulation_params['beta'] = torch.tensor(simulation_params['beta'], device='cpu', dtype=torch.float32)
simulation_params['noise_sigma'] = math.sqrt(2 * simulation_params['gamma'] / simulation_params['beta'])




if os.path.exists(save_dir + 'training_metrics.json'):
    with open( save_dir + 'training_metrics.json', 'r') as f:
        metrics = json.load(f)
    mean_distance_list = metrics['mean_distance_list']
    var_distance_list = metrics['var_distance_list']
    work_list = metrics['work_list']
    print("Loaded previous training metrics.")
else:
    print("No available training metrics to plot.")

if os.path.exists(save_dir + 'work_checkpoint.pt'):
    checkpoint = torch.load(save_dir + 'work_checkpoint.pt',weights_only=True)
    
    protocol_params['a_list'] = checkpoint['a_list'].to(torch_device).requires_grad_()
    protocol_params['b_list']= checkpoint['b_list'].to(torch_device).requires_grad_()
    print("Loaded alist and blist from checkpoint.")

phase_data_generator = bit_flip.quartic_simulation(num_paths=training_params['batch_size'], params=simulation_params, a_list=protocol_params['a_list'], b_list=protocol_params['b_list'], a_endpoints=protocol_params['a_endpoints'], b_endpoints=protocol_params['b_endpoints'])
phase_data, noise=phase_data_generator.trajectory_generator()
left_phase_data, left_noise = phase_data[phase_data[:, 0, 0] < 0], noise[phase_data[:, 0, 0] < 0]
right_phase_data, right_noise = phase_data[phase_data[:, 0, 0] > 0], noise[phase_data[:, 0, 0] > 0]



def protocol_plot():
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    a_value = bit_flip.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['a_list'],protocol_params['a_endpoints'])
    b_value = bit_flip.piecewise_protocol_value(simulation_params['num_steps'], protocol_params['b_list'],protocol_params['b_endpoints'])
    time = torch.linspace(0,1, 1000+1)
    ax[0].plot(time.detach().cpu(), a_value.detach().cpu(), label='a(t)')
    ax[1].plot(time.detach().cpu(), b_value.detach().cpu(), label='b(t)')
    ax[0].set_title('a(t)')
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('a(t)')
    ax[1].set_title('b(t)')
    ax[1].set_xlabel('Time step')
    ax[1].set_ylabel('b(t)')
    # Save the figure
    fig.savefig(plot_dir + f'protocol_plot_{len(mean_distance_list)}.png', dpi=300, bbox_inches='tight')
    #plt.show()

def loss_plot():
    # loss and work plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot([float(x) for x in mean_distance_list], label='loss')
    ax[1].plot([float(x) for x in var_distance_list], label='variance loss')
    ax[2].plot([float(x) for x in work_list], label='work')

    #ax[0].set_title('mean loss')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('loss')
    #ax[1].set_title('variance loss')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('variance loss')
    #ax[2].set_title('work')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('work')
    plt.savefig(plot_dir + f"mean_var_work_{training_params['alpha']}_{training_params['alpha_1']}_{training_params['alpha_2']}_{len(mean_distance_list)}.png", dpi=300)
    #plt.show()

def position_traj_plot(num_plot=1000):
    fig, ax = plt.subplots()
    position_left = left_phase_data.detach().cpu()
    position_right = right_phase_data.detach().cpu()
    time = np.linspace(0, 1, 1000 + 1)
    for idx in range(num_plot):
        ax.plot(position_left[idx,:,0].numpy(), label=f'path {idx}', alpha=0.5)

    for idx in range(num_plot):
        ax.plot(position_right[idx,:,0].numpy(), label=f'path {idx}',alpha=0.5)
    ax.set_title('Trajectories')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    plt.savefig(plot_dir+f"trajectories_flip_position__{training_params['alpha']}_{training_params['alpha_1']}_{training_params['alpha_2']}_{len(mean_distance_list)}.png", dpi=300)
    #plt.show()

def phase_traj_plot(num_plot=1000):
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    position_left = left_phase_data.detach().cpu()
    position_right = right_phase_data.detach().cpu()
    for i in range(100):
        position = position_left[i, :, 0]
        velocity = position_left[i, :, 1]
        ax[0].plot(position, velocity, lw=1)
    for i in range(100):
        position = position_right[i, :, 0]
        velocity = position_right[i, :, 1]
        ax[1].plot(position, velocity, lw=1)

    ax[0].set_xlabel('Position')
    ax[0].set_ylabel('Velocity')
    ax[0].set_title('Phase Space: Left Trajectory')
    ax[0].grid(True)

    ax[1].set_xlabel('Position')
    ax[1].set_ylabel('Velocity')
    ax[1].set_title('Phase Space: Right Trajectory')
    ax[1].grid(True)

    plt.savefig(plot_dir + f"phase_space_flip_{training_params['alpha']}_{training_params['alpha_1']}_{training_params['alpha_2']}_{len(mean_distance_list)}.png", dpi=300)
    #plt.show()

def initial_final_distribution_plot():
    position_left = left_phase_data.detach().cpu()
    position_right = right_phase_data.detach().cpu()
    fig, ax = plt.subplots(2,2,figsize=(12, 9))
    position = position_left.detach().cpu()
    hist, bins = np.histogram(position[:,-1].numpy(), bins=50, density=True)
    ax[0,1].hist(position_left[:,-1,0].numpy(), bins=50, density=True, alpha=0.5, label='Final Left Distribution')
    ax[0,0].hist(position_left[:,0,0].numpy(), bins=50, density=True, alpha=0.5, label='Initial Left Distribution')
    ax[1,1].hist(position_right[:,-1,0].numpy(), bins=50, density=True, alpha=0.5, label='Final Right Distribution')
    ax[1,0].hist(position_right[:,0,0].numpy(), bins=50, density=True, alpha=0.5, label='Initial Right Distribution')
    ax[0,0].set_title('Initial Left Distribution')
    ax[0,1].set_title('Final Left Distribution')
    ax[1,0].set_title('Initial Right Distribution')
    ax[1,1].set_title('Final Right Distribution')

    x = np.linspace(-3, 3, 1000)
    plt.savefig(plot_dir + f"distribution_{training_params['alpha']}_{training_params['alpha_1']}_{training_params['alpha_2']}_{len(mean_distance_list)}.png", dpi=300)


def work_distribution_plot():
    batch_size = 10_000
    work1 = grad_left.work_array().sum(axis=1)
    work2 = grad_right.work_array().sum(axis=1)
    work1 = work1.detach().cpu().numpy()
    work2 = work2.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.hist(work1, bins=90, density=True, alpha=0.5, label='Work Distribution (Left)')
    ax.hist(work2, bins=90, density=True, alpha=0.5, label='Work Distribution (Right)')
    ax.set_title('Work Distribution')
    ax.set_xlabel('Work')
    ax.set_ylabel('Density')
    ax.legend()
    plt.savefig(plot_dir + f"work_distribution_{training_params['alpha']}_{training_params['alpha_1']}_{training_params['alpha_2']}_{len(mean_distance_list)}.png", dpi=300)


def phase_animation_plot():
    # Convert to CPU numpy arrays
    left_phase = left_phase_data.detach().cpu().numpy()
    right_phase = right_phase_data.detach().cpu().numpy()

    n_traj, n_time, _ = left_phase.shape

    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)     # Adjust based on your position range
    ax.set_ylim(-8, 8)     # Adjust based on your velocity range
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Phase Space Trajectories')

    # Create scatter plots for both datasets
    left_scatter = ax.scatter([], [], color='blue', label='Left', s=10)
    right_scatter = ax.scatter([], [], color='red', label='Right', s=10)
    ax.legend()

    # Update function for animation
    def update(frame):
        left_pos = left_phase[:, frame, 0]
        left_vel = left_phase[:, frame, 1]
        right_pos = right_phase[:, frame, 0]
        right_vel = right_phase[:, frame, 1]

        left_scatter.set_offsets(list(zip(left_pos, left_vel)))
        right_scatter.set_offsets(list(zip(right_pos, right_vel)))
        return left_scatter, right_scatter

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=n_time, interval=10, blit=True)

    # To display in a Jupyter notebook:
    #from IPython.display import HTML
    #HTML(ani.to_jshtml())

    # To save the animation (optional):
    ani.save(plot_dir+"phase_space_animation.mp4", writer='ffmpeg', fps=120)

    plt.show()

def position_animation_plot():

    fig, ax = plt.subplots()
    trajectory_data_left = left_phase_data
    trajectory_data_right = right_phase_data

    # Trajectory data (move to CPU if needed)
    trajectory_data_left = trajectory_data_left.detach().cpu()
    trajectory_data_right = trajectory_data_right.detach().cpu()
    n_left, _ = trajectory_data_left.shape[:-1]
    n_right, _ = trajectory_data_right.shape[:-1]
    a_list = phase_data_generator.piecewise_protocol_value(protocol_params['a_list'],protocol_params['a_endpoints'])
    b_list = phase_data_generator.piecewise_protocol_value(protocol_params['b_list'],protocol_params['b_endpoints'])
    a_list = a_list.detach().cpu()
    b_list = b_list.detach().cpu()
    num_frames = a_list.shape[0]
    # x-domain for plotting the potential
    x = torch.linspace(-5, 5, 500)

    # Setup the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2, label="V(x, t)")
    dots_left, = ax.plot([], [], 'ro', markersize=3, alpha=0.6, label="Left particles")
    dots_right, = ax.plot([], [], 'bo', markersize=3, alpha=0.6, label="Right particles")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-120, 120)
    ax.set_xlabel("x")
    ax.set_ylabel("V(x, t)")
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")
    ax.legend(loc="upper right")

    # Update function for animation
    def update(frame):
        a = a_list[frame]
        b = b_list[frame]
        V = a*x**4 - b * x**2 
        line.set_data(x.numpy(), V.numpy())
        title.set_text(f"Frame {frame}, a = {a.item():.2f}, b = {b.item():.2f}")

        # Left particle positions
        x_left = trajectory_data_left[:, frame,0]
        y_left = a*x_left**4 - b * x_left**2 
        dots_left.set_data(x_left.numpy(), y_left.numpy())

        # Right particle positions
        x_right = trajectory_data_right[:, frame,0]
        y_right = a*x_right**4 - b * x_right**2 
        dots_right.set_data(x_right.numpy(), y_right.numpy())

        return line, dots_left, dots_right, title

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=30, blit=False
    )

    # Save to video
    writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(plot_dir + f"potential_flip_{training_params['alpha']}_{training_params['alpha_1']}_{training_params['alpha_2']}_{len(mean_distance_list)}.mp4", writer=writer, dpi=100)

def plot_all():
    protocol_plot()
    loss_plot()
    position_traj_plot()
    phase_traj_plot()
    initial_final_distribution_plot()
    #work_distribution_plot()
    #phase_animation_plot()
    #position_animation_plot()

if __name__ == "__main__":
    plot_all()