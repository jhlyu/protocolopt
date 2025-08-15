import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import gc
from collections import defaultdict
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
import opt_kyle as erase
import argparse

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


parser = argparse.ArgumentParser(description='Run memory test on MPS tensors.')
parser.add_argument('directory', type=str, default='./results/', help='Path to the configuration file.')
args = parser.parse_args()
print(f"Running plot script from {args.directory}")

save_dir = args.directory

with open(save_dir+'config.yaml', "r") as f:
    config = yaml.safe_load(f)

simulation_params = config["simulation_params"]
protocol_params = config["protocol_params"]
training_params = config["training_params"]

plot_dir = save_dir + 'plots/' 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


protocol_params['a_endpoints'] = torch.tensor(protocol_params['a_endpoints'], device='cpu', dtype=torch.float32)
protocol_params['b_endpoints'] = torch.tensor(protocol_params['b_endpoints'], device='cpu', dtype=torch.float32)
protocol_params['c_endpoints'] = torch.tensor(protocol_params['c_endpoints'], device='cpu', dtype=torch.float32)
simulation_params['beta'] = torch.tensor(simulation_params['beta'], device='cpu', dtype=torch.float32)
simulation_params['noise_sigma'] = math.sqrt(2* simulation_params['gamma'] / simulation_params['beta'])




if os.path.exists(save_dir + 'training_metrics.json'):
    with open( save_dir + 'training_metrics.json', 'r') as f:
        metrics = json.load(f)
    mean_distance_list = metrics['mean_distance_list']
    work_list = metrics['work_list']
    #var_list = metrics['var_list']
    total_loss_list = metrics['total_loss_list']
    print("Loaded previous training metrics.")
else:
    print("No available training metrics to plot.")

if os.path.exists(save_dir + 'work_checkpoint.pt'):
    checkpoint = torch.load(save_dir + 'work_checkpoint.pt',weights_only=True)
    
    protocol_params['a_list'] = checkpoint['a_list'].to(torch_device).requires_grad_()
    protocol_params['b_list'] = checkpoint['b_list'].to(torch_device).requires_grad_()
    protocol_params['c_list'] = checkpoint['c_list'].to(torch_device).requires_grad_()
    print("Loaded alist and blist from checkpoint.")

phase_data_generator = erase.quartic_simulation_with_linear_od(num_paths=training_params['batch_size']*20, params=simulation_params, a_list=protocol_params['a_list'], b_list=protocol_params['b_list'], c_list=protocol_params['c_list'], a_endpoints=protocol_params['a_endpoints'], b_endpoints=protocol_params['b_endpoints'], c_endpoints=protocol_params['c_endpoints'])
phase_data, noise=phase_data_generator.trajectory_generator()
biterase = erase.bit_erasure(simulation_params, phase_data, protocol_params['a_list'], protocol_params['b_list'], protocol_params['c_list'], protocol_params['a_endpoints'], protocol_params['b_endpoints'], protocol_params['c_endpoints'])
grad = erase.grad_calc(simulation_params, protocol_params, device=torch_device)
work= grad.work_array(biterase.potential_value_array(), biterase.potential_value_advance_array())
#left_phase_data, left_noise = phase_data[phase_data[:, 0, 0] < 0], noise[phase_data[:, 0, 0] < 0]
#right_phase_data, right_noise = phase_data[phase_data[:, 0, 0] > 0], noise[phase_data[:, 0, 0] > 0]

workerrname = 'work_error_list.json'
workerr_path = os.path.join(save_dir, workerrname)
num_error_paths = (phase_data[:, -1, 0] < 0).sum().item()
mean_work = work.sum(axis=1).mean()

if os.path.exists(workerr_path):
    print("Work and error lists exist. Loading and updating...")
    with open(workerr_path, 'r') as f:
        work_err_list = json.load(f)

    # Example update
    work_err_list['num_error_paths'].append(int(num_error_paths))
    work_err_list['mean_work'].append(float(mean_work))
else:
    print("Work and error lists do not exist. Creating new...")
    work_err_list = {
        "num_error_paths": [int(num_error_paths)],
        "mean_work": [float(mean_work)]
    }

# Save the updated or new data
with open(workerr_path, 'w') as f:
    json.dump(work_err_list, f, indent=2)



def protocol_plot():


    fig, ax = plt.subplots(1,3, figsize=(24, 6))
    a_value = erase.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['a_list'],protocol_params['a_endpoints'])
    b_value = erase.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['b_list'],protocol_params['b_endpoints'])
    c_value = erase.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['c_list'],protocol_params['c_endpoints'])
    time = torch.linspace(0,simulation_params['num_steps']*simulation_params['dt'], simulation_params['num_steps'] + 1)
    ax[0].plot(time.detach().cpu(), a_value.detach().cpu(), label='$a_t$')
    ax[1].plot(time.detach().cpu(), b_value.detach().cpu(), label='$b_t$')
    ax[2].plot(time.detach().cpu(), c_value.detach().cpu(), label='$c_t$')
    ax[0].set_title('Protocol $a_t$ Values', fontsize=20)
    ax[0].set_xlabel('Time', fontsize=20)
    ax[0].set_ylabel('$a_t$', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=16)


    ax[1].set_title('Protocol $b_t$ Values', fontsize=20)
    ax[1].set_xlabel('Time', fontsize=20)
    ax[1].set_ylabel('$b_t$', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=16)

    ax[2].set_title('Protocol $c_t$ Values', fontsize=20)
    ax[2].set_xlabel('Time', fontsize=20)
    ax[2].set_ylabel('$c_t$', fontsize=20)
    ax[2].tick_params(axis='both', which='major', labelsize=16)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # Save the figure
    fig.savefig(plot_dir + f'protocol_plot_{len(mean_distance_list)}.png', dpi=300, bbox_inches='tight')
    #plt.show()

def loss_plot():
   
    
    # loss and work plot
    fig, ax = plt.subplots(1, 3, figsize=(27, 6))
    ax[0].plot([float(x) for x in mean_distance_list], label='loss')
    ax[1].plot([float(x) for x in work_list], label='work')
    ax[2].plot([float(x) for x in total_loss_list], label='total loss')

    ax[0].set_title('mean loss', fontsize=20)
    ax[0].set_xlabel('Step', fontsize=20)
    ax[0].set_ylabel('loss', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[1].set_title('work', fontsize=20)
    ax[1].set_xlabel('Step', fontsize=20)
    ax[1].set_ylabel('work', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    ax[2].set_title('total loss', fontsize=20)
    ax[2].set_xlabel('Step', fontsize=20)
    ax[2].set_ylabel('total loss', fontsize=20)
    ax[2].tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(plot_dir + f"mean_var_work_{training_params['alpha']}_{training_params['alpha_2']}_{len(mean_distance_list)}.png", dpi=300)
    #plt.show()

def position_traj_plot(num_plot=1000):
    fig, ax = plt.subplots(figsize=(12, 10))
    num_iteration = len(work_list)
    position = phase_data.detach().cpu()
    mean_work = work.sum(axis=-1).mean()
    time = np.linspace(0,simulation_params['num_steps']*simulation_params['dt'], simulation_params['num_steps'] + 1)
    for idx in range(num_plot):
        ax.plot(time, position[idx,:,0].numpy(), label=f'path {idx}', alpha=0.5)
    ax.axhline(0, color='black', linestyle='dashed', linewidth=16.0, label='y=0')
    #ax.set_title('Trajectories', fontsize=40)
    #ax.set_xlabel('Time', fontsize=40)
    #ax.set_ylabel('Position', fontsize=40)
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=64)
    ax.set_ylim(-2, 2)
    ax.text(0.6, 0.05, f'Iteration {num_iteration}', transform=ax.transAxes, ha='center', fontsize=64)

    plt.savefig(plot_dir+f"trajectories_flip_position__{training_params['alpha']}_{training_params['alpha_1']}_{len(mean_distance_list)}.png", dpi=300)
    #plt.show()

def initial_final_distribution_plot():
    position = phase_data.detach().cpu()
    fig, ax = plt.subplots(1,2,figsize=(12, 6))
    hist, bins = np.histogram(position[:,-1].numpy(), bins=50, density=True)
    ax[1].hist(position[:,-1,0].numpy(), bins=50, density=True, alpha=0.5, label='Final Histogram')
    ax[0].hist(position[:,0,0].numpy(), bins=50, density=True, alpha=0.5, label='Initial Histogram')
    ax[0].set_title('Initial Histogram', fontsize=20)
    ax[1].set_title('Final Histogram', fontsize=20)
    ax[0].set_xlabel('Position', fontsize=20)
    ax[1].set_xlabel('Position', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    x = np.linspace(-3, 3, 1000)
    plt.savefig(plot_dir + f"distribution_{training_params['alpha']}_{training_params['alpha_1']}_{len(mean_distance_list)}.png", dpi=300)


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

def position_animation_plot():

    fig, ax = plt.subplots()
    trajectory_data = phase_data.detach().cpu()[:100,...]
    a_value = erase.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['a_list'],protocol_params['a_endpoints'])
    b_value = erase.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['b_list'],protocol_params['b_endpoints'])
    c_value = erase.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['c_list'],protocol_params['c_endpoints'])

    a_list = a_value.detach().cpu()
    b_list = b_value.detach().cpu()
    c_list = c_value.detach().cpu()
    num_frames = a_list.shape[0]
    # x-domain for plotting the potential
    x = torch.linspace(-2, 2, 500)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=2, label="V(x, t)")
    dots_left, = ax.plot([], [], 'ro', markersize=3, alpha=0.6, label="Left particles")
    

    ax.set_xlim(-2, 2)
    ax.set_ylim(-11, 3)
    ax.set_xlabel("x", fontsize=20)
    ax.set_ylabel("V(x, t)", fontsize=20)
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc="upper right")

    # Update function for animation
    def update(frame):
        a = a_list[frame]
        b = b_list[frame]
        c = c_list[frame]
        V = a*x**4 - b * x**2 + c*x
        line.set_data(x.numpy(), V.numpy())
        title.set_text(f"Time {(frame*0.001):.3f}, a = {a.item():.2f}, b = {b.item():.2f}, c = {c.item():.2f}")
        title.set_fontsize(20)

        # Left particle positions
        x_left = trajectory_data[:, frame,0]
        y_left = a*x_left**4 - b * x_left**2 + c*x_left
        dots_left.set_data(x_left.numpy(), y_left.numpy())


        return line, dots_left, title

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
    #initial_final_distribution_plot()
    #work_distribution_plot()
    #phase_animation_plot()
    #position_animation_plot()

if __name__ == "__main__":
    plot_all()
