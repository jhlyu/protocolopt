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
import opt_kyle as move
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
simulation_params['beta'] = torch.tensor(simulation_params['beta'], device='cpu', dtype=torch.float32)
simulation_params['noise_sigma'] = math.sqrt(2 / simulation_params['beta'])




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
    print("Loaded alist and blist from checkpoint.")

phase_data_generator = move.quadratic_simulation_od(num_paths=training_params['batch_size'], params=simulation_params, a_list=protocol_params['a_list'], a_endpoints=protocol_params['a_endpoints'])
phase_data, noise=phase_data_generator.trajectory_generator()
#left_phase_data, left_noise = phase_data[phase_data[:, 0, 0] < 0], noise[phase_data[:, 0, 0] < 0]
#right_phase_data, right_noise = phase_data[phase_data[:, 0, 0] > 0], noise[phase_data[:, 0, 0] > 0]



def protocol_plot():
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    a_value = move.piecewise_protocol_value(simulation_params['num_steps'],protocol_params['a_list'],protocol_params['a_endpoints'])
    time = torch.linspace(0,1, 1000+1)
    ax.plot(time.detach().cpu(), a_value.detach().cpu(), label='a(t)')
    ax.set_title('a(t)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('a(t)')
    # Save the figure
    fig.savefig(plot_dir + f'protocol_plot_{len(mean_distance_list)}.png', dpi=300, bbox_inches='tight')
    #plt.show()

def loss_plot():
    # loss and work plot
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    ax[0].plot([float(x) for x in mean_distance_list], label='loss')
    ax[1].plot([float(x) for x in work_list], label='work')
    ax[2].plot([float(x) for x in total_loss_list], label='total loss')
    
    ax[0].set_title('mean loss')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('loss')
    ax[1].set_title('work')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('work')
    ax[2].set_title('total loss')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('total loss')
    plt.savefig(plot_dir + f"mean_var_work_{training_params['alpha']}_{training_params['alpha_2']}_{len(mean_distance_list)}.png", dpi=300)
    #plt.show()

def position_traj_plot(num_plot=1000):
    fig, ax = plt.subplots()
    position = phase_data.detach().cpu()
    time = np.linspace(0, 1, 1000 + 1)
    for idx in range(num_plot):
        ax.plot(position[idx,:,0].numpy(), label=f'path {idx}', alpha=0.5)
    ax.set_title('Trajectories')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    plt.savefig(plot_dir+f"trajectories_flip_position__{training_params['alpha']}_{training_params['alpha_1']}_{len(mean_distance_list)}.png", dpi=300)
    #plt.show()

def initial_final_distribution_plot():
    position = phase_data.detach().cpu()
    fig, ax = plt.subplots(1,2,figsize=(12, 5))
    hist, bins = np.histogram(position[:,-1].numpy(), bins=50, density=True)
    ax[1].hist(position[:,-1,0].numpy(), bins=50, density=True, alpha=0.5, label='Final Distribution')
    ax[0].hist(position[:,0,0].numpy(), bins=50, density=True, alpha=0.5, label='Initial Distribution')
    ax[0].set_title('Initial Distribution')
    ax[1].set_title('Final Distribution')
    ax[0].set_xlabel('Position')
    ax[1].set_xlabel('Position')
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



def plot_all():
    protocol_plot()
    loss_plot()
    position_traj_plot()
    initial_final_distribution_plot()
    #work_distribution_plot()
    #phase_animation_plot()
    #position_animation_plot()

if __name__ == "__main__":
    plot_all()
