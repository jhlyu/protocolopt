import torch
import gc
import os
import json
import numpy as np
import opt_kyle as opt
import yaml
import math
import argparse
import shutil
import subprocess

parser = argparse.ArgumentParser(description='Run memory test on MPS tensors.')
parser.add_argument('config', type=str, default='config.yaml', help='Path to the configuration file.')
args = parser.parse_args()
print(f"Running optimization from {args.config}")

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

simulation_params = config["simulation_params"]
protocol_params = config["protocol_params"]
training_params = config["training_params"]
save_dir = config['save_directory'] 
initilize_to_mc = config['initilize_to_mc']
simulation_params['noise_sigma'] = math.sqrt(2  / simulation_params['beta'])  # od Einstein relation

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#setting the devices, simulation can be on a sepatate device

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
#torch_device = torch.device('cpu')
simulation_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
#simulation_device = torch.device('cpu')

if os.path.exists(save_dir + 'training_metrics.json'):
    with open(save_dir + 'training_metrics.json', 'r') as f:
        metrics = json.load(f)
    mean_distance_list = metrics['mean_distance_list']
    #var_list = metrics['var_list']
    work_list = metrics['work_list']
    total_loss_list = metrics['total_loss_list']
    a_list_history = metrics['a_list_history']
    print("Loaded previous training metrics.")
else:
    mean_distance_list = []
    work_list = []
    total_loss_list = []
    #var_list = []
    a_list_history = []
    print("Initialized new training metrics.")
# Load or initialize alist and blist
if os.path.exists(save_dir + 'work_checkpoint.pt'):
    checkpoint = torch.load(save_dir + 'work_checkpoint.pt', weights_only=True)
    
    protocol_params['a_list'] = checkpoint['a_list'].to(torch_device).requires_grad_()
    print("Loaded alist from checkpoint.")

    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    
    optimizer_a.load_state_dict(checkpoint['optimizer_a_dict'])
    print("Loaded optimizer states.")
else:
    protocol_params['a_list'] = torch.zeros(protocol_params['num_coefficients'], device=torch_device, requires_grad=True)
    if initilize_to_mc:
        analytic_k = torch.pi**2 * simulation_params['mass'] / (simulation_params['dt']*simulation_params['num_steps'])
        protocol_params['b_list'].data[:] = torch.tensor( [-analytic_k / 2 for i in range(protocol_params['num_coefficients'])] )

    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    print("Initialized new alist and.")

array_kwargs = {'dtype': torch.float32, 'device': torch_device}
phase_data = torch.empty((training_params['batch_size'], simulation_params['num_steps']+1, 2), **array_kwargs)
noise = torch.empty((training_params['batch_size'], simulation_params['num_steps']), **array_kwargs)
potential_value = torch.empty((training_params['batch_size'], simulation_params['num_steps']+1), **array_kwargs)
potential_value_advance = torch.empty((training_params['batch_size'], simulation_params['num_steps']), **array_kwargs)
drift_protocol_grad = torch.empty((protocol_params['num_coefficients'], training_params['batch_size'], simulation_params['num_steps']+1), **array_kwargs)
distance_sq_current = torch.empty((training_params['batch_size']), **array_kwargs)
potential_value_grad = torch.empty((training_params['batch_size'], simulation_params['num_steps']+1), **array_kwargs)
potential_value_advance_grad =  torch.empty((training_params['batch_size'], simulation_params['num_steps']), **array_kwargs)

for step in range(training_params['training_iterations']):
    optimizer_a.zero_grad()

    phase_data_generator = opt.quadratic_simulation_od(
        num_paths=training_params['batch_size'], 
        params=simulation_params, 
        a_list=protocol_params['a_list'].detach().clone(), 
        a_endpoints=protocol_params['a_endpoints'], 
        device=simulation_device)

    phase_data.data, noise.data = [item.to(torch_device) for item in phase_data_generator.trajectory_generator()]

    p_lists=[protocol_params['a_list']]
    p_endpoints=[protocol_params['a_endpoints']]

    move = opt.moving_harmonic(simulation_params, phase_data, protocol_params['a_list'], protocol_params['a_endpoints'] )
    grad = opt.grad_calc(simulation_params, protocol_params, device=torch_device)

    # assign the arrays to the GPU tensors
    potential_value.data = move.potential_value_array()
    potential_value_advance.data = move.potential_value_advance_array()
    distance_sq_current.data = move.distance_sq_current(phase_data, p_lists, p_endpoints[0][-1],order=2)

    for list_name, drift_gradient_func, potential_function, potential_advance_function in zip( ['a_list'],
                                               [move.drift_grad_a_array],
                                               [move.potential_grad_a_value_array],
                                               [move.potential_grad_a_value_advance_array]):

        # assign the arrays to the GPU tensors
        drift_protocol_grad.data = grad.drift_protocol_grad_array(drift_gradient_func())
        potential_value_grad.data = potential_function()
        potential_value_advance_grad.data = potential_advance_function()
        
        # Calculate the gradients
        work_grad = grad.work_protocol_grad(potential_value, potential_value_advance, potential_value_grad, potential_value_advance_grad, drift_protocol_grad, noise)
        distance_grad = grad.current_grad(distance_sq_current, drift_protocol_grad, noise)
        param_grad = training_params['alpha_2'] * work_grad + training_params['alpha'] * distance_grad
        protocol_params[list_name].grad = param_grad

    distance_loss = distance_sq_current.mean()
    work_mean = grad.work_array(potential_value, potential_value_advance).sum(axis=-1).mean()
    total_loss = training_params['alpha'] * distance_loss + training_params['alpha_2'] * work_mean
    mean_distance_list.append(distance_loss.item())
    work_list.append(work_mean.item())
    total_loss_list.append(total_loss.item())

    a_list_history.append(protocol_params['a_list'].detach().cpu().clone().tolist())
    
    optimizer_a.step()

    total_iterations = len(mean_distance_list)
    if step % config['report_frequency'] == 0:
        print(f"Step {step}: a = {protocol_params['a_list']}, x = {distance_loss}, work = {work_mean}, total loss = {total_loss}")

    #check to see if we want to plot or checkpoint this iteration
    checkpoint_boolean = total_iterations % config['checkpoint_frequency'] == 0 or step == training_params['training_iterations'] - 1 or total_iterations == 1
    plot_boolean = total_iterations % config['plot_frequency'] == 0 or step == training_params['training_iterations'] - 1 or total_iterations == 1

    # Save the lists after training using JSON
    if checkpoint_boolean or plot_boolean:
        print(f"Saving checkpoint at step {step}, total_iter {total_iterations}...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the training metrics
        metrics = {
                'mean_distance_list': mean_distance_list,
                #'var_list': var_list,
                'work_list': work_list,
                'total_loss_list': total_loss_list,
                'a_list_history': a_list_history,
                #'b_list_history': b_list_history
            }
        with open(save_dir + 'training_metrics.json', 'w') as f:
            json.dump(metrics, f)   


        torch.save({
            'a_list': protocol_params['a_list'],
            #'b_list': protocol_params['b_list'],
            #'optimizer_b_dict': optimizer_b.state_dict(),
            'optimizer_a_dict': optimizer_a.state_dict(),
        }, save_dir + 'work_checkpoint.pt') 

        shutil.copyfile(args.config, save_dir + 'config.yaml')
        print("Checkpoint saved.")

        if plot_boolean:
            print(f"Plotting checkpoint at step {step}, total_iter {total_iterations}")
            res = subprocess.call(['python', 'moving_plot.py', save_dir])
            print("Checkpoint plotted.")