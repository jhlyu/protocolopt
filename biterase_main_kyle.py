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
from datetime import datetime
from torch.optim.lr_scheduler import StepLR


timestamp = datetime.now().strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser(description='Run memory test on MPS tensors.')
parser.add_argument('config', type=str, default='config.yaml', help='Path to the configuration file.')
args = parser.parse_args()
print(f"Running optimization from {args.config}")


with open(args.config, "r") as f:
    config = yaml.safe_load(f)

simulation_params = config["simulation_params"]
protocol_params = config["protocol_params"]
training_params = config["training_params"]
save_dir = os.path.join(config['save_directory'], timestamp)+ '/'
initilize_to_mc = config['initilize_to_mc']


simulation_params['noise_sigma'] = math.sqrt(2 * simulation_params['gamma'] / simulation_params['beta'])  # od Einstein relation
centers = math.sqrt(protocol_params['b_endpoints'][0]/(2*protocol_params['a_endpoints'][0]))
local_var = 1/(4 * protocol_params['b_endpoints'][0] * simulation_params['beta'])

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
    b_list_history = metrics['b_list_history']
    c_list_history = metrics['c_list_history']
    print("Loaded previous training metrics.")
else:
    mean_distance_list = []
    work_list = []
    total_loss_list = []
    #var_list = []
    a_list_history = []
    b_list_history = []
    c_list_history = []
    print("Initialized new training metrics.")
# Load or initialize alist and blist
if os.path.exists(save_dir + 'work_checkpoint.pt'):
    checkpoint = torch.load(save_dir + 'work_checkpoint.pt', weights_only=True)
    
    protocol_params['a_list'] = checkpoint['a_list'].to(torch_device).requires_grad_()
    protocol_params['b_list']= checkpoint['b_list'].to(torch_device).requires_grad_()
    protocol_params['c_list'] = checkpoint['c_list'].to(torch_device).requires_grad_()
    print("Loaded alist, blist and clist from checkpoint.")

    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    optimizer_b = torch.optim.Adam([protocol_params['b_list']], lr= training_params['learning_rate'])
    optimizer_c = torch.optim.Adam([protocol_params['c_list']], lr= training_params['learning_rate'])
    optimizer_a.load_state_dict(checkpoint['optimizer_a_dict'])
    optimizer_b.load_state_dict(checkpoint['optimizer_b_dict'])
    optimizer_c.load_state_dict(checkpoint['optimizer_c_dict'])
    scheduler_a = StepLR(optimizer_a, step_size=50, gamma=0.8)
    scheduler_b = StepLR(optimizer_b, step_size=50, gamma=0.8)
    scheduler_c = StepLR(optimizer_c, step_size=50, gamma=0.8)
    print("Loaded optimizer states.")
else:
    protocol_params['a_list'] = torch.zeros(protocol_params['num_coefficients'], device=torch_device, requires_grad=True)
    protocol_params['b_list'] = torch.zeros(protocol_params['num_coefficients'], device=torch_device, requires_grad=True)
    protocol_params['c_list'] = torch.zeros(protocol_params['num_coefficients'], device=torch_device, requires_grad=True)
    
    if initilize_to_mc:
        analytic_k = torch.pi**2 * simulation_params['mass'] / (simulation_params['dt']*simulation_params['num_steps'])
        protocol_params['b_list'].data[:] = torch.tensor( [-analytic_k / 2 for i in range(protocol_params['num_coefficients'])] )

    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    optimizer_b = torch.optim.Adam([protocol_params['b_list']], lr=training_params['learning_rate'])
    optimizer_c = torch.optim.Adam([protocol_params['c_list']], lr=training_params['learning_rate'])
    scheduler_a = StepLR(optimizer_a, step_size=50, gamma=0.8)
    scheduler_b = StepLR(optimizer_b, step_size=50, gamma=0.8)
    scheduler_c = StepLR(optimizer_c, step_size=50, gamma=0.8)
    print("Initialized new alist, blist, and clist.")



#initialize the GPU tensors for memory management
array_kwargs = {'dtype': torch.float32, 'device': torch_device}
phase_data = torch.empty((training_params['batch_size'], simulation_params['num_steps']+1, 2), **array_kwargs)
noise = torch.empty((training_params['batch_size'], simulation_params['num_steps']), **array_kwargs)
potential_value = torch.empty((training_params['batch_size'], simulation_params['num_steps']+1), **array_kwargs)
potential_value_advance = torch.empty((training_params['batch_size'], simulation_params['num_steps']), **array_kwargs)
drift_protocol_grad = torch.empty((protocol_params['num_coefficients'], training_params['batch_size'], simulation_params['num_steps']+1), **array_kwargs)
distance_sq_current = torch.empty((training_params['batch_size']), **array_kwargs)
#var_current = torch.empty((training_params['batch_size']), **array_kwargs)
potential_value_grad = torch.empty((training_params['batch_size'], simulation_params['num_steps']+1), **array_kwargs)
potential_value_advance_grad =  torch.empty((training_params['batch_size'], simulation_params['num_steps']), **array_kwargs)



for step in range(training_params['training_iterations']):
    optimizer_a.zero_grad()
    optimizer_b.zero_grad()
    optimizer_c.zero_grad() 
    ######

    phase_data_generator = opt.quartic_simulation_with_linear_od(
        num_paths=training_params['batch_size'], 
        params=simulation_params, 
        a_list=protocol_params['a_list'].detach().clone(), 
        b_list=protocol_params['b_list'].detach().clone(), 
        c_list=protocol_params['c_list'].detach().clone(),
        a_endpoints=protocol_params['a_endpoints'], 
        b_endpoints=protocol_params['b_endpoints'],
        c_endpoints=protocol_params['c_endpoints'],
        device=simulation_device)

    phase_data.data, noise.data = [ item.to(torch_device) for item in phase_data_generator.trajectory_generator()]


    '''
    # Check the number of right and left paths
    num_left = torch.sum(phase_data[:, 0, 0] < 0)
    num_right = torch.sum(phase_data[:, 0, 0] > 0)
    # Reorder the paths so that left paths are first
    mask_left = (phase_data[:, 0, 0] < 0)
    mask_right = ~mask_left  # everything else

    phase_data_left = phase_data[mask_left]
    phase_data_right = phase_data[mask_right]
    noise_left = noise[mask_left]
    noise_right = noise[mask_right]
    phase_data_reordered = torch.cat([phase_data_left, phase_data_right], dim=0)
    noise_reordered = torch.cat([noise_left, noise_right], dim=0)
    '''

    p_lists=[protocol_params['a_list'], protocol_params['b_list'], protocol_params['c_list']]
    p_endpoints=[protocol_params['a_endpoints'], protocol_params['b_endpoints'], protocol_params['c_endpoints']]

    #arrays= opt.DerivativeArrays(simulation_params, phase_data, p_lists, p_endpoints)

    biterase = opt.bit_erasure(simulation_params, phase_data, protocol_params['a_list'], protocol_params['b_list'], protocol_params['c_list'], protocol_params['a_endpoints'], protocol_params['b_endpoints'], protocol_params['c_endpoints'])
    grad = opt.grad_calc(simulation_params, protocol_params, device=torch_device)


    # assign the arrays to the GPU tensors
    potential_value.data = biterase.potential_value_array()
    potential_value_advance.data = biterase.potential_value_advance_array()
    distance_sq_current.data = biterase.distance_sq_current(phase_data, p_lists, target=centers, order=2)
    #var_current.data = biterase.var_current(phase_data, p_lists)

    for list_name, drift_gradient_func, potential_function, potential_advance_function in zip( ['a_list', 'b_list', 'c_list'],
                                               [biterase.drift_grad_a_array, biterase.drift_grad_b_array, biterase.drift_grad_c_array],
                                               [biterase.potential_grad_a_value_array, biterase.potential_grad_b_value_array, biterase.potential_grad_c_value_array],
                                               [biterase.potential_grad_a_value_advance_array, biterase.potential_grad_b_value_advance_array, biterase.potential_grad_c_value_advance_array]):

        # assign the arrays to the GPU tensors
        drift_protocol_grad.data = grad.drift_protocol_grad_array(drift_gradient_func())
        potential_value_grad.data = potential_function()
        potential_value_advance_grad.data = potential_advance_function()
        
        # Calculate the gradients
        work_grad = grad.work_protocol_grad(potential_value, potential_value_advance, potential_value_grad, potential_value_advance_grad, drift_protocol_grad, noise)
        distance_grad = grad.current_grad(distance_sq_current, drift_protocol_grad, noise)
        #var_grad = grad.current_grad(var_current, drift_protocol_grad, noise)
        param_grad = training_params['alpha_2'] * work_grad + training_params['alpha'] * distance_grad #+ training_params['alpha_1'] * var_grad
        protocol_params[list_name].grad = param_grad


    distance_loss = distance_sq_current.mean()
    work_mean = grad.work_array(potential_value, potential_value_advance).sum(axis=-1).mean()
    #var_loss = var_current.mean()
    total_loss = training_params['alpha'] * distance_loss + training_params['alpha_2'] * work_mean #+ training_params['alpha_1'] * var_loss
    mean_distance_list.append(distance_loss.item())
    #var_list.append(var_loss.item())
    work_list.append(work_mean.item())
    total_loss_list.append(total_loss.item())

    a_list_history.append(protocol_params['a_list'].detach().cpu().clone().tolist())
    b_list_history.append(protocol_params['b_list'].detach().cpu().clone().tolist())
    c_list_history.append(protocol_params['c_list'].detach().cpu().clone().tolist())
    
    optimizer_a.step()
    optimizer_b.step()
    optimizer_c.step()

    scheduler_a.step()
    scheduler_b.step()
    scheduler_c.step()

    total_iterations = len(mean_distance_list)
    if step % config['report_frequency'] == 0:
        print(f"Step {step}: a = {protocol_params['a_list']}, b = {protocol_params['b_list']}, c = {protocol_params['c_list']}, x = {distance_loss}, work = {work_mean}, total loss = {total_loss}")

        for param_group in optimizer_a.param_groups:
            print("Current LR (a):", param_group['lr'])
        for param_group in optimizer_b.param_groups:
            print("Current LR (b):", param_group['lr'])
        for param_group in optimizer_c.param_groups:
            print("Current LR (c):", param_group['lr'])
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
                'b_list_history': b_list_history,
                'c_list_history': c_list_history
            }
        with open(save_dir + 'training_metrics.json', 'w') as f:
            json.dump(metrics, f)   


        torch.save({
            'a_list': protocol_params['a_list'],
            'b_list': protocol_params['b_list'],
            'c_list': protocol_params['c_list'],
            'optimizer_b_dict': optimizer_b.state_dict(),
            'optimizer_a_dict': optimizer_a.state_dict(),
            'optimizer_c_dict': optimizer_c.state_dict(),
        }, save_dir + 'work_checkpoint.pt') 

        shutil.copyfile(args.config, save_dir + 'config.yaml')
        print("Checkpoint saved.")

        if plot_boolean:
            print(f"Plotting checkpoint at step {step}, total_iter {total_iterations}")
            res = subprocess.call(['python', 'erase_plot.py', save_dir])
            print("Checkpoint plotted.")