import torch
import gc
import os
import json
import numpy as np
import bit_flip
import yaml
import math


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

simulation_params = config["simulation_params"]
protocol_params = config["protocol_params"]
training_params = config["training_params"]

simulation_params['noise_sigma'] = math.sqrt(2 * simulation_params['gamma'] * simulation_params['kBT'])


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if os.path.exists('training_metrics.json'):
    with open( 'training_metrics.json', 'r') as f:
        metrics = json.load(f)
    mean_distance_list = metrics['mean_distance_list']
    var_distance_list = metrics['var_distance_list']
    work_list = metrics['work_list']
    print("Loaded previous training metrics.")
else:
    mean_distance_list = []
    var_distance_list = []
    work_list = []
    print("Initialized new training metrics.")
# Load or initialize alist and blist
if os.path.exists('work_checkpoint.pt'):
    checkpoint = torch.load('work_checkpoint.pt', weights_only=True)
    
    protocol_params['a_list'] = checkpoint['a_list'].to(torch_device).requires_grad_()
    protocol_params['b_list']= checkpoint['b_list'].to(torch_device).requires_grad_()
    print("Loaded alist and blist from checkpoint.")

    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    optimizer_b = torch.optim.Adam([protocol_params['b_list']], lr= training_params['learning_rate'])
    optimizer_a.load_state_dict(checkpoint['optimizer_a_dict'])
    optimizer_b.load_state_dict(checkpoint['optimizer_b_dict'])
    print("Loaded optimizer states.")
else:
    protocol_params['a_list'] = torch.zeros(protocol_params['num_coefficient'], device=torch_device, requires_grad=True)
    protocol_params['b_list'] = torch.zeros(protocol_params['num_coefficient'], device=torch_device, requires_grad=True)
    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    optimizer_b = torch.optim.Adam([protocol_params['b_list']], lr=training_params['learning_rate'])
    print("Initialized new alist and blist.")
######## Initialize the parameters for the protocol



# the total loss is the sum of alpha* mean loss + alpha1 * variance loss + alpha2 * work loss


for step in range(training_params['training_iterations']):
    optimizer_a.zero_grad()
    optimizer_b.zero_grad() 
    ######
 
    phase_data_generator = bit_flip.bit_flip_simulation(num_paths=5000, params=simulation_params, a_list=protocol_params['a_list'], b_list=protocol_params['b_list'], a_endpoints=protocol_params['a_endpoints'], b_endpoints=protocol_params['b_list'])
    left_phase_data, left_noise=phase_data_generator.left_trajectory_generator()
    right_phase_data, right_noise = phase_data_generator.right_trajectory_generator()

    grad_left =  bit_flip.bit_flip_gradient_tensor_comp(simulation_params, left_phase_data, left_noise, a_list=protocol_params['a_list'], b_list=protocol_params['b_list'], a_endpoints=protocol_params['a_endpoints'], b_endpoints=protocol_params['b_endpoints'])
    grad_right =  bit_flip.bit_flip_gradient_tensor_comp(simulation_params, right_phase_data, right_noise, a_list=protocol_params['a_list'], b_list=protocol_params['b_list'], a_endpoints=protocol_params['a_endpoints'], b_endpoints=protocol_params['b_endpoints'])

    cntr = simulation_params['center']

    mean_grad_a_left =  2*(left_phase_data[:,-1,0].mean() - cntr) * grad_left.x_mean_grad(grad_left.drift_grad_a_array())
    mean_grad_a_right = 2*(right_phase_data[:,-1,0].mean() + cntr) * grad_right.x_mean_grad(grad_right.drift_grad_a_array())

    mean_grad_b_left = 2*(left_phase_data[:,-1,0].mean() - cntr) * grad_left.x_mean_grad(grad_left.drift_grad_b_array())
    mean_grad_b_right = 2*(right_phase_data[:,-1,0].mean() + cntr) * grad_right.x_mean_grad(grad_right.drift_grad_b_array())

    var_grad_a_left = 2*(left_phase_data[:,-1,0].var() - 0.025) * grad_left.x_var_grad(grad_left.drift_grad_a_array())
    var_grad_a_right = 2*(right_phase_data[:,-1,0].var() - 0.025) * grad_right.x_var_grad(grad_right.drift_grad_a_array())

    var_grad_b_left = 2*(left_phase_data[:,-1,0].var() - 0.025) * grad_left.x_var_grad(grad_left.drift_grad_b_array())
    var_grad_b_right = 2*(right_phase_data[:,-1,0].var() - 0.025) * grad_right.x_var_grad(grad_right.drift_grad_b_array())

    work_grad_a_left = grad_left.work_grad(grad_left.potential_grad_a_value_array(), grad_left.potential_grad_a_value_advance_array(), grad_left.drift_grad_a_array())
    work_grad_a_right = grad_right.work_grad(grad_right.potential_grad_a_value_array(), grad_right.potential_grad_a_value_advance_array(), grad_right.drift_grad_a_array())
        
    work_grad_b_left = grad_left.work_grad(grad_left.potential_grad_b_value_array(), grad_left.potential_grad_b_value_advance_array(), grad_left.drift_grad_b_array())
    work_grad_b_right = grad_right.work_grad(grad_right.potential_grad_b_value_array(), grad_right.potential_grad_b_value_advance_array(), grad_right.drift_grad_b_array())
        
    protocol_params['a_list'].grad = training_params['alpha'] * (mean_grad_a_left + mean_grad_a_right) + training_params['alpha_1'] * (var_grad_a_left + var_grad_a_right) + training_params['alpha_2'] * (work_grad_a_left + work_grad_a_right)
    protocol_params['b_list'].grad = training_params['alpha'] * (mean_grad_b_left + mean_grad_b_right) + training_params['alpha_1'] * (var_grad_b_left + var_grad_b_right) + training_params['alpha_2'] * (work_grad_b_left + work_grad_b_right)
    
    optimizer_a.step()
    optimizer_b.step()


    x_loss = torch.abs((left_phase_data[:, -1,0].mean() - cntr))/1 + torch.abs((right_phase_data[:, -1,0].mean() + cntr))/1
    var_loss = torch.abs((left_phase_data[:, -1,0].var() - 0.025))/0.025 + torch.abs((right_phase_data[:, -1,0].var() - 0.025))/0.025
    work = (grad_left.work_array().sum(axis=1).mean() + grad_right.work_array().sum(axis=1).mean())/2
    mean_distance_list.append(x_loss.item())
    var_distance_list.append(var_loss.item())
    work_list.append(work.item())
   
   
    if step % 10 == 0:
        print(f"Step {step}: a = {protocol_params['a_list']}, b = {protocol_params['b_list']}, relative mean distance = {x_loss}, relative variance distance = {var_loss}, work = {work}")
    

 # Save the lists after training using JSON

metrics = {
        'mean_distance_list': mean_distance_list,
        'var_distance_list': var_distance_list,
        'work_list': work_list
    }
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f)


torch.save({
    'a_list': protocol_params['a_list'].detach().cpu(),
    'b_list': protocol_params['b_list'].detach().cpu(),
    'optimizer_a_dict': optimizer_a.state_dict(),
    'optimizer_b_dict': optimizer_b.state_dict()
}, 'work_checkpoint.pt')
print("Checkpoint saved. Please restart kernel before next run.")
