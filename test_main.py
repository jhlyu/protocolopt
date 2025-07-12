import torch
import gc
import os
import json
import numpy as np
import opt
import yaml
import math



with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

simulation_params = config["simulation_params"]
protocol_params = config["protocol_params"]
training_params = config["training_params"]

simulation_params['noise_sigma'] = math.sqrt(2 * simulation_params['gamma'] / simulation_params['beta'])  # od Einstein relation
centers = math.sqrt(protocol_params['b_endpoints'][0]/(2*protocol_params['a_endpoints'][0]))
local_var = 1/(4 * protocol_params['b_endpoints'][0] * simulation_params['beta'])


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if os.path.exists('training_metrics.json'):
    with open( 'training_metrics.json', 'r') as f:
        metrics = json.load(f)
    mean_distance_list = metrics['mean_distance_list']
    var_distance_list = metrics['var_distance_list']
    work_list = metrics['work_list']
    total_loss_list = metrics['total_loss_list']
    a_list_history = metrics['a_list_history']
    b_list_history = metrics['b_list_history']
    print("Loaded previous training metrics.")
else:
    mean_distance_list = []
    var_distance_list = []
    work_list = []
    total_loss_list = []
    a_list_history = []
    b_list_history = []
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

for step in range(training_params['training_iterations']):
    optimizer_a.zero_grad()
    optimizer_b.zero_grad() 
    ######

    phase_data_generator = opt.quartic_simulation(num_paths=training_params['batch_size'], params=simulation_params, a_list=protocol_params['a_list'], b_list=protocol_params['b_list'], a_endpoints=protocol_params['a_endpoints'], b_endpoints=protocol_params['b_endpoints'])
    phase_data, noise=phase_data_generator.trajectory_generator()
    left_phase_data, left_noise = phase_data[phase_data[:, 0, 0] < 0], noise[phase_data[:, 0, 0] < 0]
    right_phase_data, right_noise = phase_data[phase_data[:, 0, 0] > 0], noise[phase_data[:, 0, 0] > 0]

    tensor = opt.bit_flip(
        params=simulation_params, 
        phase_data=left_phase_data,
        a_list=protocol_params['a_list'], 
        b_list=protocol_params['b_list'], 
        a_endpoints=protocol_params['a_endpoints'], 
        b_endpoints=protocol_params['b_endpoints'])

    left_potential_advance = tensor.potential_value_advance_array()
    left_potential = tensor.potential_value_array()
    left_drift_a_grad = tensor.drift_grad_a_array()
    left_drift_b_grad = tensor.drift_grad_b_array()
    left_potential_a_grad = tensor.potential_grad_a_value_array()
    left_potential_b_grad = tensor.potential_grad_b_value_array()
    left_potential_a_advance_grad = tensor.potential_grad_a_value_advance_array()
    left_potential_b_advance_grad = tensor.potential_grad_b_value_advance_array()
    



    grad = opt.grad_calc(
        sim_params=simulation_params,
        protocol_params=protocol_params['a_list'],
        noise_array=left_noise,
        potential_array=left_potential,
        potential_advance_array=left_potential_advance,
        potential_grad_array=left_potential_a_grad,
        potential_grad_advance_array=left_potential_a_advance_grad,
        drift_grad_array=left_drift_a_grad
    )
    left_work = grad.work_array().sum(axis=1).mean() 
    # Set current for gradient calculation
    left_x_last = left_phase_data[:,-1,0]
    left_x_last_square = left_x_last**2
    left_x_var_last = (left_phase_data[:,-1,0] - left_x_last.mean())**2
    
    # compute gradients for a protocol
    work_grad_a_left = grad.work_grad()
    mean_grad_a_left = grad.current_grad(torch.abs(left_x_last-centers))
    var_grad_a_left = grad.current_grad(left_x_last_square) - 2 * left_x_last.mean() * grad.current_grad(left_x_last)
    # switch to b protocol
    grad.protocol_params = protocol_params['b_list']
    grad.potential_grad_array = left_potential_b_grad
    grad.potential_grad_advance_array = left_potential_b_advance_grad
    grad.drift_grad_array = left_drift_b_grad

    # compute gradients for b protocol
    work_grad_b_left = grad.work_grad()
    mean_grad_b_left = grad.current_grad(torch.abs(left_x_last-centers))
    var_grad_b_left = grad.current_grad(left_x_last_square) - 2 * left_x_last.mean() * grad.current_grad(left_x_last)
    

    # will do the same for right side
    tensor.phase_data = right_phase_data
    
    
    right_potential_advance = tensor.potential_value_advance_array()
    right_potential = tensor.potential_value_array()
    right_drift_a_grad = tensor.drift_grad_a_array()
    right_drift_b_grad = tensor.drift_grad_b_array()
    right_potential_a_grad = tensor.potential_grad_a_value_array()
    right_potential_b_grad = tensor.potential_grad_b_value_array()
    right_potential_a_advance_grad = tensor.potential_grad_a_value_advance_array()
    right_potential_b_advance_grad = tensor.potential_grad_b_value_advance_array()

    grad = opt.grad_calc(
        sim_params=simulation_params,
        noise_array=right_noise,
        protocol_params=protocol_params['a_list'],
        potential_array=right_potential,
        potential_advance_array=right_potential_advance,
        potential_grad_array=right_potential_a_grad,
        potential_grad_advance_array=right_potential_a_advance_grad,
        drift_grad_array=right_drift_a_grad
    )
    right_work = grad.work_array().sum(axis=1).mean()

    # Set current for gradient calculation
    right_x_last = right_phase_data[:,-1,0]
    right_x_var_last = (right_phase_data[:,-1,0] - right_x_last.mean())**2
    right_x_last_square = right_x_last**2
   
    # compute gradients for a protocol
    work_grad_a_right = grad.work_grad()
    mean_grad_a_right = grad.current_grad(torch.abs(right_x_last+centers))
    var_grad_a_right = grad.current_grad(right_x_last_square) - 2 * right_x_last.mean() * grad.current_grad(right_x_last)

    # switch to b protocol
    grad.protocol_params = protocol_params['b_list']
    grad.potential_grad_array = right_potential_b_grad
    grad.potential_grad_advance_array = right_potential_b_advance_grad
    grad.drift_grad_array = right_drift_b_grad

    # compute gradients for b protocol
    work_grad_b_right = grad.work_grad()
    mean_grad_b_right = grad.current_grad(torch.abs(right_x_last+centers))
    var_grad_b_right = grad.current_grad(right_x_last_square) - 2 * right_x_last.mean() * grad.current_grad(right_x_last)

    a_grad = training_params['alpha'] * (mean_grad_a_left + mean_grad_a_right) + training_params['alpha_1'] * (var_grad_a_left + var_grad_a_right) + training_params['alpha_2'] * (work_grad_a_left + work_grad_a_right) 
    b_grad = training_params['alpha'] * (mean_grad_b_left + mean_grad_b_right) + training_params['alpha_1'] * (var_grad_b_left + var_grad_b_right) + training_params['alpha_2'] * (work_grad_b_left + work_grad_b_right)

    protocol_params['a_list'].grad = a_grad
    protocol_params['b_list'].grad = b_grad

   
    x_loss = (left_x_last.mean() - centers)**2 + (right_x_last.mean() + centers)**2
    var_loss = (left_x_var_last.mean() - local_var)**2 + (right_x_var_last.mean() - local_var)**2
    work = left_work + right_work
    total_loss = training_params['alpha'] * x_loss + training_params['alpha_1'] * var_loss + training_params['alpha_2'] * work
    mean_distance_list.append(x_loss.item())
    var_distance_list.append(var_loss.item())
    work_list.append(work.item())
    total_loss_list.append(total_loss.item())
    a_list_history.append(protocol_params['a_list'].detach().cpu().clone().tolist())
    b_list_history.append(protocol_params['b_list'].detach().cpu().clone().tolist())

    optimizer_a.step()
    optimizer_b.step()

    if step % 10 == 0:
        print(f"Step {step}: a = {protocol_params['a_list']}, b = {protocol_params['b_list']}, x = {x_loss}, work = {work}, var = {var_loss}, total loss = {total_loss}")


 # Save the lists after training using JSON

metrics = {
        'mean_distance_list': mean_distance_list,
        'var_distance_list': var_distance_list,
        'work_list': work_list,
        'total_loss_list': total_loss_list,
        'a_list_history': a_list_history,
        'b_list_history': b_list_history
    }
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f)


torch.save({
    'a_list': protocol_params['a_list'],
    'b_list': protocol_params['b_list'],
    'optimizer_b_dict': optimizer_b.state_dict(),
    'optimizer_a_dict': optimizer_a.state_dict(),
}, 'work_checkpoint.pt')
print("Checkpoint saved. Please restart kernel before next run.")
