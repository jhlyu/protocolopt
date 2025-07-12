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

simulation_params['noise_sigma'] = math.sqrt(2 * simulation_params['gamma'] / simulation_params['beta'])  # od Einstein relation
centers = math.sqrt(protocol_params['b_endpoints'][0]/(2*protocol_params['a_endpoints'][0]))
#local_var = 1/(4 * protocol_params['b_endpoints'][0] * simulation_params['beta'])


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if os.path.exists('training_metrics.json'):
    with open( 'training_metrics.json', 'r') as f:
        metrics = json.load(f)
    mean_distance_list = metrics['mean_distance_list']
    #var_distance_list = metrics['var_distance_list']
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
    a_list_history = []
    b_list_history = []
    c_list_history = []
    print("Initialized new training metrics.")
# Load or initialize alist and blist
if os.path.exists('work_checkpoint.pt'):
    checkpoint = torch.load('work_checkpoint.pt', weights_only=True)
    
    protocol_params['a_list'] = checkpoint['a_list'].to(torch_device).requires_grad_()
    protocol_params['b_list']= checkpoint['b_list'].to(torch_device).requires_grad_()
    protocol_params['c_list'] = checkpoint['c_list'].to(torch_device).requires_grad_()
    print("Loaded alist and blist from checkpoint.")

    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    optimizer_b = torch.optim.Adam([protocol_params['b_list']], lr= training_params['learning_rate'])
    optimizer_c = torch.optim.Adam([protocol_params['c_list']], lr= training_params['learning_rate'])
    optimizer_a.load_state_dict(checkpoint['optimizer_a_dict'])
    optimizer_b.load_state_dict(checkpoint['optimizer_b_dict'])
    optimizer_c.load_state_dict(checkpoint['optimizer_c_dict'])
    print("Loaded optimizer states.")
else:
    protocol_params['a_list'] = torch.zeros(protocol_params['num_coefficient'], device=torch_device, requires_grad=True)
    protocol_params['b_list'] = torch.zeros(protocol_params['num_coefficient'], device=torch_device, requires_grad=True)
    protocol_params['c_list'] = torch.zeros(protocol_params['num_coefficient'], device=torch_device, requires_grad=True)
    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    optimizer_b = torch.optim.Adam([protocol_params['b_list']], lr=training_params['learning_rate'])
    optimizer_c = torch.optim.Adam([protocol_params['c_list']], lr=training_params['learning_rate'])
    print("Initialized new alist, blist, and clist.")

for step in range(training_params['training_iterations']):
    optimizer_a.zero_grad()
    optimizer_b.zero_grad()
    optimizer_c.zero_grad()
    ######

    phase_data_generator = bit_flip.quartic_simulation(
        num_paths=training_params['batch_size'], 
        params=simulation_params, 
        a_list=protocol_params['a_list'], 
        b_list=protocol_params['b_list'], 
        a_endpoints=protocol_params['a_endpoints'], 
        b_endpoints=protocol_params['b_endpoints'])
    phase_data, noise=phase_data_generator.trajectory_generator()

    tensor = bit_flip.bit_erasure(
        params=simulation_params, 
        phase_data=phase_data,
        a_list=protocol_params['a_list'], 
        b_list=protocol_params['b_list'],
        c_list=protocol_params['c_list'], 
        a_endpoints=protocol_params['a_endpoints'], 
        b_endpoints=protocol_params['b_endpoints'],
        c_endpoints=protocol_params['c_endpoints'])

    potential_advance = tensor.potential_value_advance_array()
    potential = tensor.potential_value_array()
    drift_a_grad = tensor.drift_grad_a_array()
    drift_b_grad = tensor.drift_grad_b_array()
    drift_c_grad = tensor.drift_grad_c_array()
    potential_a_grad = tensor.potential_grad_a_value_array()
    potential_b_grad = tensor.potential_grad_b_value_array()
    potential_c_grad = tensor.potential_grad_c_value_array()
    potential_a_advance_grad = tensor.potential_grad_a_value_advance_array()
    potential_b_advance_grad = tensor.potential_grad_b_value_advance_array()
    potential_c_advance_grad = tensor.potential_grad_c_value_advance_array()


 
    
    grad = bit_flip.grad_calc(
        sim_params=simulation_params,
        protocol_params=protocol_params['a_list'],
        noise_array=noise,
        potential_array=potential,
        potential_advance_array=potential_advance,
        potential_grad_array=potential_a_grad,
        potential_grad_advance_array=potential_a_advance_grad,
        drift_grad_array=drift_a_grad
    )
    work = grad.work_array().sum(axis=1).mean() 
    # Set current for gradient calculation
    x_last = phase_data[:,-1,0]
    
    # compute gradients for a protocol
    work_grad_a = grad.work_grad()
    mean_grad_a = grad.current_grad(x_last)
    distance_grad_a = 2 * mean_grad_a * (x_last.mean()-centers)
    
    # switch to b protocol
    grad.protocol_params = protocol_params['b_list']
    grad.potential_grad_array = potential_b_grad
    grad.potential_grad_advance_array = potential_b_advance_grad
    grad.drift_grad_array = drift_b_grad

    # compute gradients for b protocol
    work_grad_b = grad.work_grad()
    mean_grad_b = grad.current_grad(x_last)
    distance_grad_b = 2 * mean_grad_b * (x_last.mean()-centers)
  
    # switch to c protocol
    grad.protocol_params = protocol_params['c_list']
    grad.potential_grad_array = potential_c_grad
    grad.potential_grad_advance_array = potential_c_advance_grad
    grad.drift_grad_array = drift_c_grad

    # compute gradients for c protocol
    work_grad_c = grad.work_grad()
    mean_grad_c = grad.current_grad(x_last)
    distance_grad_c = 2 * mean_grad_c * (x_last.mean()-centers)


    a_grad = training_params['alpha'] * (distance_grad_a) + training_params['alpha_2'] * (work_grad_a)
    b_grad = training_params['alpha'] * (distance_grad_b) + training_params['alpha_2'] * (work_grad_b)
    c_grad = training_params['alpha'] * (distance_grad_c) + training_params['alpha_2'] * (work_grad_c)

    protocol_params['a_list'].grad = a_grad
    protocol_params['b_list'].grad = b_grad
    protocol_params['c_list'].grad = c_grad

    x_loss = x_last.mean()


    total_loss = training_params['alpha'] * x_loss + training_params['alpha_2'] * work
    mean_distance_list.append(x_loss.item())
    work_list.append(work.item())
    total_loss_list.append(total_loss.item())
    a_list_history.append(protocol_params['a_list'].detach().cpu().clone().tolist())
    b_list_history.append(protocol_params['b_list'].detach().cpu().clone().tolist())
    c_list_history.append(protocol_params['c_list'].detach().cpu().clone().tolist())

    optimizer_a.step()
    optimizer_b.step()
    optimizer_c.step()

    if step % 10 == 0:
        print(f"Step {step}: a = {protocol_params['a_list']}, b = {protocol_params['b_list']}, c = {protocol_params['c_list']}, x = {x_loss}, work = {work}, total loss = {total_loss}")


 # Save the lists after training using JSON

metrics = {
        'mean_distance_list': mean_distance_list,
        'work_list': work_list,
        'total_loss_list': total_loss_list,
        'a_list_history': a_list_history,
        'b_list_history': b_list_history,
        'c_list_history': c_list_history
    }
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f)


torch.save({
    'a_list': protocol_params['a_list'],
    'b_list': protocol_params['b_list'],
    'c_list': protocol_params['c_list'],
    'optimizer_b_dict': optimizer_b.state_dict(),
    'optimizer_a_dict': optimizer_a.state_dict(),
    'optimizer_c_dict': optimizer_c.state_dict(),
}, 'work_checkpoint.pt')
print("Checkpoint saved. Please restart kernel before next run.")
