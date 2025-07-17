import torch
import gc
import os
import json
import numpy as np
import opt_kyle as opt
import yaml
import math



with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

simulation_params = config["simulation_params"]
protocol_params = config["protocol_params"]
training_params = config["training_params"]
save_dir = config['save_directory'] 


simulation_params['noise_sigma'] = math.sqrt(2 * simulation_params['gamma'] / simulation_params['beta'])  # od Einstein relation
centers = math.sqrt(protocol_params['b_endpoints'][0]/(2*protocol_params['a_endpoints'][0]))
local_var = 1/(4 * protocol_params['b_endpoints'][0] * simulation_params['beta'])

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if os.path.exists(save_dir + 'training_metrics.json'):
    with open(save_dir + 'training_metrics.json', 'r') as f:
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
if os.path.exists(save_dir + 'work_checkpoint.pt'):
    checkpoint = torch.load(save_dir + 'work_checkpoint.pt', weights_only=True)
    
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

    phase_data_generator = opt.quartic_simulation(
        num_paths=training_params['batch_size'], 
        params=simulation_params, 
        a_list=protocol_params['a_list'], 
        b_list=protocol_params['b_list'], 
        a_endpoints=protocol_params['a_endpoints'], 
        b_endpoints=protocol_params['b_endpoints'])
    
    phase_data, noise=phase_data_generator.trajectory_generator()
    
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

    distance_loss = ((phase_data_reordered[:num_left, -1, 0] - centers)**2).mean() +((phase_data_reordered[num_left:, -1, 0] + centers)**2).mean()
    var_loss = ((phase_data_reordered[:num_left, -1, 0] - centers)**2).var() + ((phase_data_reordered[num_left:, -1, 0] + centers)**2).var()
     

    tensor = opt.bit_flip(
        params=simulation_params, 
        phase_data=phase_data_reordered,
        a_list=protocol_params['a_list'], 
        b_list=protocol_params['b_list'], 
        a_endpoints=protocol_params['a_endpoints'], 
        b_endpoints=protocol_params['b_endpoints'])

    potential_advance = tensor.potential_value_advance_array()
    potential = tensor.potential_value_array()
    drift_a_grad = tensor.drift_grad_a_array()
    drift_b_grad = tensor.drift_grad_b_array()
    potential_a_grad = tensor.potential_grad_a_value_array()
    potential_b_grad = tensor.potential_grad_b_value_array()
    potential_a_advance_grad = tensor.potential_grad_a_value_advance_array()
    potential_b_advance_grad = tensor.potential_grad_b_value_advance_array()


    grad = opt.grad_calc(
        sim_params=simulation_params,
        protocol_params=protocol_params['a_list'],
        noise_array=noise_reordered,
        potential_array=potential,
        potential_advance_array=potential_advance,
        potential_grad_array=potential_a_grad,
        potential_grad_advance_array=potential_a_advance_grad,
        drift_grad_array=drift_a_grad
    )
    work = grad.work_array().sum(axis=-1)
    work_mean = work.mean()
    left_work = work[:num_left].mean()
    right_work = work[num_left:].mean()
    # The loss function gradients is dL = dW + d(x+centers)^2|_left + d(x-centers)^2|_right + d var|_left + d var|_right
    work_grad = grad.work_grad()

    last_position_distance = torch.zeros(phase_data_reordered.shape[0], device=torch_device)
    last_position_distance[:num_left] = (phase_data_reordered[:num_left, -1, 0] - centers)**2
    last_position_distance[num_left:] = (phase_data_reordered[num_left:, -1, 0] + centers)**2
    last_position = phase_data_reordered[:, -1, 0]
    last_position_sq = last_position**2


    x_grad_array = grad.current_grad_without_mean(last_position_distance)
    sq_grad_array = grad.current_grad_without_mean(last_position_sq)
    # compute the distance grad = <d(X_{L}-center)**2> + <d(X_{R}+center)**2> or d<(x-target)^2>
    #x_grad = (x_grad_array[:,:num_left]).mean(axis=-1) + (x_grad_array[:,num_left:]).mean(axis=-1)
    x_grad = x_grad_array.mean(axis=-1)
    # compute the variance grad = d <X_{L}**2> - 2 d <X_{L}>* <X_{L}>
    var_grad = (sq_grad_array[:,:num_left]).mean(axis=-1) + (sq_grad_array[:,num_left:]).mean(axis=-1) - 2 * (last_position[:num_left].mean(axis=-1) * x_grad_array[:,:num_left].mean(axis=-1)) - 2 * (x_grad_array[:,num_left:].mean(axis=-1) * last_position[num_left:].mean(axis=-1))

    a_grad = training_params['alpha'] * x_grad + training_params['alpha_1'] * var_grad + training_params['alpha_2'] * work_grad
    protocol_params['a_list'].grad = a_grad
    
    grad = opt.grad_calc(
        sim_params=simulation_params,
        protocol_params=protocol_params['b_list'],
        noise_array=noise_reordered,
        potential_array=potential,
        potential_advance_array=potential_advance,
        potential_grad_array=potential_b_grad,
        potential_grad_advance_array=potential_b_advance_grad,
        drift_grad_array=drift_b_grad
    )

    x_grad_array = grad.current_grad_without_mean(last_position_distance)
    sq_grad_array = grad.current_grad_without_mean(last_position_sq)
    # compute the distance grad = <d(X_{L}-center)**2> + <d(X_{R}+center)**2> 
    x_grad = (x_grad_array[:,:num_left]).mean(axis=-1) + (x_grad_array[:,num_left:]).mean(axis=-1)
    
    # compute the variance grad = d <X_{L}**2> - 2 d <X_{L}>* <X_{L}>
    var_grad = (sq_grad_array[:,:num_left]).mean(axis=-1) + (sq_grad_array[:,num_left:]).mean(axis=-1) - 2 * (last_position[:num_left].mean(axis=-1) * x_grad_array[:,:num_left].mean(axis=-1)) - 2 * (x_grad_array[:,num_left:].mean(axis=-1) * last_position[num_left:].mean(axis=-1))

    b_grad = training_params['alpha'] * x_grad + training_params['alpha_1'] * var_grad + training_params['alpha_2'] * work_grad
    protocol_params['b_list'].grad = b_grad


    protocol_params['a_list'].grad = a_grad
    protocol_params['b_list'].grad = b_grad

    
    total_loss = training_params['alpha'] * distance_loss + training_params['alpha_1'] * var_loss + training_params['alpha_2'] * work_mean
    mean_distance_list.append(distance_loss.item())
    var_distance_list.append(var_loss.item())
    work_list.append(work_mean.item())
    total_loss_list.append(total_loss.item())
    a_list_history.append(protocol_params['a_list'].detach().cpu().clone().tolist())
    b_list_history.append(protocol_params['b_list'].detach().cpu().clone().tolist())

    optimizer_a.step()
    optimizer_b.step()

    if step % 10 == 0:
        print(f"Step {step}: a = {protocol_params['a_list']}, b = {protocol_params['b_list']}, x = {distance_loss}, work = {work_mean}, var = {var_loss}, total loss = {total_loss}")


 # Save the lists after training using JSON

metrics = {
        'mean_distance_list': mean_distance_list,
        'var_distance_list': var_distance_list,
        'work_list': work_list,
        'total_loss_list': total_loss_list,
        'a_list_history': a_list_history,
        'b_list_history': b_list_history
    }
with open(save_dir + 'training_metrics.json', 'w') as f:
    json.dump(metrics, f)


torch.save({
    'a_list': protocol_params['a_list'],
    'b_list': protocol_params['b_list'],
    'optimizer_b_dict': optimizer_b.state_dict(),
    'optimizer_a_dict': optimizer_a.state_dict(),
}, save_dir + 'work_checkpoint.pt')
print("Checkpoint saved. Please restart kernel before next run.")
