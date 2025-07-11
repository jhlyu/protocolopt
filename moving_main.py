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

simulation_params['noise_sigma'] = math.sqrt(2 * simulation_params['kBT']) # od Einstein relation


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if os.path.exists('training_metrics.json'):
    with open( 'training_metrics.json', 'r') as f:
        metrics = json.load(f)
    mean_distance_list = metrics['mean_distance_list']
    work_list = metrics['work_list']
    total_loss_list = metrics['total_loss_list']
    print("Loaded previous training metrics.")
else:
    mean_distance_list = []
    work_list = []
    total_loss_list = []
    print("Initialized new training metrics.")
# Load or initialize alist and blist
if os.path.exists('work_checkpoint.pt'):
    checkpoint = torch.load('work_checkpoint.pt', weights_only=True)
    
    protocol_params['a_list'] = checkpoint['a_list'].to(torch_device).requires_grad_()
    print("Loaded alist and blist from checkpoint.")

    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    optimizer_a.load_state_dict(checkpoint['optimizer_a_dict'])
    print("Loaded optimizer states.")
else:
    protocol_params['a_list'] = torch.zeros(protocol_params['num_coefficient'], device=torch_device, requires_grad=True)
    optimizer_a = torch.optim.Adam([protocol_params['a_list']], lr=training_params['learning_rate'])
    print("Initialized new alist.")
######## Initialize the parameters for the protocol



# the total loss is the sum of alpha* mean loss + alpha1 * variance loss + alpha2 * work loss


for step in range(training_params['training_iterations']):
    optimizer_a.zero_grad()
    ######
 
    phase_data_generator = bit_flip.od_moving_simulation_jump(num_paths=training_params['batch_size'], params=simulation_params, a_list=protocol_params['a_list'], a_endpoints=protocol_params['a_endpoints'])
    phase_data, noise=phase_data_generator.trajectory_generator(phase_data_generator.initial_sample_distribution)

    grad = bit_flip.od_moving_gradient_tensor_comp_jump(params=simulation_params, phase_data=phase_data, noise=noise, a_list=protocol_params['a_list'], a_endpoints=protocol_params['a_endpoints'])

    work_grad = grad.work_grad(grad.potential_grad_a_value_array(), grad.potential_grad_a_value_advance_array(), grad.drift_grad_a_array())
    mean_grad = 2*grad.x_mean_grad(grad.drift_grad_a_array()) *(phase_data[:,-1].mean() - protocol_params['a_endpoints'][-1])
    protocol_params['a_list'].grad = training_params['alpha'] * mean_grad  + training_params['alpha_2'] * work_grad

    optimizer_a.step()

    x_loss = ((phase_data[:, -1].mean() -  protocol_params['a_endpoints'][-1]))**2
    work = grad.work_array().sum(axis=1).mean() 
    total_loss = training_params['alpha'] * x_loss + training_params['alpha_2'] * work
    mean_distance_list.append(x_loss.item())
    work_list.append(work.item())
    total_loss_list.append(total_loss.item())

    if step % 10 == 0:
        print(f"Step {step}: a = {protocol_params['a_list']}, x distance = {x_loss}, work = {work}, total loss = {total_loss}")
    

 # Save the lists after training using JSON

metrics = {
        'mean_distance_list': mean_distance_list,
        'work_list': work_list,
        'total_loss_list': total_loss_list
    }
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f)


torch.save({
    'a_list': protocol_params['a_list'],
    'optimizer_a_dict': optimizer_a.state_dict(),
}, 'work_checkpoint.pt')
print("Checkpoint saved. Please restart kernel before next run.")
