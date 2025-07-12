import bit_flip
class bit_flip_loss_calc:
    ### THe potential is V(x) = a * x^4 - b * x^2 with flipping bits
    def __init__(self, left_phase_data, right_phase_data, left_noise, right_noise, simulation_params, protocol_params, training_params):
        self.training_params = training_params
        self.left_phase_data = left_phase_data
        self.right_phase_data = right_phase_data
        self.left_noise = left_noise
        self.right_noise = right_noise
        self.simulation_params = simulation_params
        self.protocol_params = protocol_params
    
        tensor = bit_flip.bit_flip(
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

    grad = bit_flip.grad_calc(
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
    left_x_var_last = (left_phase_data[:,-1,0] - left_x_last.mean())**2
    
    # compute gradients for a protocol
    work_grad_a_left = grad.work_grad()
    mean_grad_a_left = 2*(left_x_last.mean() - centers) * grad.current_grad(left_x_last)
    var_grad_a_left = 2*(left_x_var_last.mean() - local_var) * grad.current_grad(left_x_var_last)
    
    # switch to b protocol
    grad.protocol_params = protocol_params['b_list']
    grad.potential_grad_array = left_potential_b_grad
    grad.potential_grad_advance_array = left_potential_b_advance_grad
    grad.drift_grad_array = left_drift_b_grad

    # compute gradients for b protocol
    work_grad_b_left = grad.work_grad()
    mean_grad_b_left = 2*(left_x_last.mean() - centers) * grad.current_grad(left_x_last)
    var_grad_b_left = 2*(left_x_var_last.mean() - local_var) * grad.current_grad(left_x_var_last)

    

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

    grad = bit_flip.grad_calc(
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
   
    # compute gradients for a protocol
    work_grad_a_right = grad.work_grad()
    mean_grad_a_right = 2*(right_x_last.mean() + centers) * grad.current_grad(right_x_last)
    var_grad_a_right = 2*(right_x_var_last.mean() - local_var) * grad.current_grad(right_x_var_last)

    # switch to b protocol
    grad.protocol_params = protocol_params['b_list']
    grad.potential_grad_array = right_potential_b_grad
    grad.potential_grad_advance_array = right_potential_b_advance_grad
    grad.drift_grad_array = right_drift_b_grad

    # compute gradients for b protocol
    work_grad_b_right = grad.work_grad()
    mean_grad_b_right = 2*(right_x_last.mean() + centers) * grad.current_grad(right_x_last)
    var_grad_b_right = 2*(right_x_var_last.mean() - local_var) * grad.current_grad(right_x_var_last)

    a_grad = training_params['alpha'] * (mean_grad_a_left + mean_grad_a_right) + training_params['alpha_1'] * (var_grad_a_left + var_grad_a_right) + training_params['alpha_2'] * (work_grad_a_left + work_grad_a_right) 
    b_grad = training_params['alpha'] * (mean_grad_b_left + mean_grad_b_right) + training_params['alpha_1'] * (var_grad_b_left + var_grad_b_right) + training_params['alpha_2'] * (work_grad_b_left + work_grad_b_right)
