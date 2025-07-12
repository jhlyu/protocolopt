import torch
import gc
import matplotlib.pyplot as plt
import math

"""
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""

def piecewise_protocol_value(num_steps, coefficientlist, endpoints, torch_device):
   
    """
    Generate piecewise linear protocol with endpoints at start and end,
    and internal coefficients spread between step 1 and step num_steps - 1.

    Args:
        coefficientlist (torch.Tensor): 1D tensor of intermediate values (e.g., [x1, ..., xn]).
        endpoints (tuple): (start, end) values for t=0 and t=num_steps.

    Returns:
        torch.Tensor: Protocol of length (num_steps + 1).
    """
    num_coeffs = len(coefficientlist)
        # Define anchor positions:
        # [0] -> endpoints[0]
        # [1] -> coefficientlist[0]
        # [2] -> interpolated between coefficientlist[0] and coefficientlist[1]
        # ...
        # [num_steps - 1] -> coefficientlist[-1]
        # [num_steps] -> endpoints[1]
    protocol = torch.empty(num_steps + 1, device=torch_device)

    # Set start and end
    protocol[0] = endpoints[0]
    protocol[-1] = endpoints[1]

    # Set anchors
    internal_indices = torch.linspace(1, num_steps - 1, num_coeffs, device=torch_device).long()
    full_anchor_indices = torch.cat([
        torch.tensor([0], device=torch_device),
        internal_indices,
        torch.tensor([num_steps], device=torch_device)
    ])

    full_anchor_values = torch.cat([
        torch.tensor([endpoints[0]], device=torch_device),
        coefficientlist.to(torch_device),
        torch.tensor([endpoints[1]], device=torch_device)
    ])

    # Fill in the protocol by linear interpolation between anchors
    for i in range(len(full_anchor_indices) - 1):
        start_idx = full_anchor_indices[i].item()
        end_idx = full_anchor_indices[i + 1].item()
        segment_len = end_idx - start_idx
        if segment_len == 0:
            protocol[start_idx] = full_anchor_values[i]
            continue
        x = torch.linspace(0, 1, segment_len + 1, device=torch_device)
        protocol[start_idx:end_idx + 1] = (
            full_anchor_values[i] * (1 - x) + full_anchor_values[i + 1] * x
        )

    return protocol


class PiecewiseProtocolGradient:
    def __init__(self, params, phase_data, noise, a_list, b_list, a_endpoints, b_endpoints):
        self.params = params
        self.phase_data = phase_data
        self.num_steps = phase_data.shape[1] - 1  # Adjust for zero-based indexing
        self.num_paths = phase_data.shape[0]
        self.noise = noise
        self.a_list = a_list
        self.b_list = b_list
        self.a_endpoints = a_endpoints
        self.b_endpoints = b_endpoints
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    def potential_value(data):
        '''
        returns the value of the potential given some data in the form of self.phase_data coordinates
        '''
        pass

    def dphi_a(data):
        '''
        returns the derivative of the force with respect to a
        '''
        pass

    def dphi_b(data):
        '''
        returns the derivative of the force with respect to b
        '''
        pass

    def dv_da(data):
        '''
        returns the derivative of potential with respect to a
        '''
        pass

    def dv_db(data):
        '''
        returns the derivative of potential with respect to b
        '''
        pass
    
    def piecewise_protocol_value(self, coefficientlist, endpoints):
        return piecewise_protocol_value(self.num_steps, coefficientlist, endpoints, self.torch_device)

    def protocol_grad(self):
        """
        Generates the gradient protocol for a given list of coefficients and number of steps.
        Each coefficient derivative will have a rising and falling slope.

        Args:
            coefficientlist (list): List of coefficients for the protocol.
            numsteps (int): Total number of steps in the protocol.
            torch_device (str): Torch device to place the resulting tensor.

        Returns:
            torch.Tensor: Tensor of shape (len(coefficientlist), numsteps+1).
        """
        m = len(self.a_list)
        num_per_segment = self.num_steps // (m + 1)
    
        if self.num_steps % (m + 1) != 0:
            raise ValueError("num_steps must be divisible by the number of segments plus one.")
    
        grad_array = torch.zeros((m, self.num_steps + 1), dtype=torch.float32, device=self.torch_device)

        for i in range(m):
            start = i * num_per_segment
            peak = start + num_per_segment
            end = start + 2 * num_per_segment

            # Manually generate linspace without endpoint
            rising = torch.arange(num_per_segment, device=self.torch_device) / num_per_segment
            falling = 1.0 - rising

            grad_array[i, start:peak] = rising
            grad_array[i, peak:end] = falling

        return grad_array
    
    def malliavian_weight_array(self, drift_grad_array):
        """
        Calculate the malliavin weight.
        The malliavin weight is given by:
        mw = drift_grad * noise /sigma^2

        Args:
            x (ndarray): Array of positions along the trajectory.
            noise (ndarray): Array of noise values along the trajectory.
            sigma (float): Standard deviation of the noise.

        Returns:
            ndarray: Array of malliavin weights at each time step.
        """
        sigma = self.params['noise_sigma']
        df = drift_grad_array[:,:,:-1] # chopping the last step
        mw = df * self.noise / (sigma ** 2)

        return mw
    
    def work_array(self):
        """
        Calculate the work done along the trajectory.
        The work is V(x0,t1)-V(x0,t0) + V(x1,t2)-V(x1,t1) + ... + V(xN-1,tN) - V(xN-1,tN-1).
        Args:
            trajectory (ndarray): Array of positions along the trajectory.
            a (float): Linear coefficient of the protocol.
            dt (float): Time step size.

        Returns:
            float: Total work done.
        """
    
        potential = self.potential_value_array()[:,:-1]
        potential_advance = self.potential_value_advance_array()
        # Calculate the work done along the trajectory
        work = (potential_advance - potential)
        return work

    def work_grad_array(self, potential_grad, potential_grad_advance):
        potential_grad_value = potential_grad[:,:,:-1]
        potential_grad_advance_value = potential_grad_advance
        work_grad = (potential_grad_advance_value - potential_grad_value)
        return work_grad
    
    def work_grad(self, potential_grad, potential_grad_advance, drift_grad_array):
        """
        Calculate the work gradient.
        The work gradient is given by:
        dw p + dp w

        Args:
            x (ndarray): Array of positions along the trajectory.
            noise (ndarray): Array of noise values along the trajectory.
            sigma (float): Standard deviation of the noise.

        Returns:
            ndarray: Array of work gradients at each time step.
        """
        dwp =self.work_grad_array(potential_grad, potential_grad_advance).sum(axis=2).mean(axis=1)
        w = self.work_array().sum(axis=1)
        mw = self.malliavian_weight_array(drift_grad_array).sum(axis=2)
        pdw= (w * mw).mean(axis=1)
        return dwp + pdw
    
    def x_mean_grad(self, drift_grad_array):
        malliavian_weight = self.malliavian_weight_array(drift_grad_array).sum(axis = 2)
        current= self.phase_data[:,-1,0]
        x_mean_grad = (current * malliavian_weight).mean(axis=-1)

        return x_mean_grad
    
    def x_var_grad(self, drift_grad_array):
        malliavian_weight = self.malliavian_weight_array(drift_grad_array).sum(axis = 2)
        current= (self.phase_data[:,-1,0] - self.phase_data[:,-1,0].mean())**2
        x_var_grad = (current * malliavian_weight).mean(axis=-1)

        return x_var_grad
    
    def get_grad_array(self, derivative_data):
        da_n = self.protocol_grad()
        return torch.einsum('ik,jk->jik', 1*derivative_data, da_n)

    def drift_grad_a_array(self):
        return get_grad_array( self.dphi_a(self.phase_data) )

    def drift_grad_b_array(self):
        return get_grad_array( self.dphi_b(self.phase_data) )
    
    def potential_grad_a_value_array(self):
        return(self.get_grad_array(self.dv_da(self.phase_data))) 
    
    def potential_grad_a_value_advance_array(self):
        return self.get_grad_array( self.dv_da(self.phase_data[:,:-1]) )

    def potential_grad_b_value_array(self):
        return self.get_grad_array(self.dv_db(self.phase_data))

    def potential_grad_b_value_advance_array(self):
        return self.get_grad_array(self.dv_db(self.phase_data[:,:-1]))
    
    def potential_value_array(self):
        return self.potential_value(self.phase_data)
    
    def potential_value_advance_array(self):
        return self.potential_value(self.phase_data[:,:-1])
    

     
class bit_flip_gradient_tensor_comp(PiecewiseProtocolGradient):
    def __init__(self, *args):
        super().__init__(*args)
    
    def potential_value(data):
        protocol_a = piecewise_protocol_value(self.num_steps, self.a_list, self.a_endpoints)
        protocol_b = piecewise_protocol_value(self.num_steps, self.b_list, self.b_endpoints)
        potential = protocol_a * data[...,0]**4 - protocol_b * data[...,0]**2
        return potential

    def dphi_a(self,data):
        return -4*data[...,0]**3
        
    def dphi_b(self,data):
        return 2*data[...,0]
    
    def dv_da(self,data):
        return data[...,0]**4

    def dv_db(self,data):
        return -1*data[...,0]**2

# now we can easily define other classes for othee potentials
class moving_well_gradient_tensor_comp(PiecewiseProtocolGradient):
    def __init__(self, *args):
        super().__init__(*args)
    
    def potential_value(self,data):
        protocol_a = piecewise_protocol_value(self.num_steps, self.a_list, self.a_endpoints)
        protocol_b = piecewise_protocol_value(self.num_steps, self.b_list, self.b_endpoints)
        potential = protocol_a * ( data[...,0] - protocol_b )**2 / 2
        return potential

    def dphi_a(self,data):
        protocol_b = self.piecewise_protocol_value(self.b_list, self.b_endpoints)
        return protocol_b-data[...,0]
        
    def dphi_b(self,data):
        protocol_a = self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        return protocol_a
    
    def dv_da(self,data):
        protocol_b = self.piecewise_protocol_value(self.b_list, self.b_endpoints)
        return (data[...,0]-protocol_b)**2 / 2

    def dv_db(self,data):
        return -1*protocol_a * ( data[...,0] - protocol_b )

        

class bit_flip_simulation:
    """
    A class to generate trajectories for a bit flip gradient tensor computation.
    The potential has the form V(x,t) = a(t) x^4 - b(t)x^2.
    The force is phi(t) =  2 b(t) x - 4 a(t) x^3.
    The initial potential is 5x^4- 10x^2 and the final potential is 5x^4- 10x^2 .

    """
    def __init__(self, num_paths, params, a_list, b_list, a_endpoints, b_endpoints):
        self.params = params
        self.a_list = a_list
        self.b_list = b_list
        self.a_endpoints = a_endpoints
        self.b_endpoints = b_endpoints
        self.num_paths = num_paths
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.variance = 1 /(8* a_endpoints[0] * params['beta']) # variance of the initial distribution


     def piecewise_protocol_value(self, coefficientlist, endpoints):
        return piecewise_protocol_value(self.num_steps, coefficientlist, endpoints, self.torch_device)
    
    def left_sample_distribution(self):
        ### the distribution is ~ exp(- beta * a_endpoints[0]*x^4 - beta * b_endpoints[0]*x^2)
        center = -1* self.params['center']
        var = self.variance
        samples = torch.randn(self.num_paths, device=self.torch_device) * math.sqrt(var) + center
        return samples
    
    def right_sample_distribution(self):
        ### the distribution is ~ exp(- beta * a_endpoints[0]*x^4 - beta * b_endpoints[0]*x^2)
        center = 1* self.params['center']
        var = self.variance
        samples = torch.randn(self.num_paths, device=self.torch_device) * math.sqrt(var) + center
        return samples
    
    def velocity_sample_distribution(self):
        """
        Sample velocities from a Gaussian distribution with mean 0 and variance 1/(2*beta).
        
        Args:
            n_samples (int): Number of samples to generate.
            beta (float): Inverse temperature parameter.
            
        Returns:
            torch.Tensor: A tensor of sampled velocities.
        """
        var = 1 / (2 * self.params['beta'])
        samples = torch.randn(self.num_paths, device=self.torch_device) * math.sqrt(var)
        return samples
    
    def trajectory_generator(self, position_sampler):
        """
        Generates trajectories with protocol.
        The potential has the form V(x,t) = a(t) x^4 - b(t)x^2.
        The force is phi(t) =  2 b(t) x - 4 a(t) x^3.
        The initial potential is 5x^4- 10x^2 and the final potential is 5x^4- 10x^2 .
        The initial distribution is at equilibrium with the initial potential. 
        

        Returns:
            trajectory array and the noise array with [path_index , step_index].
        """
        #time = np.linspace(0, num_steps * dt, num_steps + 1)
        num_steps = self.params['num_steps']
        dt = self.params['dt']
        gamma = self.params['gamma']
        noise_sigma = self.params['noise_sigma']

        protocol_a_value = self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        protocol_b_value = self.piecewise_protocol_value(self.b_list, self.b_endpoints)
        noise_array = torch.randn(self.num_paths, num_steps, device=self.torch_device) * (noise_sigma * math.sqrt(self.params['dt']))
        phase_array = torch.zeros((self.num_paths, num_steps + 1, 2), device=self.torch_device) # 2 for position and velocity
        phase_array[:, 0, 0] = position_sampler() # initial position
        phase_array[:, 0, 1] = self.velocity_sample_distribution() # initial velocity
        for i in range(num_steps):
            # Update position using the Euler-Maruyama method
            # dx = v * dt
            # dv = -gamma * v + ( 4*b*x - 4*a*x^3) * dt + noise
            phase_array[:,i + 1, 0] = phase_array[:, i, 0] + dt * phase_array[:, i, 1]
            phase_array[:,i + 1, 1] = phase_array[:, i, 1] - dt * gamma * phase_array[:, i, 1] - dt * (4 * protocol_a_value[i] * phase_array[:, i, 0]**3 - 2 * protocol_b_value[i] * phase_array[:,i,0] ) + noise_array[:,i]
        return phase_array, noise_array
    
    def left_trajectory_generator(self):
        return self.trajectory_generator(self.left_sample_distribution)
    
    def right_trajectory_generator(self):
        return self.trajectory_generator(self.right_sample_distribution)
    

    