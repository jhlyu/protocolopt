import torch
import gc
import matplotlib.pyplot as plt
from functools import partial
import string
import math

def piecewise_protocol_value(num_steps, coefficientlist, endpoints, torch_device=torch.device('cuda' if torch.cuda.is_available() else 'mps')):
   
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

class quartic_simulation:
    """
    A class to generate trajectories for a particle in quartic potential 
    The potential has the form V(x,t) = a(t) x^4 - b(t)x^2.
    The force is phi(t) =  2 b(t) x - 4 a(t) x^3.
    The default is that the initial potential is 5x^4- 10x^2 and the final potential is 5x^4- 10x^2 .

    """
    def __init__(self, num_paths, params, a_list, b_list, a_endpoints, b_endpoints, device=torch.device('cuda' if torch.cuda.is_available() else 'mps')):
        
        self.params = params
        self.a_list = a_list
        self.b_list = b_list
        self.a_endpoints = a_endpoints
        self.b_endpoints = b_endpoints
        self.num_paths = num_paths
        self.torch_device = device
        self.burn_in = 4000
        self.local_std = 0.4     # standard deviation for local proposal
        self.jump_prob = 0.2     # probability of jumping to opposite mode
# --- Log probability function ---
    def log_prob(self,x):
        return -self.params['beta'] * (self.a_endpoints[0] * x**4 - self.b_endpoints[0] * x**2)

# --- Metropolis-Hastings sampler with mixture proposal ---
    def mh_sampler(self):
        total_steps = self.burn_in + self.num_paths
        samples = torch.empty(total_steps, device='cpu')
        x = torch.tensor([0.0], dtype=torch.float32, device='cpu')  # start at 0

        for i in range(total_steps):
            if torch.rand(1).item() < self.jump_prob:
                x_new = -x + torch.randn(1, device='cpu') * self.local_std
            else:
                x_new = x + torch.randn(1, device='cpu') * self.local_std

            delta = self.log_prob(x_new) - self.log_prob(x)
            if torch.log(torch.rand(1, device='cpu')) < delta:
                x = x_new

            samples[i] = x  # store directly without .item()

        return samples[self.burn_in:].to(self.torch_device)
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
    
    def trajectory_generator(self):
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
        noise_sigma = torch.sqrt( torch.tensor(2 * gamma) / self.params['beta'] )  # noise standard deviation

        protocol_a_value = piecewise_protocol_value(num_steps, self.a_list, self.a_endpoints, torch_device=self.torch_device)
        protocol_b_value = piecewise_protocol_value(num_steps, self.b_list, self.b_endpoints, torch_device=self.torch_device)
        noise_array = torch.randn(self.num_paths, num_steps, device=self.torch_device) * (noise_sigma * math.sqrt(self.params['dt']))
        phase_array = torch.zeros((self.num_paths, num_steps + 1, 2), device=self.torch_device) # 2 for position and velocity
        phase_array[:, 0, 0] = self.mh_sampler() # initial position
        phase_array[:, 0, 1] = self.velocity_sample_distribution() # initial velocity
        for i in range(num_steps):
            # Update position using the Euler-Maruyama method
            # dx = v * dt
            # dv = -gamma * v + ( 4*b*x - 4*a*x^3) * dt + noise
            phase_array[:,i + 1, 0] = phase_array[:, i, 0] + dt * phase_array[:, i, 1]
            phase_array[:,i + 1, 1] = phase_array[:, i, 1] - dt * gamma * phase_array[:, i, 1] - dt * (4 * protocol_a_value[i] * phase_array[:, i, 0]**3 - 2 * protocol_b_value[i] * phase_array[:,i,0] ) + noise_array[:,i]
        return phase_array, noise_array

class quadratic_simulation_od:
    """
    A class to generate trajectories for a particle in quadratic potential 
    The potential has the form V(x,t) = 1/2*(x(t) - a(t))^2.
    The force is phi(t) = -(x(t) - a(t)).
    """
    def __init__(self, num_paths, params, a_list, a_endpoints, device=torch.device('cuda' if torch.cuda.is_available() else 'mps')):
        self.params = params
        self.a_list = a_list
        self.a_endpoints = a_endpoints
        self.num_paths = num_paths
        self.torch_device = device
        self.initial_var = 1 / ( self.params['beta'] )

    def trajectory_generator(self):
        """
        Generates trajectories with protocol.
        The potential has the form V(x,t) = 1/2*(x(t) - a(t))^2.
        The force is phi(t) =  -(x(t) - a(t)).
        
        Returns:
            trajectory array and the noise array with [path_index , step_index, 2].
            The [...,1] is always 0, since the velocity is not used in this simulation.
        """
        #time = np.linspace(0, num_steps * dt, num_steps + 1)
        num_steps = self.params['num_steps']
        dt = self.params['dt']
        noise_sigma = torch.sqrt(torch.tensor(2.0 / self.params['beta'], device=self.torch_device)) # noise standard deviation

        protocol_a_value = piecewise_protocol_value(num_steps, self.a_list, self.a_endpoints)
        noise_array = torch.randn(self.num_paths, num_steps, device=self.torch_device) * (noise_sigma * math.sqrt(self.params['dt']))
        phase_array = torch.zeros((self.num_paths, num_steps + 1, 2), device=self.torch_device) # 2 for position and zeros
        phase_array[:, 0, 0] = torch.randn(self.num_paths, device=self.torch_device) * math.sqrt(self.initial_var) # initial position
        
        for i in range(num_steps):
            # Update position using the Euler-Maruyama method
            # dx = -(x-a)dt+ noise
            phase_array[:,i + 1, 0] = phase_array[:, i, 0] - dt * (phase_array[:, i, 0] - protocol_a_value[i]) + noise_array[:,i]
        return phase_array, noise_array

class grad_calc:
    def __init__(self, sim_params, protocol_params, device = torch.device('cuda' if torch.cuda.is_available() else 'mps') ):
        self.sim_params = sim_params
        self.protocol_params = protocol_params
        #self.torch_device = torch.device('cpu') 
        self.torch_device = device

    def work_array(self, potential, potential_advance):   
        return potential_advance - potential[:,:-1]
    

    def protocol_grad(self):

        """
        Returns gradient of protocol (length num_steps + 1) w.r.t. each coefficient.
        Computes it if it hasnt been run yet

        Anchor indices:
        - index 0        → endpoints[0]
        - index num_steps → endpoints[1]
        - coefficients placed evenly between 1 and num_steps - 1 (inclusive)

        Returns:
            torch.Tensor: shape (m, num_steps + 1)
        """

        try:
            return self.protocol_grad_array
        except AttributeError:
            m = self.protocol_params['num_coefficients']
            num_steps = self.sim_params['num_steps']
            self.protocol_grad_array = torch.empty((m, num_steps + 1), device=self.torch_device)
        


            m = self.protocol_params['num_coefficients']
            num_steps = self.sim_params['num_steps']
            # Anchor indices: [0, 1, ..., num_steps - 1, num_steps]
            internal_anchors = torch.linspace(1, num_steps - 1, m, device=self.torch_device).long()
            full_anchors = torch.cat([
                torch.tensor([0], device=self.torch_device),
                internal_anchors,
                torch.tensor([num_steps], device=self.torch_device)
            ])



            for i in range(m):
                left = full_anchors[i].item()
                center = full_anchors[i + 1].item()
                right = full_anchors[i + 2].item()

                # left → center (inclusive)
                if center > left:
                    x = torch.linspace(0, 1, center - left + 1, device=self.torch_device)
                    self.protocol_grad_array[i, left:center + 1] = x

                # center → right (inclusive)
                if right > center:
                    x = torch.linspace(1, 0, right - center + 1, device=self.torch_device)
                    self.protocol_grad_array[i, center:right + 1] = x

            return self.protocol_grad_array

    def potential_protocol_grad_array(self, potential_array, advance=False):
        da_cn = 1*self.protocol_grad()
        if advance:
            da_cn = da_cn[:,1:]
        return torch.einsum('ik,jk->jik', potential_array, da_cn)
    
    def drift_protocol_grad_array(self, drift_grad_array):
        da_cn = self.protocol_grad()
        dphi_cn = torch.einsum('ik,jk->jik', drift_grad_array, da_cn)
        return dphi_cn
    
    def potential_diff_protocol_grad_array(self, potential, potential_advance):
        potential_grad_value = self.potential_protocol_grad_array(potential)[:,:,:-1]
        potential_grad_advance_value = self.potential_protocol_grad_array(potential_advance, advance=True)
        potential_diff_protocol_grad_array = (potential_grad_advance_value - potential_grad_value)
        return potential_diff_protocol_grad_array

    def malliavian_weight_array(self, drift_protocol_grad_array, noise_array):
        sigma = self.sim_params['noise_sigma']
        df = drift_protocol_grad_array[:,:,:-1] # chopping the last step
        mw = df * noise_array / (sigma ** 2)
        return mw


    def work_protocol_grad(self, potential_array, potential_advance_array, dpotential_array, dpotential_advance_array, drift_protocol_grad_array, noise_array):
        dwp =self.potential_diff_protocol_grad_array(dpotential_array, dpotential_advance_array).sum(axis=2).mean(axis=1)
        w = self.work_array(potential_array, potential_advance_array).sum(axis=1)
        mw = self.malliavian_weight_array(drift_protocol_grad_array, noise_array).sum(axis=2)
        dpw = (w * mw).mean(axis=1)
        return dwp + dpw
    
    def current_grad(self, current, drift_grad_value, noise_array):
        malliavian_weight = self.malliavian_weight_array(drift_grad_value, noise_array).sum(axis=2)
        current_grad = (current * malliavian_weight).mean(axis=-1)
        return current_grad

class DerivativeArrays:
    def __init__(self, params, phase_data, p_lists, p_endpoints):
        self.params = params
        self.phase_data = phase_data
        self.num_steps = phase_data.shape[1] - 1
        self.num_paths = phase_data.shape[0]
        self.parameter_lists = p_lists
        self.parameter_endpoint_lists = p_endpoints
        self.num_parameters = len(p_lists)
        self.torch_device = phase_data.device
    
    def get_protocol_values(self):
        return [ piecewise_protocol_value(self.num_steps, p_item, p_endpoint, torch_device=self.torch_device) for p_item, p_endpoint in zip(self.parameter_lists, self.parameter_endpoint_lists) ]

    def get_array(self, function, advance=False):
        if advance:
            advance_protocol_values = [item[1:] for item in self.get_protocol_values()]
            return function(self.phase_data[:,:-1], advance_protocol_values)
        else:
            return function(self.phase_data, self.get_protocol_values())

    def potential_function(self, coordinates, parameters):
        '''
        returns the potential energy
        This should be overridden in subclasses.
        '''
        pass

    def potential_value_array(self):
        return self.get_array(self.potential_function)

    def potential_value_advance_array(self):
        return self.get_array(self.potential_function, advance=True)

    def dV_a(self, coordinates, parameters):
        """
        Returns the gradient functions for the potential with respect to each parameter.
        These should be overridden in subclasses.
        There will be several of these and should be named dV_a, dV_b, etc...
        """
        pass

    def dF_a(self, coordinates, parameters):
        """
        Returns the gradient functions for the potential with respect to each parameter.
        These should be overridden in subclasses.
        There will be several of these and should be named dF_a, dF_b, etc...
        """
        pass

    def set_derivative_methods(self):
        '''
        sets the methods above to be named according to Jinghao's convention
        currently just meant so that this code interacts with test_main.py
        can be changed later, since there is no reason to name the functions "a", "b", etc...
        will work up to 26 parameters currently
        '''
        parameter_labels = list(string.ascii_lowercase) #just  a list of letters a-z
        
        for i in range(self.num_parameters):
            label = parameter_labels[i]
            dF, dV = getattr(self, f'dF_{label}'), getattr(self, f'dV_{label}')

            dF_label = f'drift_grad_{label}_array'
            dV_label = f'potential_grad_{label}_value_array'
            dV_advance_label = f'potential_grad_{label}_value_advance_array'

            try:
                setattr(self, dF_label, partial(self.get_array, dF) )
            except:
                setattr(self,  dF_label, None)

            try:
                setattr(self, dV_label, partial(self.get_array, dV) )
            except:
                setattr(self, dV_label, None)

            try:
                setattr(self, dV_advance_label, partial(self.get_array, dV, advance=True) )
            except:
                setattr(self, dV_advance_label, None )

    
class bit_flip(DerivativeArrays):
    """
    The potential is V(x) = a * x^4 - b * x^2 with flipping bits
    The potential gradients are d_a V = x^4, d_b V = -x^2
    The drift is F = -d_x V = -4 * a * x^3 + 2 * b * x
    The drift gradients are d_a F = -4 * x^3, d_b F = 2 * x
    """
    def __init__(self, params, phase_data, a_list, b_list, a_endpoints, b_endpoints):
        super().__init__(params, phase_data, [a_list, b_list], [a_endpoints, b_endpoints])

        self.centers = math.sqrt(self.parameter_endpoint_lists[1][1]/(2*self.parameter_endpoint_lists[0][1]))
        self.set_derivative_methods()

    def potential_function(self, coordinates, protocol_values):
        protocol_a, protocol_b = protocol_values
        potential = protocol_a * coordinates[...,0]**4 -  protocol_b * coordinates[...,0]**2
        return potential
    
    def dF_a(self, coordinates, protocol_values):
        return -4 * coordinates[...,0]**3
    
    def dF_b(self, coordinates, protocol_values):
        return 2 * coordinates[...,0]
    
    def dV_a(self, coordinates, protocol_values):
        return 1 * coordinates[...,0]**4
    
    def dV_b(self, coordinates, protocol_values):
        return -1 * coordinates[...,0]**2
    
    def distance_sq_current(self, coordinates, protocol_values, order=2):
        target = (coordinates[:, 0, 0] < 0) * self.centers - (coordinates[:, 0, 0] > 0) * self.centers
        distance_sq = (coordinates[:,-1,0] - target)**order
        return distance_sq
    
    def var_current_separation(self, coordinates, protocol_values):
        # Calculate the variance of the current, the coordinates must be categorized into left and right paths in advance!
        num_left = torch.sum(coordinates[:, 0, 0] < 0)
        num_right = torch.sum(coordinates[:, 0, 0] > 0)
        var = torch.zeros(coordinates.shape[0], device=coordinates.device)
        var[:num_left] = (coordinates[:num_left, -1, 0] - coordinates[:num_left, -1, 0].mean())**2
        var[num_left:] = (coordinates[num_left:, -1, 0] - coordinates[num_left:, -1, 0].mean())**2
        return var
    
    def var_current(self, coordinates, protocol_values):
        # Calculate the variance of the current,
        left = coordinates[:, 0, 0] < 0
        right = coordinates[:, 0, 0] > 0
        var = torch.zeros(coordinates.shape[0], device=coordinates.device)
        var[left] = (coordinates[left, -1, 0] - coordinates[left, -1, 0].mean())**2
        var[right] = (coordinates[right, -1, 0] - coordinates[right, -1, 0].mean())**2
        return var
    
    def var_abs_current(self, coordinates, protocol_values):
        # Calculate the variance of the absolute value of the current
        coordinates_abs = torch.abs(coordinates[:,-1,0])
        mean_abs = coordinates_abs.mean()
        return (coordinates_abs - mean_abs)**2


class bit_erasure(DerivativeArrays):
    """
    The potential is V(x) = a * x^4 - b * x^2 + c * x
    The potential gradients are d_a V = x^4, d_b V = -x^2, d_c V = x
    The drift is F = -d_x V = -4 * a * x^3 + 2 * b * x - c
    The drift gradients are d_a F = -4 * x^3, d_b F = 2 * x, d_c F = -1
    """
  
    def __init__(self, params, phase_data, a_list, b_list, c_list, a_endpoints, b_endpoints, c_endpoints):
        super().__init__(params, phase_data, [a_list, b_list, c_list], [a_endpoints, b_endpoints, c_endpoints])

        self.set_derivative_methods()

    def potential_function(self, coordinates, protocol_values):
        protocol_a, protocol_b, protocol_c = protocol_values
        potential = protocol_a * coordinates[...,0]**4 -  protocol_b * coordinates[...,0]**2 + protocol_c * coordinates[...,0]
        return potential
    
    def dF_a(self, coordinates, protocol_values):
        return -4 * coordinates[...,0]**3

    def dF_b(self, coordinates, protocol_values):
        return 2 * coordinates[...,0]

    def dF_c(self, coordinates, protocol_values):
        return -1 * torch.ones_like(coordinates[...,0])

    def dV_a(self, coordinates, protocol_values):
        return 1 * coordinates[...,0]**4

    def dV_b(self, coordinates, protocol_values):
        return -1 * coordinates[...,0]**2

    def dV_c(self, coordinates, protocol_values):
        return 1 * coordinates[...,0]
    
class moving_harmonic(DerivativeArrays):
    def __init__(self, params, phase_data, a_list, a_endpoints):
        super().__init__(params, phase_data, [a_list], [a_endpoints])

        self.set_derivative_methods()
    """
    The potential is V(x) = 1/2 * (x(t) - a(t))^2
    The potential gradients are d_a V = -(x(t) - a(t))
    The drift is F = -d_x V = - (x(t) - a(t))
    The drift gradients are d_a F = 1
    """
    def potential_function(self, coordinates, protocol_values):
        protocol_a = protocol_values
        potential = 1/2 * (coordinates[...,0] - protocol_values[0])**2
        return potential
    
    def dF_a(self, coordinates, protocol_values):
        return torch.ones_like(coordinates[...,0])

    def dV_a(self, coordinates, protocol_values):
        return -1 * (coordinates[...,0] - protocol_values[0])

    def distance_sq_current(self, coordinates, protocol_values, target, order=2):
        distance_sq = (coordinates[:,-1,0] - target)**order
        return distance_sq