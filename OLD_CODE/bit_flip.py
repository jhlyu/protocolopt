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
def piecewise_protocol_value(num_steps, coefficientlist, endpoints):
   
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
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
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

class bit_flip_gradient_tensor_comp:
    ### THe potential is V(x) = a * x^4 - b * x^2 with flipping bits
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
    

    
    def piecewise_protocol_value(self, coefficientlist, endpoints):
        """
        PyTorch version of piecewise linear protocol value generation using global interpolation.
        
        Args:
            coefficientlist (torch.Tensor): 1D tensor of intermediate values (e.g., [x1, x2, ..., xn]).
            endpoints (tuple): Tuple of two scalars (start, end), e.g., (0.0, 1.0).
            numsteps (int): Number of time steps.
        Returns:
            torch.Tensor: 1D tensor of length (numsteps + 1), holding protocol values on torch_device.
        """
        total_anchors = len(coefficientlist) + 2
        num_per_segment = self.num_steps // (total_anchors - 1)
        
        # Define anchor indices: [0, num_per_segment, 2*num_per_segment, ..., numsteps]
        anchor_indices = torch.arange(0, self.num_steps + 1, num_per_segment, device=self.torch_device)
        if anchor_indices[-1] != self.num_steps:
            anchor_indices[-1] = self.num_steps  # ensure final anchor is exactly at numsteps

        # Build anchor values: [start, x1, x2, ..., xn, end]
        anchor_values = torch.cat([
            torch.tensor([endpoints[0]], device=self.torch_device, dtype=torch.float32),
            coefficientlist.to(self.torch_device),
            torch.tensor([endpoints[1]], device=self.torch_device, dtype=torch.float32)
        ])

        # Time points where we want interpolated values
        #query_indices = torch.arange(numsteps + 1, device=torch_device, dtype=torch.float32)

        # Perform manual linear interpolation
        protocol = torch.empty(self.num_steps + 1, device=self.torch_device)

        for i in range(len(anchor_indices) - 1):
            start_idx = anchor_indices[i].item()
            end_idx = anchor_indices[i+1].item()
            x = torch.linspace(0, 1, end_idx - start_idx + 1, device=self.torch_device)
            protocol[start_idx:end_idx + 1] = (
                anchor_values[i] * (1 - x) + anchor_values[i + 1] * x
            )

        return protocol
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
    def drift_grad_a_array(self):
        """
    Calculate the drift gradient at position x and time t with output being same as position array.
    dphi/da = -4 * x^3 
    da_dcn = delta shape functions.
    
    Args:
        x (ndarray): Array of positions along the trajectory.
        coefficientlist (list): List of coefficients for the Fourier Series.
        dt (float): Time step size.

    Returns:
        ndarray: Array of drift gradients at each time step.
        Dimensions: (num_coefficients, num_paths, num_steps)
        """
    
        dphi_a = -4*self.phase_data[:,:,0]**3
        da_cn = self.protocol_grad()
        dphi_cn = torch.einsum('ik,jk->jik', dphi_a, da_cn)

        del dphi_a, da_cn  # Free memory
        gc.collect()
        return dphi_cn
    
    def drift_grad_b_array(self):
        """
        Calculate the drift gradient at position x and time t with output being same as position array.
        dphi/db = 2 * x 
        db_dcn = delta shape functions.
        
        Args:
            x (ndarray): Array of phase data along the trajectory. idx of x is [path_index, step_index, 0] for position and [path_index, step_index, 1] for velocity.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of drift gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        
    
        dphi_b = 2*self.phase_data[:,:,0]
        db_dn = self.protocol_grad()
        dphi_dn = torch.einsum('ik,jk->jik', dphi_b, db_dn)

        del dphi_b, db_dn  # Free memory
        gc.collect()
        return dphi_dn
    
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

        del df 
        gc.collect()
        return mw
    
    def x_mean_grad(self, drift_grad_array):
        malliavian_weight = self.malliavian_weight_array(drift_grad_array).sum(axis = 2)
        current= self.phase_data[:,-1,0] # Get the last position of each path
        x_mean_grad = (current * malliavian_weight).mean(axis=-1)

        del malliavian_weight, current  # Free memory
        gc.collect()
        return x_mean_grad
    
    def x_var_grad(self, drift_grad_array):
        malliavian_weight = self.malliavian_weight_array(drift_grad_array).sum(axis = 2)
        current= (self.phase_data[:,-1,0] - self.phase_data[:,-1,0].mean())**2 # Get the last position of each path
        x_var_grad = (current * malliavian_weight).mean(axis=-1)

        del malliavian_weight, current  # Free memory
        gc.collect()
        return x_var_grad
    
    def potential_value_array(self):
        """
        Calculate the potential value at phase position x and time t with output being same as position array.
        x[:,:,0] is the position at time t, and x[:,:,1] is the velocity at time t.
        """
       
        protocol_a= self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        protocol_b= self.piecewise_protocol_value(self.b_list, self.b_endpoints)
        potential = protocol_a * self.phase_data[:,:,0]**4 -  protocol_b * self.phase_data[:,:,0]**2
        return potential
    
    def potential_value_advance_array(self):
        """
        Calculate the potential value at position x and time t+dt with output being same as position array but chopped the last column.


        Returns:
            ndarray: Array of advance potential values.
            Dimensions: (num_paths, num_steps - 1)
        """
    
        protocol_a_advance = self.piecewise_protocol_value(self.a_list, self.a_endpoints)[1:]
        protocol_b_advance = self.piecewise_protocol_value(self.b_list, self.b_endpoints)[1:]
        potential_advance = protocol_a_advance*self.phase_data[:,:-1,0]**4 - protocol_b_advance * self.phase_data[:,:-1,0]**2 
        return potential_advance
    
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
    
    def potential_grad_a_value_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t with output being same as position array.
        dV/dcn = x  * -da_cn
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        dv_da = self.phase_data[:,:,0]**4 
        da_cn = 1*self.protocol_grad()

        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    def potential_grad_a_value_advance_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t+dt with output being same as position array.
        dV/dcn = x  * da_cn(t+dt)
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
       
        dv_da = self.phase_data[:,:-1,0]**4 
        da_cn = 1*self.protocol_grad()[:,1:]

        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    def potential_grad_b_value_array(self):
        """
        Calculate the potential value grad with parameters in $b$ (called dn) at position x and time t with output being same as position array.
        dV/dcn =  -x^2
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients of middle points.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps-1)
        """
        
        dv_db = -1*self.phase_data[:,:,0]**2 
        db_dn = self.protocol_grad()

        return torch.einsum('ik,jk->jik', dv_db, db_dn)

    
    def potential_grad_b_value_advance_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t+dt with output being same as position array.
        dV/dcn = x  * -db_dn(t+dt)
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients of middle points.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps-1)
        """
       
        dv_db = -1*self.phase_data[:,:-1,0]**2 
        db_dn = self.protocol_grad()[:,1:]

        return torch.einsum('ik,jk->jik', dv_db, db_dn)
    
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
        """
        PyTorch version of piecewise linear protocol value generation using global interpolation.
        
        Args:
            coefficientlist (torch.Tensor): 1D tensor of intermediate values (e.g., [x1, x2, ..., xn]).
            endpoints (tuple): Tuple of two scalars (start, end), e.g., (0.0, 1.0).
            numsteps (int): Number of time steps.
        Returns:
            torch.Tensor: 1D tensor of length (numsteps + 1), holding protocol values on torch_device.
        """
        num_steps = self.params['num_steps']
        total_anchors = len(coefficientlist) + 2
        num_per_segment = num_steps // (total_anchors - 1)
        
        # Define anchor indices: [0, num_per_segment, 2*num_per_segment, ..., numsteps]
        anchor_indices = torch.arange(0, num_steps + 1, num_per_segment, device=self.torch_device)
        if anchor_indices[-1] != num_steps:
            anchor_indices[-1] = num_steps  # ensure final anchor is exactly at numsteps

        # Build anchor values: [start, x1, x2, ..., xn, end]
        anchor_values = torch.cat([
            torch.tensor([endpoints[0]], device=self.torch_device, dtype=torch.float32),
            coefficientlist.to(self.torch_device),
            torch.tensor([endpoints[1]], device=self.torch_device, dtype=torch.float32)
        ])

        # Time points where we want interpolated values
        #query_indices = torch.arange(numsteps + 1, device=torch_device, dtype=torch.float32)

        # Perform manual linear interpolation
        protocol = torch.empty(num_steps + 1, device=self.torch_device)

        for i in range(len(anchor_indices) - 1):
            start_idx = anchor_indices[i].item()
            end_idx = anchor_indices[i+1].item()
            x = torch.linspace(0, 1, end_idx - start_idx + 1, device=self.torch_device)
            protocol[start_idx:end_idx + 1] = (
                anchor_values[i] * (1 - x) + anchor_values[i + 1] * x
            )

        return protocol

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
    

class od_moving_simulation:
    """
    A class to generate trajectories for a moving laser trap.
    The potential has the form V(x,t) = 1/2(x-a(t))^2 
    The force is phi(t) =  -(x-a(t)).
    """
    def __init__(self, num_paths, params, a_list, a_endpoints):
        self.params = params
        self.a_list = a_list
        self.a_endpoints = a_endpoints
        self.num_paths = num_paths
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.variance = 1 /(params['beta']) # variance of the initial distribution
    
    def piecewise_protocol_value(self, coefficientlist, endpoints):
        """
        PyTorch version of piecewise linear protocol value generation using global interpolation.
        
        Args:
            coefficientlist (torch.Tensor): 1D tensor of intermediate values (e.g., [x1, x2, ..., xn]).
            endpoints (tuple): Tuple of two scalars (start, end), e.g., (0.0, 1.0).
            numsteps (int): Number of time steps.
        Returns:
            torch.Tensor: 1D tensor of length (numsteps + 1), holding protocol values on torch_device.
        """
        num_steps = self.params['num_steps']
        total_anchors = len(coefficientlist) + 2
        num_per_segment = num_steps // (total_anchors - 1)
        
        # Define anchor indices: [0, num_per_segment, 2*num_per_segment, ..., numsteps]
        anchor_indices = torch.arange(0, num_steps + 1, num_per_segment, device=self.torch_device)
        if anchor_indices[-1] != num_steps:
            anchor_indices[-1] = num_steps  # ensure final anchor is exactly at numsteps

        # Build anchor values: [start, x1, x2, ..., xn, end]
        anchor_values = torch.cat([
            torch.tensor([endpoints[0]], device=self.torch_device, dtype=torch.float32),
            coefficientlist.to(self.torch_device),
            torch.tensor([endpoints[1]], device=self.torch_device, dtype=torch.float32)
        ])

        # Time points where we want interpolated values
        #query_indices = torch.arange(numsteps + 1, device=torch_device, dtype=torch.float32)

        # Perform manual linear interpolation
        protocol = torch.empty(num_steps + 1, device=self.torch_device)

        for i in range(len(anchor_indices) - 1):
            start_idx = anchor_indices[i].item()
            end_idx = anchor_indices[i+1].item()
            x = torch.linspace(0, 1, end_idx - start_idx + 1, device=self.torch_device)
            protocol[start_idx:end_idx + 1] = (
                anchor_values[i] * (1 - x) + anchor_values[i + 1] * x
            )

        return protocol
    
    def initial_sample_distribution(self):
        ### the distribution is ~ exp(- beta *1/2 ax^2)
        center = 1* self.params['center']
        var = self.variance
        samples = torch.randn(self.num_paths, device=self.torch_device) * math.sqrt(var) + center
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
        noise_sigma = self.params['noise_sigma']

        protocol_a_value = self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        noise_array = torch.randn(self.num_paths, num_steps, device=self.torch_device) * (noise_sigma * math.sqrt(self.params['dt']))
        phase_array = torch.zeros((self.num_paths, num_steps + 1), device=self.torch_device) 
        phase_array[:, 0] = position_sampler() # initial position
        for i in range(num_steps):
            # Update position using the Euler-Maruyama method
            dx = - (phase_array[:, i] - protocol_a_value[i]) * dt + noise_array[:, i]
            phase_array[:, i + 1] = phase_array[:, i] + dx
        
        return phase_array, noise_array
    

class od_moving_gradient_tensor_comp:
    ### THe potential is V(x) = 1/2(x-a)^2
    def __init__(self, params, phase_data, noise, a_list, a_endpoints):
        self.params = params
        self.phase_data = phase_data
        self.num_steps = phase_data.shape[1] - 1  # Adjust for zero-based indexing
        self.num_paths = phase_data.shape[0]
        self.noise = noise
        self.a_list = a_list
        self.a_endpoints = a_endpoints
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    

    
    def piecewise_protocol_value(self, coefficientlist, endpoints):
        """
        PyTorch version of piecewise linear protocol value generation using global interpolation.
        
        Args:
            coefficientlist (torch.Tensor): 1D tensor of intermediate values (e.g., [x1, x2, ..., xn]).
            endpoints (tuple): Tuple of two scalars (start, end), e.g., (0.0, 1.0).
            numsteps (int): Number of time steps.
        Returns:
            torch.Tensor: 1D tensor of length (numsteps + 1), holding protocol values on torch_device.
        """
        total_anchors = len(coefficientlist) + 2
        num_per_segment = self.num_steps // (total_anchors - 1)
        
        # Define anchor indices: [0, num_per_segment, 2*num_per_segment, ..., numsteps]
        anchor_indices = torch.arange(0, self.num_steps + 1, num_per_segment, device=self.torch_device)
        if anchor_indices[-1] != self.num_steps:
            anchor_indices[-1] = self.num_steps  # ensure final anchor is exactly at numsteps

        # Build anchor values: [start, x1, x2, ..., xn, end]
        anchor_values = torch.cat([
            torch.tensor([endpoints[0]], device=self.torch_device, dtype=torch.float32),
            coefficientlist.to(self.torch_device),
            torch.tensor([endpoints[1]], device=self.torch_device, dtype=torch.float32)
        ])

        # Time points where we want interpolated values
        #query_indices = torch.arange(numsteps + 1, device=torch_device, dtype=torch.float32)

        # Perform manual linear interpolation
        protocol = torch.empty(self.num_steps + 1, device=self.torch_device)

        for i in range(len(anchor_indices) - 1):
            start_idx = anchor_indices[i].item()
            end_idx = anchor_indices[i+1].item()
            x = torch.linspace(0, 1, end_idx - start_idx + 1, device=self.torch_device)
            protocol[start_idx:end_idx + 1] = (
                anchor_values[i] * (1 - x) + anchor_values[i + 1] * x
            )

        return protocol
    
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
    def drift_grad_a_array(self):
        """
    Calculate the drift gradient at position x and time t with output being same as position array.
    dphi/da = 1
    da_dcn = delta shape functions.
    
    Args:
        x (ndarray): Array of positions along the trajectory.
        coefficientlist (list): List of coefficients for the Fourier Series.
        dt (float): Time step size.

    Returns:
        ndarray: Array of drift gradients at each time step.
        Dimensions: (num_coefficients, num_paths, num_steps)
        """
    
        dphi_a = torch.ones(self.phase_data.shape[0], self.phase_data.shape[1], device=self.torch_device)  # dphi/da = 1
        da_cn = self.protocol_grad()
        dphi_cn = torch.einsum('ik,jk->jik', dphi_a, da_cn)

        del dphi_a, da_cn  # Free memory
        gc.collect()
        return dphi_cn
    
    
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

        del df 
        gc.collect()
        return mw
    
    def x_mean_grad(self, drift_grad_array):
        malliavian_weight = self.malliavian_weight_array(drift_grad_array).sum(axis = 2)
        current= self.phase_data[:,-1] # Get the last position of each path
        x_mean_grad = (current * malliavian_weight).mean(axis=-1)

        del malliavian_weight, current  # Free memory
        gc.collect()
        return x_mean_grad
    
    
    def potential_value_array(self):
        """
        Calculate the potential value at phase position x and time t with output being same as position array.
        x[:,:,0] is the position at time t, and x[:,:,1] is the velocity at time t.
        """
        protocol_a = self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        potential = 1/2 * (self.phase_data - protocol_a)**2
        return potential
    
    def potential_value_advance_array(self):
        """
        Calculate the potential value at position x and time t+dt with output being same as position array but chopped the last column.


        Returns:
            ndarray: Array of advance potential values.
            Dimensions: (num_paths, num_steps - 1)
        """
    
        protocol_a_advance = self.piecewise_protocol_value(self.a_list, self.a_endpoints)[1:]
        potential_advance = 1/2 * (self.phase_data[:,:-1] - protocol_a_advance)**2
        return potential_advance
    
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
    
    def potential_grad_a_value_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t with output being same as position array.
        dV/da = (a-x) * da_cn
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps+1)
        """
        protocol_a= self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        dv_da = protocol_a - self.phase_data 
        da_cn = self.protocol_grad()

        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    def potential_grad_a_value_advance_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t+dt with output being same as position array.
        dV/dcn = x  * da_cn(t+dt)
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        protocol_a_advance = self.piecewise_protocol_value(self.a_list, self.a_endpoints)[1:]
        dv_da = protocol_a_advance - self.phase_data[:,:-1]
        da_cn = self.protocol_grad()[:,1:]

        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    
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
    

class od_moving_simulation_jump:
    """
    A class to generate trajectories for a moving laser trap.
    The potential has the form V(x,t) = 1/2(x-a(t))^2 
    The force is phi(t) =  -(x-a(t)).
    """
    def __init__(self, num_paths, params, a_list, a_endpoints):
        self.params = params
        self.a_list = a_list
        self.a_endpoints = a_endpoints
        self.num_paths = num_paths
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.variance = 1 /(params['beta']) # variance of the initial distribution
    
    def piecewise_protocol_value(self, coefficientlist, endpoints):
   
        """
        Generate piecewise linear protocol with endpoints at start and end,
        and internal coefficients spread between step 1 and step num_steps - 1.

        Args:
            coefficientlist (torch.Tensor): 1D tensor of intermediate values (e.g., [x1, ..., xn]).
            endpoints (tuple): (start, end) values for t=0 and t=num_steps.

        Returns:
            torch.Tensor: Protocol of length (num_steps + 1).
        """
        num_steps = self.params['num_steps']
        num_coeffs = len(coefficientlist)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Define anchor positions:
        # [0] -> endpoints[0]
        # [1] -> coefficientlist[0]
        # [2] -> interpolated between coefficientlist[0] and coefficientlist[1]
        # ...
        # [num_steps - 1] -> coefficientlist[-1]
        # [num_steps] -> endpoints[1]
        protocol = torch.empty(num_steps + 1, device=self.torch_device)

        # Set start and end
        protocol[0] = endpoints[0]
        protocol[-1] = endpoints[1]

        # Set anchors
        internal_indices = torch.linspace(1, num_steps - 1, num_coeffs, device=self.torch_device).long()
        full_anchor_indices = torch.cat([
            torch.tensor([0], device=self.torch_device),
            internal_indices,
            torch.tensor([num_steps], device=self.torch_device)
        ])

        full_anchor_values = torch.cat([
            torch.tensor([endpoints[0]], device=self.torch_device),
            coefficientlist.to(self.torch_device),
            torch.tensor([endpoints[1]], device=self.torch_device)
        ])

        # Fill in the protocol by linear interpolation between anchors
        for i in range(len(full_anchor_indices) - 1):
            start_idx = full_anchor_indices[i].item()
            end_idx = full_anchor_indices[i + 1].item()
            segment_len = end_idx - start_idx
            if segment_len == 0:
                protocol[start_idx] = full_anchor_values[i]
                continue
            x = torch.linspace(0, 1, segment_len + 1, device=self.torch_device)
            protocol[start_idx:end_idx + 1] = (
                full_anchor_values[i] * (1 - x) + full_anchor_values[i + 1] * x
            )

        return protocol

    
    def initial_sample_distribution(self):
        ### the distribution is ~ exp(- beta *1/2 ax^2)
        center = 1* self.params['center']
        var = self.variance
        samples = torch.randn(self.num_paths, device=self.torch_device) * math.sqrt(var) + center
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
        noise_sigma = self.params['noise_sigma']

        protocol_a_value = self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        noise_array = torch.randn(self.num_paths, num_steps, device=self.torch_device) * (noise_sigma * math.sqrt(self.params['dt']))
        phase_array = torch.zeros((self.num_paths, num_steps + 1), device=self.torch_device) 
        phase_array[:, 0] = position_sampler() # initial position
        for i in range(num_steps):
            # Update position using the Euler-Maruyama method
            dx = - (phase_array[:, i] - protocol_a_value[i]) * dt + noise_array[:, i]
            phase_array[:, i + 1] = phase_array[:, i] + dx
        
        return phase_array, noise_array
    
class od_moving_gradient_tensor_comp_jump:
    ### THe potential is V(x) = 1/2(x-a)^2
    def __init__(self, params, phase_data, noise, a_list, a_endpoints):
        self.params = params
        self.phase_data = phase_data
        self.num_steps = phase_data.shape[1] - 1  # Adjust for zero-based indexing
        self.num_paths = phase_data.shape[0]
        self.noise = noise
        self.a_list = a_list
        self.a_endpoints = a_endpoints
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    

    
    def piecewise_protocol_value(self, coefficientlist, endpoints):
   
        """
        Generate piecewise linear protocol with endpoints at start and end,
        and internal coefficients spread between step 1 and step num_steps - 1.

        Args:
            coefficientlist (torch.Tensor): 1D tensor of intermediate values (e.g., [x1, ..., xn]).
            endpoints (tuple): (start, end) values for t=0 and t=num_steps.

        Returns:
            torch.Tensor: Protocol of length (num_steps + 1).
        """
        num_steps = self.params['num_steps']
        num_coeffs = len(coefficientlist)
        # Define anchor positions:
        # [0] -> endpoints[0]
        # [1] -> coefficientlist[0]
        # [2] -> interpolated between coefficientlist[0] and coefficientlist[1]
        # ...
        # [num_steps - 1] -> coefficientlist[-1]
        # [num_steps] -> endpoints[1]
        protocol = torch.empty(num_steps + 1, device=self.torch_device)

        # Set start and end
        protocol[0] = endpoints[0]
        protocol[-1] = endpoints[1]

        # Set anchors
        internal_indices = torch.linspace(1, num_steps - 1, num_coeffs, device=self.torch_device).long()
        full_anchor_indices = torch.cat([
            torch.tensor([0], device=self.torch_device),
            internal_indices,
            torch.tensor([num_steps], device=self.torch_device)
        ])

        full_anchor_values = torch.cat([
            torch.tensor([endpoints[0]], device=self.torch_device),
            coefficientlist.to(self.torch_device),
            torch.tensor([endpoints[1]], device=self.torch_device)
        ])

        # Fill in the protocol by linear interpolation between anchors
        for i in range(len(full_anchor_indices) - 1):
            start_idx = full_anchor_indices[i].item()
            end_idx = full_anchor_indices[i + 1].item()
            segment_len = end_idx - start_idx
            if segment_len == 0:
                protocol[start_idx] = full_anchor_values[i]
                continue
            x = torch.linspace(0, 1, segment_len + 1, device=self.torch_device)
            protocol[start_idx:end_idx + 1] = (
                full_anchor_values[i] * (1 - x) + full_anchor_values[i + 1] * x
            )

        return protocol
    
    def protocol_grad(self):
        """
        Compute gradient of protocol (length num_steps + 1) w.r.t. each coefficient.

        Anchor indices:
        - index 0        → endpoints[0]
        - index num_steps → endpoints[1]
        - coefficients placed evenly between 1 and num_steps - 1 (inclusive)

        Returns:
            torch.Tensor: shape (m, num_steps + 1)
        """
        m = len(self.a_list)
        num_steps = self.params['num_steps']
        # Anchor indices: [0, 1, ..., num_steps - 1, num_steps]
        internal_anchors = torch.linspace(1, num_steps - 1, m, device=self.torch_device).long()
        full_anchors = torch.cat([
            torch.tensor([0], device=self.torch_device),
            internal_anchors,
            torch.tensor([num_steps], device=self.torch_device)
        ])

        grad_array = torch.zeros((m, num_steps + 1), device=self.torch_device)

        for i in range(m):
            left = full_anchors[i].item()
            center = full_anchors[i + 1].item()
            right = full_anchors[i + 2].item()

            # left → center (inclusive)
            if center > left:
                x = torch.linspace(0, 1, center - left + 1, device=self.torch_device)
                grad_array[i, left:center + 1] = x

            # center → right (inclusive)
            if right > center:
                x = torch.linspace(1, 0, right - center + 1, device=self.torch_device)
                grad_array[i, center:right + 1] = x

        return grad_array


    def drift_grad_a_array(self):
        """
    Calculate the drift gradient at position x and time t with output being same as position array.
    dphi/da = 1
    da_dcn = delta shape functions.
    
    Args:
        x (ndarray): Array of positions along the trajectory.
        coefficientlist (list): List of coefficients for the Fourier Series.
        dt (float): Time step size.

    Returns:
        ndarray: Array of drift gradients at each time step.
        Dimensions: (num_coefficients, num_paths, num_steps)
        """
    
        dphi_a = torch.ones(self.phase_data.shape[0], self.phase_data.shape[1], device=self.torch_device)  # dphi/da = 1
        da_cn = self.protocol_grad()
        dphi_cn = torch.einsum('ik,jk->jik', dphi_a, da_cn)

        del dphi_a, da_cn  # Free memory
        gc.collect()
        return dphi_cn
    
    
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

        del df 
        gc.collect()
        return mw
    
    def x_mean_grad(self, drift_grad_array):
        malliavian_weight = self.malliavian_weight_array(drift_grad_array).sum(axis = 2)
        current= self.phase_data[:,-1] # Get the last position of each path
        x_mean_grad = (current * malliavian_weight).mean(axis=-1)

        del malliavian_weight, current  # Free memory
        gc.collect()
        return x_mean_grad
    
    
    def potential_value_array(self):
        """
        Calculate the potential value at phase position x and time t with output being same as position array.
        x[:,:,0] is the position at time t, and x[:,:,1] is the velocity at time t.
        """
        protocol_a = self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        potential = 1/2 * (self.phase_data - protocol_a)**2
        return potential
    
    def potential_value_advance_array(self):
        """
        Calculate the potential value at position x and time t+dt with output being same as position array but chopped the last column.


        Returns:
            ndarray: Array of advance potential values.
            Dimensions: (num_paths, num_steps - 1)
        """
    
        protocol_a_advance = self.piecewise_protocol_value(self.a_list, self.a_endpoints)[1:]
        potential_advance = 1/2 * (self.phase_data[:,:-1] - protocol_a_advance)**2
        return potential_advance
    
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
    
    def potential_grad_a_value_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t with output being same as position array.
        dV/da = (a-x) * da_cn
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps+1)
        """
        protocol_a= self.piecewise_protocol_value(self.a_list, self.a_endpoints)
        dv_da = protocol_a - self.phase_data 
        da_cn = self.protocol_grad()

        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    def potential_grad_a_value_advance_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t+dt with output being same as position array.
        dV/dcn = x  * da_cn(t+dt)
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        protocol_a_advance = self.piecewise_protocol_value(self.a_list, self.a_endpoints)[1:]
        dv_da = protocol_a_advance - self.phase_data[:,:-1]
        da_cn = self.protocol_grad()[:,1:]

        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    
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
    

class quartic_simulation:
    """
    A class to generate trajectories for a particle in quartic potential 
    The potential has the form V(x,t) = a(t) x^4 - b(t)x^2.
    The force is phi(t) =  2 b(t) x - 4 a(t) x^3.
    The default is that the initial potential is 5x^4- 10x^2 and the final potential is 5x^4- 10x^2 .

    """
    def __init__(self, num_paths, params, a_list, b_list, a_endpoints, b_endpoints):
        self.params = params
        self.a_list = a_list
        self.b_list = b_list
        self.a_endpoints = a_endpoints
        self.b_endpoints = b_endpoints
        self.num_paths = num_paths
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
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
        noise_sigma = torch.sqrt(2 * gamma / torch.tensor(self.params['beta']))  # noise standard deviation

        protocol_a_value = piecewise_protocol_value(num_steps, self.a_list, self.a_endpoints)
        protocol_b_value = piecewise_protocol_value(num_steps, self.b_list, self.b_endpoints)
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

class grad_calc:
    def __init__(self, sim_params, protocol_params, noise_array, potential_array, potential_advance_array, potential_grad_array, potential_grad_advance_array, drift_grad_array):
        self.sim_params = sim_params
        self.protocol_params = protocol_params
        self.noise_array = noise_array
        self.potential_array = potential_array
        self.potential_advance_array = potential_advance_array
        self.potential_grad_array = potential_grad_array
        self.potential_grad_advance_array = potential_grad_advance_array
        self.drift_grad_array = drift_grad_array
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    

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
    
        potential = self.potential_array[:,:-1]
        potential_advance = self.potential_advance_array
        # Calculate the work done along the trajectory
        work = (potential_advance - potential)
        return work
    
    def protocol_grad(self):
        """
        Compute gradient of protocol (length num_steps + 1) w.r.t. each coefficient.

        Anchor indices:
        - index 0        → endpoints[0]
        - index num_steps → endpoints[1]
        - coefficients placed evenly between 1 and num_steps - 1 (inclusive)

        Returns:
            torch.Tensor: shape (m, num_steps + 1)
        """
        m = len(self.protocol_params)
        num_steps = self.sim_params['num_steps']
        # Anchor indices: [0, 1, ..., num_steps - 1, num_steps]
        internal_anchors = torch.linspace(1, num_steps - 1, m, device=self.torch_device).long()
        full_anchors = torch.cat([
            torch.tensor([0], device=self.torch_device),
            internal_anchors,
            torch.tensor([num_steps], device=self.torch_device)
        ])

        grad_array = torch.zeros((m, num_steps + 1), device=self.torch_device)

        for i in range(m):
            left = full_anchors[i].item()
            center = full_anchors[i + 1].item()
            right = full_anchors[i + 2].item()

            # left → center (inclusive)
            if center > left:
                x = torch.linspace(0, 1, center - left + 1, device=self.torch_device)
                grad_array[i, left:center + 1] = x

            # center → right (inclusive)
            if right > center:
                x = torch.linspace(1, 0, right - center + 1, device=self.torch_device)
                grad_array[i, center:right + 1] = x

        return grad_array
    
    def potential_grad_value(self):
        """
        Calculate the potential value grad with parameters in cn at position x and time t with output being same as position array.
        dV/dcn = potential_grad_array  * grad_protocol
        
        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        dv_da = self.potential_grad_array
        da_cn = 1*self.protocol_grad()
        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    def potential_grad_value_advance(self):
        """
        Calculate the potential value grad with parameters in cn at position x and time t+dt with output being same as position array.
        dV/dcn = x  * da_cn(t+dt)
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        dv_da = self.potential_grad_advance_array
        da_cn = self.protocol_grad()[:,1:]
        return torch.einsum('ik,jk->jik', dv_da, da_cn)
    
    def work_grad_value(self):
        potential_grad_value = self.potential_grad_value()[:,:,:-1]
        potential_grad_advance_value = self.potential_grad_value_advance()
        work_grad = (potential_grad_advance_value - potential_grad_value)
        return work_grad
    
    def drift_grad_value(self):
        """
    Calculate the drift gradient at position x and time t with output being same as position array.
    
    
    Args:
        x (ndarray): Array of positions along the trajectory.
        coefficientlist (list): List of coefficients for the Fourier Series.
        dt (float): Time step size.

    Returns:
        ndarray: Array of drift gradients at each time step.
        Dimensions: (num_coefficients, num_paths, num_steps)
        """
    
        dphi_a = self.drift_grad_array
        da_cn = self.protocol_grad()
        dphi_cn = torch.einsum('ik,jk->jik', dphi_a, da_cn)
        return dphi_cn
    
    def malliavian_weight_array(self):
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
        sigma = self.sim_params['noise_sigma']
        df = self.drift_grad_value()[:,:,:-1] # chopping the last step
        mw = df * self.noise_array / (sigma ** 2)
        return mw
    #######needed to change
    def work_grad(self):
        dwp =self.work_grad_value().sum(axis=2).mean(axis=1)
        w = self.work_array().sum(axis=1)
        mw = self.malliavian_weight_array().sum(axis=2)
        dpw = (w * mw).mean(axis=1)
        return dwp + dpw
    
    def current_grad(self, current):
        malliavian_weight = self.malliavian_weight_array().sum(axis=2)
        x_mean_grad = (current * malliavian_weight).mean(axis=-1)
        return x_mean_grad
    

class bit_flip:
    ### THe potential is V(x) = a * x^4 - b * x^2 with flipping bits
    def __init__(self, params, phase_data, a_list, b_list, a_endpoints, b_endpoints):
        self.params = params
        self.phase_data = phase_data
        self.num_steps = phase_data.shape[1] - 1  # Adjust for zero-based indexing
        self.num_paths = phase_data.shape[0]
        self.a_list = a_list
        self.b_list = b_list
        self.a_endpoints = a_endpoints
        self.b_endpoints = b_endpoints
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    def drift_grad_a_array(self):
        """
    Calculate the drift gradient at position x and time t with output being same as position array.
    dphi/da = -4 * x^3 
    
    Args:
        x (ndarray): Array of positions along the trajectory.
        coefficientlist (list): List of coefficients for the Fourier Series.
        dt (float): Time step size.

    Returns:
        ndarray: Array of drift gradients at each time step.
        Dimensions: (num_paths, num_steps)
        """
    
        dphi_a = -4*self.phase_data[:,:,0]**3
        return dphi_a
    
    def drift_grad_b_array(self):
        """
        Calculate the drift gradient at position x and time t with output being same as position array.
        dphi/db = 2 * x 
        db_dcn = delta shape functions.
        
        Args:
            x (ndarray): Array of phase data along the trajectory. idx of x is [path_index, step_index, 0] for position and [path_index, step_index, 1] for velocity.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of drift gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        
    
        dphi_b = 2*self.phase_data[:,:,0]
        return dphi_b
    
    def potential_value_array(self):
        """
        Calculate the potential value at phase position x and time t with output being same as position array.
        x[:,:,0] is the position at time t, and x[:,:,1] is the velocity at time t.
        """
        protocol_a= piecewise_protocol_value(self.num_steps,self.a_list, self.a_endpoints)
        protocol_b= piecewise_protocol_value(self.num_steps,self.b_list, self.b_endpoints)
        potential = protocol_a * self.phase_data[:,:,0]**4 -  protocol_b * self.phase_data[:,:,0]**2
        return potential
    
    def potential_value_advance_array(self):
        """
        Calculate the potential value at position x and time t+dt with output being same as position array but chopped the last column.


        Returns:
            ndarray: Array of advance potential values.
            Dimensions: (num_paths, num_steps - 1)
        """
    
        protocol_a_advance = piecewise_protocol_value(self.num_steps,self.a_list, self.a_endpoints)[1:]
        protocol_b_advance = piecewise_protocol_value(self.num_steps,self.b_list, self.b_endpoints)[1:]
        potential_advance = protocol_a_advance*self.phase_data[:,:-1,0]**4 - protocol_b_advance * self.phase_data[:,:-1,0]**2 
        return potential_advance
    
    def potential_grad_a_value_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t with output being same as position array.
        dV/dcn = x  * -da_cn
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
        dv_da = self.phase_data[:,:,0]**4 
        return dv_da
    
    def potential_grad_a_value_advance_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t+dt with output being same as position array.
        dV/dcn = x  * da_cn(t+dt)
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients for the Fourier Series.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps)
        """
       
        dv_da = self.phase_data[:,:-1,0]**4 
       

        return dv_da
    
    def potential_grad_b_value_array(self):
        """
        Calculate the potential value grad with parameters in $b$ (called dn) at position x and time t with output being same as position array.
        dV/dcn =  -x^2
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients of middle points.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps-1)
        """
        
        dv_db = -1*self.phase_data[:,:,0]**2 
        return dv_db

    
    def potential_grad_b_value_advance_array(self):
        """
        Calculate the potential value grad with parameters in $a$ (called cn) at position x and time t+dt with output being same as position array.
        dV/dcn = x  * -db_dn(t+dt)
        
        Args:
            x (ndarray): Array of positions along the trajectory.
            coefficientlist (list): List of coefficients of middle points.
            dt (float): Time step size.

        Returns:
            ndarray: Array of potential gradients at each time step.
            Dimensions: (num_coefficients, num_paths, num_steps-1)
        """
        dv_db = -1*self.phase_data[:,:-1,0]**2 
        return dv_db
    
class bit_erasure:
    """
    The potential is V(x) = a * x^4 - b * x^2 + c * x
    The potential gradients are d_a V = x^4, d_b V = -x^2, d_c V = x
    The drift is F = -d_x V = -4 * a * x^3 + 2 * b * x - c
    The drift gradients are d_a F = -4 * x^3, d_b F = 2 * x, d_c F = -1
    """
  
    def __init__(self, params, phase_data, a_list, b_list, c_list, a_endpoints, b_endpoints, c_endpoints):
        self.params = params
        self.phase_data = phase_data
        self.num_steps = phase_data.shape[1] - 1  # Adjust for zero-based indexing
        self.num_paths = phase_data.shape[0]
        self.a_list = a_list
        self.b_list = b_list
        self.c_list = c_list
        self.a_endpoints = a_endpoints
        self.b_endpoints = b_endpoints
        self.c_endpoints = c_endpoints
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.protocol_a = piecewise_protocol_value(self.num_steps,self.a_list, self.a_endpoints)
        self.protocol_b = piecewise_protocol_value(self.num_steps,self.b_list, self.b_endpoints)
        self.protocol_c = piecewise_protocol_value(self.num_steps,self.c_list, self.c_endpoints)
        self.protocol_a_advance = self.protocol_a[1:]
        self.protocol_b_advance = self.protocol_b[1:]
        self.protocol_c_advance = self.protocol_c[1:]

    def potential_value_array(self):
        """
        Calculate the potential value at phase position x and time t with output being same as position array.
        x[:,:,0] is the position at time t, and x[:,:,1] is the velocity at time t.
        """

        potential = self.protocol_a * self.phase_data[:,:,0]**4 -  self.protocol_b * self.phase_data[:,:,0]**2 + self.protocol_c * self.phase_data[:,:,0]
        return potential

    def potential_value_advance_array(self):
        """
        Calculate the potential value at position x and time t+dt with output being same as position array but chopped the last column.


        Returns:
            ndarray: Array of advance potential values.
            Dimensions: (num_paths, num_steps - 1)
        """

        potential_advance = self.protocol_a_advance*self.phase_data[:,:-1,0]**4 - self.protocol_b_advance * self.phase_data[:,:-1,0]**2 + self.protocol_c_advance * self.phase_data[:,:-1,0]
        return potential_advance
    
    def drift_grad_a_array(self):
        """
        Calculate the drift gradient at position x and time t with output being same as position array.
        dF/da = -4 * x^3 
        """
        dF_da = -4*self.phase_data[:,:,0]**3
        return dF_da
    def drift_grad_b_array(self):
        """
        Calculate the drift gradient at position x and time t with output being same as position array.
        dF/db = 2 * x
        """
        dF_db = 2*self.phase_data[:,:,0]
        return dF_db
    def drift_grad_c_array(self):
        """
        Calculate the drift gradient at position x and time t with output being same as position array.
        dF/dc = -1
        """
        dF_dc = -1 * torch.ones_like(self.phase_data[:,:,0])
        return dF_dc
    
    def potential_grad_a_value_array(self):
        """
        Calculate the potential gradient at position x and time t with output being same as position array.
        dV/da = x^4
        """
        dV_da = self.phase_data[:,:,0]**4
        return dV_da
    def potential_grad_b_value_array(self):
        """
        Calculate the potential gradient at position x and time t with output being same as position array.
        dV/db = -x^2
        """
        dV_db = -self.phase_data[:,:,0]**2
        return dV_db
    def potential_grad_c_value_array(self):
        """
        Calculate the potential gradient at position x and time t with output being same as position array.
        dV/dc = x
        """
        dV_dc = self.phase_data[:,:,0]
        return dV_dc
    def potential_grad_a_value_advance_array(self):
        """
        Calculate the potential gradient at position x and time t+dt with output being same as position array but chopped the last column.
        dV/da = x^4
        """
        dV_da_advance = self.phase_data[:,:-1,0]**4
        return dV_da_advance
    def potential_grad_b_value_advance_array(self):
        """
        Calculate the potential gradient at position x and time t+dt with output being same as position array but chopped the last column.
        dV/db = -x^2
        """
        dV_db_advance = -self.phase_data[:,:-1,0]**2
        return dV_db_advance
    def potential_grad_c_value_advance_array(self):
        """
        Calculate the potential gradient at position x and time t+dt with output being same as position array but chopped the last column.
        dV/dc = x
        """
        dV_dc_advance = self.phase_data[:,:-1,0]
        return dV_dc_advance