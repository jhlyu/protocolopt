import torch
import gc
import matplotlib.pyplot as plt
import math

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
    ### The potential is V(x) = a * x^4 - b * x^2 with flipping bits
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
        self.protocol_a = piecewise_protocol_value(self.num_steps, self.a_list, self.a_endpoints)
        self.protocol_b = piecewise_protocol_value(self.num_steps, self.b_list, self.b_endpoints)
        self.protocol_c = piecewise_protocol_value(self.num_steps, self.c_list, self.c_endpoints)
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