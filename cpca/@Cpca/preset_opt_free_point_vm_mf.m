function self = preset_opt_free_point_vm_mf(self)
% preset_opt_free_point_vm_mf.m
% Sets the cpca to learn only the mean field under the point-estimate model
% using vM distributions for the latent angle mean field distributions.
    self.q_phi_type = 'von Mises';

    % Initialise the mean field random values
    %--------------------------------------------------------
    self.q_phi_k1.xv = rand(self.D, self.N);
    self.q_phi_m1.xv = rand(self.D, self.N);

    % Gradients
    %--------------------------------------------------------
    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));

    % Lower bounds
    %--------------------------------------------------------
    self.q_phi_k1.lb = zeros(size(self.q_phi_k1.xv));
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));

    % Upper bounds
    %--------------------------------------------------------
    self.q_phi_k1.ub = inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_m1.ub = pi * ones(size(self.q_phi_m1.xv));

    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = {...
                                    'q_phi_k1',...
                                    'q_phi_m1',...
                                    };

    self.name_map('Fq_subset_ini') = {...
                                      [1, 1],...
                                      [1, 1],...
                                     };

    self.name_map('Fq_subset_end') = {...
                                      [self.D, self.N],...
                                      [self.D, self.N],...
                                     };

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_point;
    self.grad_free = @self.grad_free_energy_point;
end