function self = preset_opt_free_point_gvm_mf(self)
% preset_opt_free_point_gvm_mf
% Preset function to prime the model to learn only the mean field over
% latent angles using gvm distributions

    self.q_phi_type = 'GvM';
    
    % Initialise the mean field random values
    %--------------------------------------------------------
    self.q_phi_k1.xv = rand(self.D, self.N);
    self.q_phi_k2.xv = rand(self.D, self.N);
    self.q_phi_m1.xv = rand(self.D, self.N);
    self.q_phi_m2.xv = rand(self.D, self.N);

    % Gradients
    %--------------------------------------------------------
    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_k2.dx = rand(size(self.q_phi_k2.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));
    self.q_phi_m2.dx = rand(size(self.q_phi_m2.xv));

    % Lower bounds
    %--------------------------------------------------------

    self.q_phi_k1.lb = zeros(size(self.q_phi_k1.xv));
    self.q_phi_k2.lb = zeros(size(self.q_phi_k2.xv));
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.lb = -pi / 2 * ones(size(self.q_phi_m2.xv));

    % Upper bounds
    %--------------------------------------------------------

    self.q_phi_k1.ub = inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_k2.ub = inf * ones(size(self.q_phi_k2.xv));
    self.q_phi_m1.ub = pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.ub = pi / 2 * ones(size(self.q_phi_m2.xv));

    % Moments
    self.T0 = rand(size(self.q_phi_k1.xv));
    self.T1 = rand(size(self.q_phi_k1.xv));
    self.T2 = rand(size(self.q_phi_k1.xv));
    self.T3 = rand(size(self.q_phi_k1.xv));
    self.T4 = rand(size(self.q_phi_k1.xv));

    % Add mean field parameters
    self.name_map('Fq_variables') = {'q_phi_k1',...
                                    'q_phi_k2',...
                                    'q_phi_m1',...
                                    'q_phi_m2'};

    self.name_map('Fq_subset_ini') = {[1, 1],...
                                    [1, 1],...
                                    [1, 1],...
                                    [1, 1]};

    self.name_map('Fq_subset_end') = {[self.D, self.N],...
                                    [self.D, self.N],...
                                    [self.D, self.N],...
                                    [self.D, self.N]};

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_point;
    self.grad_free = @self.grad_free_energy_point;
end
