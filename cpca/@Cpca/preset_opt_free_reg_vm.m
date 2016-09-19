function self = preset_opt_free_reg_vm(self)
% preset_opt_free_reg_vm
% Sets the mgvm for the regression model with von Mises mean field.

    self.q_phi_type = 'von Mises';

    % Model parameters
    %--------------------------------------------------------
    self.p_kappa.xv = rand(self.D, 1);
    
    % To enforce symmetric random matrices
    rand_mat1 = randn(self.D);
    rand_mat2 = randn(self.D);
    
    self.p_Wcc.xv = randn() .* eye(self.D) + rand_mat1 * rand_mat1';
    self.p_Wcs.xv = randn(self.D);
    self.p_Wss.xv = randn() .* eye(self.D) + rand_mat2 * rand_mat2';

    % Model parameter gradients
    %--------------------------------------------------------
    self.p_kappa.dx = rand(size(self.p_kappa.xv));

    % Model parameter lower bounds
    %--------------------------------------------------------
%     self.p_kappa.lb = 1E-6 * ones(size(self.p_kappa.xv));
    self.p_kappa.lb = -inf * ones(size(self.p_kappa.xv));
    self.p_kappa.et = true;
    

    % Model parameter upper bounds
    %--------------------------------------------------------
    % Prior for latent angles
    self.p_kappa.ub = inf * ones(size(self.p_kappa.xv));

    % Mean field parameters
    %--------------------------------------------------------
    % Latent angles
    self.q_phi_k1.xv = rand(self.D, self.N);
    self.q_phi_m1.xv = rand(self.D, self.N);

    % Mean field parameters gradients
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));

    % Mean field parameter lower bounds
    %--------------------------------------------------------

    % Latent angles
%     self.q_phi_k1.lb = 1e-6 * ones(size(self.q_phi_k1.xv));
    self.q_phi_k1.lb = -inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_k1.et = true;
    
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));

    % Mean field parameter upper bounds
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.ub = +inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_m1.ub = +pi * ones(size(self.q_phi_m1.xv));

    
    % Internal parameters
    %--------------------------------------------------------
    
    % Variables
    self.m_sin_phi.xv = rand(self.D, self.N);
    self.m_cos_phi.xv = rand(self.D, self.N);
    self.m_sin2_phi.xv = rand(self.D, self.N);
    self.m_sincos_phi.xv = rand(self.D, self.N);
    self.m_cos2_phi.xv = rand(self.D, self.N);
    
    % Functions
    self.obj_free = @self.obj_free_energy_reg;
    self.grad_free = @self.grad_free_energy_reg;
end