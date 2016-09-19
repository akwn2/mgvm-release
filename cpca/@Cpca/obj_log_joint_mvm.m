function nlogP = obj_log_joint_mvm(self, vararray)
% obj_log_joint_mvm
% Negative average log-joint function for free energy maximisation under 
% the mvM model.

    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % Pre-computable relations - Trigonometric terms - OK1
    ccT = self.m_cos_phi.xv * self.m_cos_phi.xv' ...
        + diag(- diag(self.m_cos_phi.xv * self.m_cos_phi.xv') ...
               + sum(self.m_cos2_phi.xv, 2));

    scT = self.m_sin_phi.xv * self.m_cos_phi.xv' ...
        + diag(- diag(self.m_sin_phi.xv * self.m_cos_phi.xv') ...
               + sum(self.m_sincos_phi.xv, 2));

    ssT = self.m_sin_phi.xv * self.m_sin_phi.xv' ...
        + diag(- diag(self.m_sin_phi.xv * self.m_sin_phi.xv') ...
               + sum(self.m_sin2_phi.xv, 2));


    % PRIOR TERMS
    %-------------------------------------------------

    Kcc = self.p_Ucc.xv' * self.p_Ucc.xv; %K(1:self.D, 1:self.D);
    Kcs = self.p_Ucc.xv' * self.p_Ucs.xv; %K(1:self.D, self.D+1:end);
    Kss = self.p_Ucs.xv' * self.p_Ucs.xv ...
        + self.p_Uss.xv' * self.p_Uss.xv; %K(self.D+1:end, self.D+1:end);
    
    logdetKinv = 0;%sum( log(diag(self.p_Ucc.xv) .^ -2) ...
                   % + log(diag(self.p_Uss.xv) .^ -2) );
    
    priors = - self.D * log(2 * pi) ...
             - 0.5 * logdetKinv ...
             - 0.5 * trace(Kss * ssT);

    % LIKELIHOOD TERM
    %------------------------------------------

    % Prior for latent angles - OK1
    likelihood = - self.N * log(2 * pi) ...
                 - self.N * sum(log_iv(0, self.p_kappa.xv)) ...
                 + sum(self.p_kappa.xv' * ...
                            ( self.m_cos_phi.xv .* self.cos_y ...
                            + self.m_sin_phi.xv .* self.sin_y) );


    % AGREGATE TERMS FOR AVERAGE LOG JOINT
    %-----------------------------------------
    logP = likelihood + priors;
    nlogP = -logP;

    % Checking for numerical errors
    assert_real(nlogP);
end