function nlogP = obj_log_joint_point_ts(self, vararray)
% obj_log_joint_point_ts
% Negative average log-joint function for free energy maximisation under 
% the point-estimate model.

    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    if self.set_matrices_tril2
        self.p_A.xv = tril2(self.p_A.xv);
        self.p_B.xv = tril2(self.p_B.xv);
    end

    % PRIOR TERMS
    %-------------------------------------------------
    AA = self.p_A.xv' * self.p_A.xv;
    BB = self.p_B.xv' * self.p_B.xv;
 
    prior_A = - 0.5 * trace(AA * diag(self.p_alph2.xv));

    prior_B = - 0.5 * trace(BB * diag(self.p_beta2.xv));
          
    cos_phi_lag0 = self.m_cos_phi.xv(2:end);
    sin_phi_lag0 = self.m_sin_phi.xv(2:end);
    
    cos_phi_lag1 = self.m_cos_phi.xv(1:end-1);
    sin_phi_lag1 = self.m_sin_phi.xv(1:end-1);

    prior_phi = - self.N * sum(log(2 * pi) + log_iv(0, self.p_kappa.xv))...
                + sum(self.p_kappa.xv(2:end)' * ...
                        (cos_phi_lag0 .* cos_phi_lag1 + ...
                         sin_phi_lag0 .* sin_phi_lag1)) ...
                + self.p_kappa_xv(1) .* self.m_cos_phi.xv(1);

    prior_omega = - log(2 * pi) + log_iv(0, self.p_kappa_omega.xv) + ...
                  sum(self.p_kappa_omega.xv' * ...
                        (self.m_cos_omega.xv .* self.p_cos_nu.xv + ...
                         self.m_sin_omega.xv .* sqrt(1 - self.p_cos_nu.xv^2)));
              
    % LIKELIHOOD TERM
    %------------------------------------------
    ccT = (self.m_cos_phi.xv * self.m_cos_phi.xv') ...
        + diag(- diag(self.m_cos_phi.xv * self.m_cos_phi.xv') ...
               + sum(self.m_cos2_phi.xv, 2));

    scT = (self.m_sin_phi.xv * self.m_cos_phi.xv') ...
        + diag(- diag(self.m_sin_phi.xv * self.m_cos_phi.xv') ...
               + sum(self.m_sincos_phi.xv, 2));

    ssT = (self.m_sin_phi.xv * self.m_sin_phi.xv') ...
        + diag(- diag(self.m_sin_phi.xv * self.m_sin_phi.xv') ...
               + sum(self.m_sin2_phi.xv, 2));

           
	% Get the sparse matrices from kroencker products (hopefully sparse
	% implementation will avoid possible memory problems)
    C = kron([self.p_A.xv; self.p_B.xv], sparse(eye(T)));
    
    R = sparse_rot(self.p_omega.xv, self.D);
    
    % Interweave sine and cosine to facilitate product formulas, hopefully
    % will not cause memory problems.
    x = zeros(2 .* self.D * self.N, 1);
    x(1:self.D:end) = self.m_cos_phi.xv(:);
    x(self.D:self.D:end) = self.m_sin_phi.xv(:);
    
    xxT = zeros(2 .* self.D .* self.N,2 .* self.D .* self.N);
    xxT(1:2 * self.D:end, 1:2 * self.D:end) = ccT;
    xxT(1:2 * self.D:end, self.D:2 * self.D:end) = scT';
    xxT(self.D:2 * self.D:end, 1:2 * self.D:end) = scT;
    xxT(self.D:2 * self.D:end, self.D:2 * self.D:end) = ssT;
     
    modelPred = C * R * x;

    % Norm within the Likelihood term
    normTerm = self.trYYT - 2 * trace(modelPred * self.Y') + ...
               trace(C * C' * xxT);

    % Likelihood term
    likelihood = self.N * self.M / 2 * log(self.p_prc2.xv / (2 * pi))...
                - 0.5 * self.p_prc2.xv * normTerm;

    % AGREGATE TERMS FOR AVERAGE LOG JOINT
    %-----------------------------------------
    logP = likelihood + prior_phi + prior_omega + prior_A + prior_B;

    nlogP = -logP;
end
