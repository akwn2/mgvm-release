function nlogP = obj_log_joint_point(self, vararray)
% obj_log_joint_point
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
    AB = self.p_A.xv' * self.p_B.xv;
 
%     prior_A = + 0.5 * self.M * sum(log(self.p_alph2.xv(:)) ) ...
%               - 0.5 * trace(AA * diag(self.p_alph2.xv));
% 
%     prior_B = + 0.5 * self.M * sum( log(self.p_beta2.xv(:)) ) ...
%               - 0.5 * trace(BB * diag(self.p_beta2.xv));

    prior_A = - 0.5 * trace(AA * diag(self.p_alph2.xv));

    prior_B = - 0.5 * trace(BB * diag(self.p_beta2.xv));
          
    aux = log(2 * pi) + log_iv(0, self.p_kappa.xv);

    prior_phi.xv = - self.N * sum(aux) ...
                + sum(self.p_kappa.xv' * self.m_cos_phi.xv);

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

    modelPred = self.p_A.xv * self.m_cos_phi.xv ...
              + self.p_B.xv * self.m_sin_phi.xv;

    % Norm within the Likelihood term
    normTerm = self.trYYT ... 
             - 2 * trace( modelPred * self.Y')...
             + trace(AA * ccT + 2 * AB * scT + BB * ssT);

    % Likelihood term
    likelihood = self.N * self.M / 2 * log(self.p_prc2.xv / (2 * pi))...
                - 0.5 * self.p_prc2.xv * normTerm;


    % AGREGATE TERMS FOR AVERAGE LOG JOINT
    %-----------------------------------------
    logP = likelihood + prior_phi.xv + prior_A + prior_B;

    nlogP = -logP;
end
