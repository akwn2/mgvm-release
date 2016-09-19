function dnFq = grad_free_energy_bayes(self, vararray)
% grad_free_energy_bayes
% Calculates the gradient of the (negative) free energy objective
% function for the variational bayes model.

    % Unpack values
    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % Update moments of all variational distributions used
    self = self.get_moments_q_alph2(true);
    self = self.get_moments_q_beta2(true);
    self = self.get_moments_q_prc2(true);
    self = self.get_moments_q_phi(true);
    self = self.get_moments_q_AB(true);

    % Compute log joint gradients
    %---------------------------------------------------
    self = self.clear_stored_grad();
    nlogP_grad = self.grad_log_joint_bayes(vararray);

    % Compute and store entropy gradients
    %---------------------------------------------------
    self = self.clear_stored_grad();
    nH_grad = self.grad_entropy_bayes(vararray);
    
    % Compute and store free energy gradients
    %---------------------------------------------------
    dnFq = (nlogP_grad + nH_grad) ./ self.N;
end  