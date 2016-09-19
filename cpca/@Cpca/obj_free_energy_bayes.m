function nFq = obj_free_energy_bayes(self, vararray)
% obj_free_energy_bayes
% Negative free energy function for the free energy maximisation under the
% variational Bayes model.

    % Unpack
    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % Update moments of all variational distributions used
    self = self.get_moments_q_alph2(false);
    self = self.get_moments_q_beta2(false);
    self = self.get_moments_q_prc2(false);
    self = self.get_moments_q_phi(false);
    self = self.get_moments_q_AB(false);

    % Calculate free energy terms
    nlogP = self.obj_log_joint_bayes(vararray);
    nH = self.obj_entropy_bayes(vararray);

    % Output free energy scaled by the dataset size
    nFq = (nH + nlogP) ./ self.N;

end