function nFq = obj_free_energy_reg(self, vararray)
% obj_free_energy_reg
% Negative free energy function for the free energy maximisation under the
% GP model.

    % Unpack
    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % Update moments of all variational distributions used
    self = self.get_moments_q_phi(false);

    % Calculate free energy terms
    nlogP = self.obj_log_joint_reg(vararray);
    nH = self.obj_entropy_reg(vararray);

    % Output free energy scaled by the dataset size
    nFq = (nH + nlogP) ./ self.N;

end