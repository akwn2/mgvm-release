function nFq = obj_free_energy_mvm(self, vararray)
% obj_free_energy_mvm
% Negative free energy function for the free energy maximisation under the
% mvm model.

    % Unpack
    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % Update moments of all variational distributions used
    self = self.get_moments_q_phi(false);

    % Calculate free energy terms
    nlogP = self.obj_log_joint_mvm(vararray);
    nH = self.obj_entropy_gp(vararray);

    % Output free energy scaled by the dataset size
    nFq = (nH + nlogP) ./ self.N;

end