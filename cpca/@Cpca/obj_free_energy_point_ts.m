function nFq = obj_free_energy_point_ts(self, vararray)
% obj_free_energy_point_ts
% Negative free energy function for the free energy maximisation under the
% point-estimate model.

    % Unpack
    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % Update trigonometric moments and entropy
    self = self.get_moments_q_phi(false);

    nlogP = self.obj_log_joint_point(vararray);
    nH = self.obj_entropy_point(vararray);

    % Output free energy scaled by the dataset size
    nFq = (nH + nlogP) ./ self.N;
end
