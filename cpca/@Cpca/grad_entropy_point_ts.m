function ndH = grad_entropy_point_ts(self, vararray)
% grad_entropy_point_ts
% gradient of the entropy for the point-estimate model.

    % Unpack
    self = self.unpack_from_array(vararray,'variables', 'Fq');

    % Gradients w.r.t. latent angles
    %-----------------------------------------
    self = self.get_entropy_q_phi(true);

%     self.q_phi_k1.dx = self.g_h_phi_k1;
%     self.q_phi_k2.dx = self.g_h_phi_k2;
%     self.q_phi_m1.dx = self.g_h_phi_m1;
%     self.q_phi_m2.dx = self.g_h_phi_m2;

    ndH = - self.pack_as_array('gradients', 'Fq');
    
    % assert absence of crude numerical errors
    assert_real(ndH);
end
