function ndH = grad_entropy_bayes(self, vararray)
% grad_entropy_bayes
% gradient of the entropy for the variational Bayes model.

    % Unpack
    self = self.unpack_from_array(vararray,'variables', 'Fq');

    % Gradients w.r.t. alph2
    %-----------------------------------------
    self = self.get_entropy_q_alph2(true);

    % Gradients w.r.t. beta2
    %-----------------------------------------
    self = self.get_entropy_q_beta2(true);

    % Gradients w.r.t. prc2
    %-----------------------------------------
    self = self.get_entropy_q_prc2(true);

    % Gradients w.r.t. latent angles
    %-----------------------------------------
    self = self.get_entropy_q_phi(true);

    % Gradients w.r.t. coefficient matrices and offset
    %-----------------------------------------
    self = self.get_entropy_q_AB(true);

    ndH = - self.pack_as_array('gradients', 'Fq');

    % assert absence of crude numerical errors
    assert_real(ndH);
end