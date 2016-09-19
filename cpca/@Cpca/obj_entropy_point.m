function nH = obj_entropy_point(self, vararray)
% obj_entropy_point
% Negative entropy function for the free energy maximisation under the
% point-estimation model.

    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    self = self.get_entropy_q_phi(false);

    nH = - self.h_q_phi;
end