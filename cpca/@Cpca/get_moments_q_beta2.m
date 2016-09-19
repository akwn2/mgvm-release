function self = get_moments_q_beta2(self, get_gradients)
% get_moments_q_beta2
% Calculates the moments and the derivatives for the beta2 ARD parameter

    if get_gradients
        
        [m, dm_a, dm_b] = moments_gamma(self.q_beta2_a.xv, ...
                                        self.q_beta2_b.xv);
        self.m_beta2.xv = m;
        self.m_beta2.dx = {dm_a, dm_b};
    else
        self.m_beta2.xv = moments_gamma(self.q_beta2_a.xv, ...
                                        self.q_beta2_b.xv);
    end
end