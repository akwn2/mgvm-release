function self = get_moments_q_prc2(self, get_gradients)
% get_moments_q_prc2
% Calculates the moments and the derivatives for the model precision prc2

    if get_gradients
        
        [m, dm_a, dm_b] = moments_gamma(self.q_prc2_a.xv, ...
                                        self.q_prc2_b.xv);
        self.m_prc2.xv = m;
        self.m_prc2.dx = {dm_a, dm_b};
    else
        self.m_prc2.xv = moments_gamma(self.q_prc2_a.xv, ...
                                       self.q_prc2_b.xv);
    end
end