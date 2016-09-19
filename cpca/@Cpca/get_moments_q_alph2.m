function self = get_moments_q_alph2(self, get_gradients)
% get_moments_q_alph2
% Calculates the moments and the derivatives for the alph2 ARD parameter

    if get_gradients
        
        [m, dm_a, dm_b] = moments_gamma(self.q_alph2_a.xv, ...
                                        self.q_alph2_b.xv);
        self.m_alph2.xv = m;                           
        self.m_alph2.dx = {dm_a, dm_b};
    else
        self.m_alph2.xv = moments_gamma(self.q_alph2_a.xv, ...
                                        self.q_alph2_b.xv);
    end
end
