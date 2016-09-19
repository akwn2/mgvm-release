function self = get_moments_q_kappa(self, getGradients)
    if getGradients
        [self.m_kappa, self.g_kappa_eta, self.g_kappa_beta0] = ...
            bexpMean(self.q_kappa_eta, self.q_kappa_beta0);
    else
        self.m_kappa = bexpMean(self.q_kappa_eta, ...
                                self.q_kappa_beta0);
    end
end
