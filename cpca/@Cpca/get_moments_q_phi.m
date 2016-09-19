function self = get_moments_q_phi(self, getGradients)
% get_moments_q_phi
% Updates the moments of the latent angles under
% the mean field distribution.

    if strcmp(self.q_phi_type, 'von Mises')
        
        if getGradients
            
            [mc, ms, mc2, ms2, msc, dmc, dms, dmc2, dms2, dmsc] = ...
                moments_von_mises(self.q_phi_m1.xv, self.q_phi_k1.xv);

            self.m_cos_phi.xv = mc;
            self.m_sin_phi.xv = ms;
            self.m_cos2_phi.xv = mc2;
            self.m_sin2_phi.xv = ms2;
            self.m_sincos_phi.xv = msc;
            
            % Trigonometric functions derivatives
            
            % m1 Section
            self.g_cos_phi_m1 = dmc{1};
            self.g_sin_phi_m1 = dms{1};
            self.g_cos2_phi_m1 = dmc2{1};
            self.g_sin2_phi_m1 = dms2{1};
            self.g_sincos_phi_m1 = dmsc{1};
            
            % k1 Section
            self.g_cos_phi_k1 = dmc{2};
            self.g_sin_phi_k1 = dms{2};
            self.g_cos2_phi_k1 = dmc2{2};
            self.g_sin2_phi_k1 = dms2{2};
            self.g_sincos_phi_k1 = dmsc{2};
            
        else
            
            [mc, ms, mc2, ms2, msc] = ...
                moments_von_mises(self.q_phi_m1.xv, self.q_phi_k1.xv);

            self.m_cos_phi.xv = mc;
            self.m_sin_phi.xv = ms;
            self.m_cos2_phi.xv = mc2;
            self.m_sin2_phi.xv = ms2;
            self.m_sincos_phi.xv = msc;
        end

    else %if strcmp(self.q_phi_type, 'GvM')

        if getGradients
            
            [mc, ms, mc2, ms2, msc, T, dmc, dms, dmc2, dms2, dmsc] = ...
                moments_gvm(self.q_phi_m1.xv, self.q_phi_m2.xv, ...
                            self.q_phi_k1.xv, self.q_phi_k2.xv);

            self.m_cos_phi.xv = mc;
            self.m_sin_phi.xv = ms;
            self.m_cos2_phi.xv = mc2;
            self.m_sin2_phi.xv = ms2;
            self.m_sincos_phi.xv = msc;
            
            self.T0 = T{1};
            self.T1 = T{2};
            self.T2 = T{3};
            self.T3 = T{4};
            self.T4 = T{5};
            
            % Trigonometric functions derivatives
            
            % m1 Section
            self.g_cos_phi_m1 = dmc{1};
            self.g_sin_phi_m1 = dms{1};
            self.g_cos2_phi_m1 = dmc2{1};
            self.g_sin2_phi_m1 = dms2{1};
            self.g_sincos_phi_m1 = dmsc{1};

            % m2 Section
            self.g_cos_phi_m2 = dmc{2};
            self.g_sin_phi_m2 = dms{2};
            self.g_cos2_phi_m2 = dmc2{2};
            self.g_sin2_phi_m2 = dms2{2};
            self.g_sincos_phi_m2 = dmsc{2};

            % k1 Section
            self.g_cos_phi_k1 = dmc{3};
            self.g_sin_phi_k1 = dms{3};
            self.g_cos2_phi_k1 = dmc2{3};
            self.g_sin2_phi_k1 = dms2{3};
            self.g_sincos_phi_k1 = dmsc{3};

            % k2 Section
            self.g_cos_phi_k2 = dmc{4};
            self.g_sin_phi_k2 = dms{4};
            self.g_cos2_phi_k2 = dmc2{4};
            self.g_sin2_phi_k2 = dms2{4};
            self.g_sincos_phi_k2 = dmsc{4};
            
        else
            
            [mc, ms, mc2, ms2, msc, T] = ...
                moments_gvm(self.q_phi_m1.xv, self.q_phi_m2.xv, ...
                            self.q_phi_k1.xv, self.q_phi_k2.xv);

            self.m_cos_phi.xv = mc;
            self.m_sin_phi.xv = ms;
            self.m_cos2_phi.xv = mc2;
            self.m_sin2_phi.xv = ms2;
            self.m_sincos_phi.xv = msc;
                        
            self.T0 = T{1};
            self.T1 = T{2};
            self.T2 = T{3};
        end
    end
end
