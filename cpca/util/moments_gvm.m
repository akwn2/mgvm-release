function [mc, ms, mc2, ms2, msc, T, dmc, dms, dmc2, dms2, dmsc] = ...
            moments_gvm(m1, m2, k1, k2)
% moments_gvm
% Moments of a Generalised von Mises distribution with parameters m1, m2,
% k1 and k2. The partial derivatives with respect to the parameters are
% also provided.

    weighting = true;
    n_pts = 1E6;

    [R, C] =  size(m1);

    T0 = gvm_euler(0, m1(:), m2(:), k1(:), k2(:), weighting, n_pts);
    T1 = gvm_euler(1, m1(:), m2(:), k1(:), k2(:), weighting, n_pts);
    T2 = gvm_euler(2, m1(:), m2(:), k1(:), k2(:), weighting, n_pts);

    T0 = reshape(T0, [R, C]);
    T1 = reshape(T1, [R, C]);
    T2 = reshape(T2, [R, C]);
    
    T = {T0, T1, T2};
    if nargout > 6
        T3 = gvm_euler(3, m1(:), m2(:), k1(:), k2(:), weighting, n_pts);
        T4 = gvm_euler(4, m1(:), m2(:), k1(:), k2(:), weighting, n_pts);
        
        T3 = reshape(T3, [R, C]);
        T4 = reshape(T4, [R, C]);
        
        T = {T0, T1, T2, T3, T4};
    end
    
    T1_re = real(T1);
    T2_re = real(T2);
    T1_im = imag(T1);
    T2_im = imag(T2);

    mc = T1_re ./ T0;
    ms = T1_im ./ T0;
    mc2 = 0.5 + 0.5 .* T2_re ./ T0;
    ms2 = 0.5 - 0.5 .* T2_re ./ T0;
    msc = 0.5 .* T2_im ./ T0;
    
    if nargout > 6

        % Conjugates of the first and second triongometric
        % moments
        conjT1 = conj(T1);
        conjT2 = conj(T2);

        e_p1im1 = exp(+1.i .* m1);
        e_m1im1 = exp(-1.i .* m1);

        e_p2im2 = exp(+2.i .* m2);
        e_m2im2 = exp(-2.i .* m2);

        dT0_k1 = +0.5 .* (e_m1im1 .* T1 + e_p1im1 .* conjT1);
        dT0_k2 = +0.5 .* (e_m2im2 .* T2 + e_p2im2 .* conjT2);
        dT0_m1 = -0.5i .* k1 .* (e_m1im1 .* T1 - e_p1im1 .* conjT1);
        dT0_m2 = -1.0i .* k2 .* (e_m2im2 .* T2 - e_p2im2 .* conjT2);

        % Derivatives of the first trigonometric moment
        dT1_k1 = +0.5 .* (e_m1im1 .* T2 + e_p1im1 .* T0);
        dT1_k2 = +0.5 .* (e_m2im2 .* T3 + e_p2im2 .* conjT1);
        dT1_m1 = -0.5i .* k1 .* (e_m1im1 .* T2 - e_p1im1 .* T0);
        dT1_m2 = -1.0i .* k2 .* (e_m2im2 .* T3 - e_p2im2 .* conjT1);

        % Derivatives of the second trigonometric moment
        dT2_k1 = +0.5 .* (e_m1im1 .* T3 + e_p1im1 .* T1);
        dT2_k2 = +0.5 .* (e_m2im2 .* T4 + e_p2im2 .* T0);
        dT2_m1 = -0.5i .* k1 .* (e_m1im1 .* T3 - e_p1im1 .* T1);
        dT2_m2 = -1.0i .* k2 .* (e_m2im2 .* T4 - e_p2im2 .* T0);

        % Trigonometric functions derivatives

        % k1 Section
        dc_k1 = real((T0 .* dT1_k1 - T1 .* dT0_k1) ./ (T0 .^ 2));
        ds_k1 = imag((T0 .* dT1_k1 - T1 .* dT0_k1) ./ (T0 .^ 2));
        dc2_k1 = +0.5 .* real((T0 .* dT2_k1 - T2 .* dT0_k1) ./ (T0 .^ 2));
        ds2_k1 = -0.5 .* real((T0 .* dT2_k1 - T2 .* dT0_k1) ./ (T0 .^ 2));
        dsc_k1 = +0.5 .* imag((T0 .* dT2_k1 - T2 .* dT0_k1) ./ (T0 .^ 2));

        % k2 Section
        dc_k2 = real((T0 .* dT1_k2 - T1 .* dT0_k2) ./ (T0 .^ 2));
        ds_k2 = imag((T0 .* dT1_k2 - T1 .* dT0_k2) ./ (T0 .^ 2));
        dc2_k2 = +0.5 .* real((T0 .* dT2_k2 - T2 .* dT0_k2) ./ (T0 .^ 2));
        ds2_k2 = -0.5 .* real((T0 .* dT2_k2 - T2 .* dT0_k2) ./ (T0 .^ 2));
        dsc_k2 = +0.5 .* imag((T0 .* dT2_k2 - T2 .* dT0_k2) ./ (T0 .^ 2));
        % m1 Section
        dc_m1 = real((T0 .* dT1_m1 - T1 .* dT0_m1) ./ (T0 .^ 2));
        ds_m1 = imag((T0 .* dT1_m1 - T1 .* dT0_m1) ./ (T0 .^ 2));
        dc2_m1 = +0.5 .* real((T0 .* dT2_m1 - T2 .* dT0_m1) ./ (T0 .^ 2));
        ds2_m1 = -0.5 .* real((T0 .* dT2_m1 - T2 .* dT0_m1) ./ (T0 .^ 2));
        dsc_m1 = +0.5 .* imag((T0 .* dT2_m1 - T2 .* dT0_m1) ./ (T0 .^ 2));

        % m2 Section
        dc_m2 = real((T0 .* dT1_m2 - T1 .* dT0_m2) ./ (T0 .^ 2));
        ds_m2 = imag((T0 .* dT1_m2 - T1 .* dT0_m2) ./ (T0 .^ 2));
        dc2_m2 = +0.5 .* real((T0 .* dT2_m2 - T2 .* dT0_m2) ./ (T0 .^ 2));
        ds2_m2 = -0.5 .* real((T0 .* dT2_m2 - T2 .* dT0_m2) ./ (T0 .^ 2));
        dsc_m2 = +0.5 .* imag((T0 .* dT2_m2 - T2 .* dT0_m2) ./ (T0 .^ 2));
        
        % packing
        dmc = {dc_m1, dc_m2, dc_k1, dc_k2};
        dms = {ds_m1, ds_m2, ds_k1, ds_k2};
        dmc2 = {dc2_m1, dc2_m2, dc2_k1, dc2_k2};
        dms2 = {ds2_m1, ds2_m2, ds2_k1, ds2_k2};
        dmsc = {dsc_m1, dsc_m2, dsc_k1, dsc_k2};
    end
end