function [mc, ms, mc2, ms2, msc, dmc, dms, dmc2, dms2, dmsc] = ...
            moments_von_mises(m1, k1)
        
    % Precalculate the bessels functions
    ive0 = ive(0, k1);
    ive1 = ive(1, k1);
    ive2 = ive(2, k1);

    % Update moments usg a von Mises
    mc = real(ive1 ./ ive0 .* exp(+1.i .* m1));

    ms = imag(ive1 ./ ive0 .* exp(+1.i .* m1));

    mc2 = 0.5 .* (1 + real(ive2 ./ ive0 .* exp(2.i .* m1)));
    
    msc = 0.5 .* imag(ive2 ./ ive0 .* exp(2.i .* m1));

    ms2 = 0.5 .* (1 - real(ive2 ./ ive0 .* exp(2.i .* m1)));

    if nargout > 5

        ive3 = ive(3, k1);

        aux_1 = 0.50 .* (ive0 + ive2)./ ive0 - (ive1 ./ ive0) .^ 2;
        aux_2 = 0.25 .* (ive1 + ive3)./ ive0 ...
              - 0.50 .* ive1 .* ive2 ./ ive0 .^ 2;
        aux_3 = ive1 ./ ive0;
        aux_4 = ive2 ./ ive0;

        c_m1 = cos(m1);
        s_m1 = sin(m1);
        s_2m1 = sin(2 .* m1);
        c_2m1 = cos(2 .* m1);

        ds_k1 = +s_m1 .* aux_1;
        dc_k1 = +c_m1 .* aux_1;
        ds2_k1 = -c_2m1 .* aux_2;
        dsc_k1 = +s_2m1 .* aux_2;
        dc2_k1 = +c_2m1 .* aux_2;

        ds_m1 = +c_m1 .* aux_3;
        dc_m1 = -s_m1 .* aux_3;
        ds2_m1 = +s_2m1 .* aux_4;
        dsc_m1 = +c_2m1 .* aux_4;
        dc2_m1 = -s_2m1 .* aux_4;
        
        dmc = {dc_m1, dc_k1};
        dms = {ds_m1, ds_k1};
        dmc2 = {dc2_m1, dc2_k1};
        dms2 = {ds2_m1, ds2_k1};
        dmsc = {dsc_m1, dsc_k1};
    end