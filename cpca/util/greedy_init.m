function model = greedy_init(Y_train, D, options)
% INIT_GREEDY_AB Initialise the A and B matrices using a greedy approach.

    [M, N] = size(Y_train);
    [A, B] = build_structure(Y_train, D, D+2);
        
    M_ini = 1;
    D_ini = 1;
    D_end = D;
    N_ini = 1;
    N_end = N;
    
    for mm = 1:floor(M / 2)
        
        M_end = 2 * mm;
        
        model = Cpca(Y_train, D, options);
        
        model = model.preset_opt_free_init(M_ini, M_end,...
                                           D_ini, D_end,...
                                           N_ini, N_end);
                                              
        model.p_A.xv(M_ini:M_end, D_ini:D_end) = ...
            A(M_ini:M_end, D_ini:D_end);
        
        model.p_B.xv(M_ini:M_end, D_ini:D_end) = ...
            B(M_ini:M_end, D_ini:D_end);
        
        model = model.optimize_free_energy();
        
        A(M_ini:M_end, D_ini:D_end) = ...
            model.p_A.xv(M_ini:M_end, D_ini:D_end);
        
        B(M_ini:M_end, D_ini:D_end) = ...
            model.p_B.xv(M_ini:M_end, D_ini:D_end);
    end
%     model = model.preset_opt_free_point_vm();
end