function errcode = get_bayes_metrics(dataset, D, noise, seed, solver)
% run_point.m
% Function to standardise the procedure for running tests with different
% models

    fix_seed(seed);
    
    case_name = [dataset, '_bayes_', num2str(D), '_',...
                 num2str(noise),'_',num2str(seed)];
    
    load(['params_synthetic_point_',num2str(D), '_', ...
           num2str(noise), '_', num2str(seed),'.mat']);
       
    load(['params_synthetic_bayes_',num2str(D), '_', ...
           num2str(noise), '_', num2str(seed),'.mat']);

    fprintf('Ok!\n');
    
    options = {...
        {'name', case_name}, ...
        {'opt_max_iter', 5E3}, ...
        {'opt_tol', 1E-6}, ...
        {'opt_solver', solver}, ...
        };
    
    % Testing by denoising
    model_test = Cpca(Y_held, D, options);
    model_test = model_test.preset_opt_free_point_vm();

    model_test.p_A.xv = A;
    model_test.p_B.xv = B;
    model_test.p_kappa.xv = kappa;
    model_test.p_prc2.xv = prc2;

    model_test = model_test.preset_opt_free_point_vm_mf();
    model_test = model_test.optimize_free_energy();
    model_test = model_test.get_moments_q_phi(false);

    Y_pred = A * model_test.m_cos_phi.xv + ...
             B * model_test.m_sin_phi.xv;

    bayes_rmse = prmse(Y_test, Y_pred');
    bayes_snr = psnr(Y_test, Y_pred');
    
    fprintf('VB CPCA:\n')
    fprintf(' RMSE: %1.4e\n', bayes_rmse);
    fprintf(' SNR:  %1.4e\n', bayes_snr);
    
    save([case_name, '.mat'],'model_test','bayes_rmse','bayes_snr');
end