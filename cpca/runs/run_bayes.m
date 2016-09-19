function run_bayes(dataset, D, noise, seed, saveme)
% run_bayes.m
% Function to standardise the procedure for running tests with different
% models
    if nargin < 5
        saveme = true;
    end
    
    fix_seed(seed);
    
    case_name = [dataset, '_bayes_', num2str(D), '_',...
                 num2str(noise),'_',num2str(seed)];
    
    fprintf('---------------------------------\n');
    fprintf('      STARTED RUNNING CASE       \n');
    fprintf('---------------------------------\n');

    fprintf('Loading data... ');
    % Initialising with point case
    base_case = [dataset, '_point_', num2str(D), '_',...
                 num2str(noise),'_',num2str(seed),'.mat'];
             
    point = load(base_case);
    fprintf('Ok!\n');
        
    % Initialise
    solver = 'C-BFGS';
%     solver = 'IPOPT';
    options = {...
        {'name', case_name}, ...
        {'opt_max_iter', 5E3}, ...
        {'opt_tol', 1E-4}, ...
        {'opt_solver', solver}, ...
        };
        
%     keyboard;
    model_train = Cpca(point.model_train.Y, D, options);
    
%     model_init = model_init.preset_opt_free_bayes_vm_exp();
    model_train = model_train.preset_opt_free_bayes_vm_ctrs();
    
    model_train.p_kappa = point.model_train.p_kappa;
    model_train.p_a_alph2.xv = 1E-3 .* ones(size(model_train.p_kappa.xv));
    model_train.p_a_beta2.xv = 1E-3 .* ones(size(model_train.p_kappa.xv));
    
    model_train.q_A_mu = point.model_train.p_A;
    model_train.q_B_mu = point.model_train.p_B;
    model_train.q_prc2_a = point.model_train.p_prc2;
    
    model_train.q_alph2_a.xv = 1E-3 .* ones(size(model_train.p_kappa.xv));
    model_train.q_beta2_a.xv = 1E-3 .* ones(size(model_train.p_kappa.xv));
    
    model_train.m_A = point.model_train.p_A;
    model_train.m_B = point.model_train.p_B;
    model_train.m_prc2 = point.model_train.p_prc2;
    
    model_train.q_phi_k1 = point.model_train.q_phi_k1;
    model_train.q_phi_m1 = point.model_train.q_phi_m1;
 
    model_train = model_train.get_moments_q_phi(false);
    
    % cross validate
%     Y_pred = model_train.q_A_mu.xv * model_train.m_cos_phi.xv + ...
%              model_train.q_B_mu.xv * model_train.m_sin_phi.xv;
%     
%     hold on
%     plot(point.Y_train(:,1:2:end), point.Y_train(:,2:2:end),'o');
%     plot(Y_pred(1:2:end,:)', Y_pred(2:2:end,:)','x');
%     keyboard
    
    % Fit
    model_train = model_train.set_var_bayes_vm();
    model_train = model_train.optimize_free_energy();
    model_train = model_train.get_moments_q_phi(false);
    
    % cross validate
%     Y_pred = model_train.q_A_mu.xv * model_train.m_cos_phi.xv + ...
%              model_train.q_B_mu.xv * model_train.m_sin_phi.xv;
%     
%     hold on
%     plot(point.Y_train(:,1:2:end), point.Y_train(:,2:2:end),'o');
%     plot(Y_pred(1:2:end,:)', Y_pred(2:2:end,:)','x');
%     keyboard
         
    % Testing by denoising
    options = {...
        {'name', case_name}, ...
        {'opt_max_iter', 2E3}, ...
        {'opt_tol', 1E-4}, ...
        {'opt_solver', 'C-BFGS'}, ...
        };
    
    model_test = Cpca(point.model_test.Y, D, options);
    model_test = model_test.preset_opt_free_point_vm();

    model_test.p_A = model_train.q_A_mu;
    model_test.p_B = model_train.q_B_mu;
    model_test.p_kappa = model_train.p_kappa;
    model_test.p_prc2 = model_train.m_prc2;
    
    model_test.q_phi_k1 = point.model_test.q_phi_k1;
    model_test.q_phi_m1 = point.model_test.q_phi_m1;

    model_test = model_test.set_var_point_vm_mf();
    model_test = model_test.optimize_free_energy();
    model_test = model_test.get_moments_q_phi(false);

%     keyboard;
    Y_pred = model_train.q_A_mu.xv * model_test.m_cos_phi.xv + ...
             model_train.q_B_mu.xv * model_test.m_sin_phi.xv;

    % cross validate
%     hold on
%     plot(point.Y_test(:,1:2:end), point.Y_test(:,2:2:end),'o');
%     plot(Y_pred(1:2:end,:)', Y_pred(2:2:end,:)','x');
         
    bayes_rmse = prmse(point.Y_test, Y_pred');
    bayes_snr = psnr(point.Y_test, Y_pred');
    
    fprintf('Bayes CPCA:\n')
    fprintf(' RMSE: %1.4e\n', bayes_rmse);
    fprintf(' SNR:  %1.4e\n', bayes_snr);
    
%     keyboard;
    if saveme
        save([case_name, '.mat'],'model_train','model_test',...
             'bayes_rmse','bayes_snr');
    end
    fprintf('---------------------------------\n');
    fprintf('     FINISHED RUNNING CASE       \n');
    fprintf('---------------------------------\n');
end