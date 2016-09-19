% dataset_list = {'synthetic'};
% dataset_list = {'running'};
% dataset_list = {'running', 'synthetic'};
% dataset_list = {'boxing','fishing'};
% dataset_list = {'boxing'};
dataset_list = {'fishing'};
% dataset_list = {'eeg'};
% solver = 'U-BFGS';
solver = 'C-BFGS';
% solver ='IPOPT';
D = 3;


for entry = 1:numel(dataset_list)    
    dataset = dataset_list{entry};
    fprintf(['Dataset for mocap_runs:', dataset,'\n']); 
    for noise = 7.5:2.5:15
        for seed = 0:10:20
            fprintf('noise = %2.1e \n',noise);
            fprintf('seed = %d \n',seed);
            
            run_point(dataset, D, noise, seed, solver);
            run_bayes(dataset, D, noise, seed);
            run_ppca(dataset, D, noise, seed);
            run_gplvm(dataset, D, noise, seed);
            run_gplvm_ppca(dataset, D, noise, seed);
            
        end
    end
end