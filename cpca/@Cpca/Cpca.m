classdef Cpca
%cpca Class that defines the Circular Principal Component Analysis model.
% The CPCA model is of the form:
%
% p(\phi_{n, d}) = GvM(k_{n,d,1}, k_{n,d,2}, m_{n,d,1}, m_{n,d,2})
% p(y_{n}) = Gaussian(A * cos(\phi) + B * sin (\phi) + c, prec^-2 * I)
%
% Since the posterior is intractable, the problem is addressed by
% calculating a Variational approximation using the Expectation
% maximisation algorithm.
%
% Here we have outlined the main parameters for the model and its
% execution under the properties section.
%
% Also outlined in this file are the main functions for the core
% algorithm under the methods section.
    
    %% PROPERTIES SECTION
    %--------------------
    properties
        
    %------------------------------------------------------------
    % General
    %------------------------------------------------------------
        q_phi_type  % Type of variational distribution for the angles
        calcT       % Type of trigonometric moment calculation
        
        
        % User suplied quantities
        %--------------------------------------------------------
        
        N % number of points (1 x 1)
        M % number of data dimensions (1 x 1)
        D % number of hidden dimensions (1 x 1)
        
        Y % Data (M x N)
        X % Inducing points for GP regression (D x N)
        
        
        % Containers for precalculated quantities
        trYYT   % Trace of the outer product of Y
        cos_y   % Cosine of the data points
        sin_y   % Sine of the data points
        
        
        % Indexes for working with subsets
        M_ini % Initial point for subsets of the data dimensions
        M_end % End point for subsets of the data dimensions
        D_ini % Initial point for subsets of the latent dimensions
        D_end % End point for subsets of the latent dimensions
        N_ini % Initial point for subsets of the latent dimensions
        N_end % End point for subsets of the latent dimensions
        
        
        % Ground truth
        %--------------------------------------------------------
        gt_kappa    % Prior term \kappa
        gt_A        % A cosine coefficient matrix (M x D)
        gt_B        % B sine coefficient matrix (M x D)
        gt_u        % Cosine rotation (D x 1)
        gt_v        % Sine rotation (D x 1)
        gt_prc2     % squared precision (1 x 1)
        
    %-----------------------------------------------------------
    % Optimisation parameters
    %-----------------------------------------------------------
        
        
        % Model parameters
        %--------------------------------------------------------
        set_matrices_tril2 = true; % Option to consider the matrices lower triangular
        
        % Priors for ARD parameters
        p_a_alph2 = xvar(); % point estimate for a of Gamma prior on alph2
        p_b_alph2 = xvar(); % point estimate for b of Gamma prior on alph2
        
        p_a_beta2 = xvar(); % point estimate for a of Gamma prior on beta2
        p_b_beta2 = xvar(); % point estimate for b of Gamma prior on beta2
        
        % ARD priors over coefficient matrices
        p_alph2 = xvar();   % point estimate for alph2 for prior on A
        p_beta2 = xvar();   % point estimate for beta2 for prior on B
        
        % Prior for model precision
        p_a_prc2 = xvar();  % point estimate for a of Gamma prior on prc2
        p_b_prc2 = xvar();  % point estimate for b of Gamma prior on prc2
        
        % Model Precision
        p_prc2 = xvar();    % point estimate for model precision
        
        % Prior for concentration of latent angles
        p_beta0 = xvar();   % point estimate for beta0 of BE prior on kappa
        p_eta = xvar();     % point estimate for eta of BE prior on kappa
        
        % Prior for latent angles
        p_kappa = xvar();   % point estimate for concentrations of phi
        
        % Coefficient matrices and offset
        p_A = xvar();       % point estimate for A matrix
        p_B = xvar();       % point estimate for B matrix
        
        % Blocks of the gp matrices (cholesky factor)
        p_Ucc = xvar();     % gp block for cos-cos terms
        p_Ucs = xvar();     % gp block for cos-sin terms
        p_Uss = xvar();     % gp block for sin-sin terms
        
        % Blocks of the covariances (full form)
        p_Wcc = xvar();     % covariance cos-cos terms
        p_Wcs = xvar();     % covariance cos-sin terms
        p_Wss = xvar();     % covariance sin-sin terms
        
        % Kernel hyperparameters (Squared exponential plus noise)
        p_s2_y = xvar();    % Isotropic noise (white noise kernel)
        p_s2se = xvar();    % SE kernel pre-exp factor
        p_ell2 = xvar();    % SE kernel lengthscale
        
        
        % Mean field parameters
        %--------------------------------------------------------
        
        % ARD priors
        q_alph2_a = xvar(); % ARD precision for coefficient matrix A
        q_alph2_b = xvar(); % ARD precision for coefficient matrix A
        m_alph2 = xvar();   % <alph2>
        
        q_beta2_a = xvar(); % ARD precision for coefficient matrix B
        q_beta2_b = xvar(); % ARD precision for coefficient matrix B
        m_beta2 = xvar();   % <beta2>
        
        % Prior for model precision
        q_prc2_a = xvar();  % Parameter a of Gamma distribution for q(prc2)
        q_prc2_b = xvar();  % Parameter b of Gamma distribution for q(prc2)
        m_prc2 = xvar();    % <prc2>
        
        % Concentration
        q_kappa_beta0 = xvar(); % Pseudomean cos parameter of bessel exp.
        q_kappa_eta = xvar();   % Pseudocount parameter of bessel exp.
        m_kappa = xvar();       % <kappa>
        
        % Latent angles
        q_phi_k1 = xvar();     % k1 parameter of GVM for q(phi)
        q_phi_k2 = xvar();     % k2 parameter of GVM for q(phi)
        q_phi_m1 = xvar();     % m1 parameter of GVM for q(phi)
        q_phi_m2 = xvar();     % m2 parameter of GVM for q(phi)
        
        m_sin_phi = xvar();    % <sin(phi)>
        m_cos_phi = xvar();    % <cos(phi)>
        m_sin2_phi = xvar();   % <sin(phi)^2>
        m_sincos_phi = xvar(); % <sin(phi)*cos(phi)>
        m_cos2_phi = xvar();   % <cos(phi)^2>
        
        % Mean field for coefficient matrices
        q_A_mu = xvar();       % Mean of Gaussian distribution for q(A,B)
        q_B_mu = xvar();       % Mean of Gaussian distribution for q(A,B)
        q_AA_cov = xvar();     % AA block of the covariance from q(A,B)
        q_BB_cov = xvar();     % AA block of the covariance from q(A,B)
        q_AB_cov = xvar();     % AA block of the covariance from q(A,B)
        
        m_A = xvar();          % <A>
        m_B = xvar();          % <B>
        m_AA = xvar();         % <A^T A>
        m_BB = xvar();         % <B^T B>
        m_AB = xvar();         % <A^T B>
        
        % Log joint - chain rule derivatives
        %--------------------------------------------------------
        
        % Chain rule for the variational parameters of q(phi)
        g_sin_phi_k1       % d/dk1 <sin(phi)>
        g_cos_phi_k1       % d/dk1 <cos(phi)>
        g_sin2_phi_k1      % d/dk1 <sin(phi)^2>
        g_sincos_phi_k1    % d/dk1 <sin(phi)*cos(phi)>
        g_cos2_phi_k1      % d/dk1 <cos(phi)^2>
        
        g_sin_phi_k2       % d/dk2 <sin(phi)>
        g_cos_phi_k2       % d/dk2 <cos(phi)>
        g_sin2_phi_k2      % d/dk2 <sin(phi)^2>
        g_sincos_phi_k2    % d/dk2 <sin(phi)*cos(phi)>
        g_cos2_phi_k2      % d/dk2 <cos(phi)^2>
        
        g_sin_phi_m1       % d/dm1 <sin(phi)>
        g_cos_phi_m1       % d/dm1 <cos(phi)>
        g_sin2_phi_m1      % d/dm1 <sin(phi)^2>
        g_sincos_phi_m1    % d/dm1 <sin(phi)*cos(phi)>
        g_cos2_phi_m1      % d/dm1 <cos(phi)^2>
        
        g_sin_phi_m2       % d/dm2 <sin(phi)>
        g_cos_phi_m2       % d/dm2 <cos(phi)>
        g_sin2_phi_m2      % d/dm2 <sin(phi)^2>
        g_sincos_phi_m2    % d/dm2 <sin(phi)*cos(phi)>
        g_cos2_phi_m2      % d/dm2 <cos(phi)^2>
        
        % Moments
        %--------------------------------------------------------
        
        T0 % normalising constant of each mean field distribution
        expw_T0 % normalising constant of each mean field distribution
        T1 % A + i*B, whith A, B 1st moment func. in Gatto (2008)
        T2 % A + i*B, whith A, B 2nd moment func. in Gatto (2008)
        T3 % A + i*B, whith A, B 3st moment func. in Gatto (2008)
        T4 % A + i*B, whith A, B 4nd moment func. in Gatto (2008)
        
        
        % Entropy
        %--------------------------------------------------------
        
        % Entropies for the variational distributions
        h_q_alph2     % H( q(alph2) )
        h_q_beta2     % H( q(beta2) )
        h_q_prc2      % H( q(prc2) )
        h_q_kappa     % H( q(kappa) )
        h_q_phi       % H( q(phi) )
        h_q_AB        % H( q(A,B) )
        
        % Free energy
        %--------------------------------------------------------
        fq                   % Free energy storage
        
        % Convergence assessment
        %--------------------------------------------------------
        atol = 1.E-2; % absolute tolerance
        rtol = 1.E-4; % relative tolerance
        
        % Optimisation options
        %--------------------------------------------------------
        opt_max_iter = 5E3;     % Maximum iterations
        opt_tol = 1E-4;         % Tolerance
        opt_info                % Output information
        opt_solver = 'U-BFGS';  % Solver (alternative is IPOPT)
        opt_mem = 5;            % Limited memory setting (3 <= m <= 20)
        
        var_old;                % array for previous iteration variables
        var_new;                % array for current iteration variables
        
        lb_array;               % array for lower bounds on variables
        ub_array;               % array for upper bounds on variables
        
        obj_free;               % objective function for the optimisation
        grad_free;              % gradient function for the optimisation

        % Moment calculation type
        %--------------------------------------------------------
        startGrid = 0; % point of switch from series to grid
        
        
    %------------------------------------------------------------
    % Gibbs sampling
    %------------------------------------------------------------

        % Sample containers
        %--------------------------------------------------------
        s_phi           % Samples for angles phi
        s_kappa         % Samples for kappa
        s_A             % Samples for matrix A
        s_B             % Samples for matrix B
        s_prc2          % Samples for prc2
        samples = {};
        
        pred_idx = []; % indices for sampled predictions.
        pred_sample = {}; % container for prediction samples
        
        % Options
        %--------------------------------------------------------
        E = 1;          % Thinning (keep every E samples)
        S = 1E5;        % Number of samples to be obtained
        burn = 1E3;     % Burn in period
        s_print = 100;  % Print every s_print samples

        
        % Other options
        %--------------------------------------------------------
        verbose = false;                % written output
        name = 'untitled';              % Model name
        name_map = containers.Map();    % Model variable dictionary
        
        
    %------------------------------------------------------------
    % Expectation Propagation
    %------------------------------------------------------------

        % Sample containers
        %--------------------------------------------------------
        N_factors           % Number of EP factors
        ep_factor = VonMisesFactor(); % Array of EP factors
        ep_factor_old = VonMisesFactor(); % Factors at previous iteration
        ep_q_log_z          % Cell to store z individual approximations.
        ep_q_T0             % Cell to store 0th moments of the q_new dist.
        ep_q_T1             % Cell to store 1st moments of the q_new dist.
        ep_log_z = 1; % log normalizing constant of the approximation q
        ep_log_z_hist = [];
        
        % Options
        %--------------------------------------------------------
        ep_damping = 0.99;  % Damping of updates (1 = all old, 0 = all new)
        ep_max_iter = 500;  % Maximum number of EP iterations
        ep_tol = 1E-4;      % Tolerance needed for convergence
        not_converged = true;
    end
    
    %% METHODS SECTION
    %------------------
    methods
        % Constructor
        %-----------------------------
        
        function self = Cpca(Y, D, options, X)
        % CPCA constructor for the cpca class.
        % Takes the dataset Y, the number of hidden dimensions D and an
        % option cell array.
        
            [R, C] = size(Y);
            self.Y = Y;
            self.M = R;
            self.N = C;
            
            if nargin > 3
                self.X = X;
            end
            
            self.D = D;
            % Precalculate conditions to be used within algorithms
            self.trYYT = trace(self.Y * self.Y'); % point and bayes
            self.cos_y = cos(self.Y); % gp
            self.sin_y = sin(self.Y); % gp
            
            % The default subset of indices to be used is all indexes
            self.M_ini = 1;
            self.M_end = self.M;
            
            self.D_ini = 1;
            self.D_end = self.D;
            
            self.N_ini = 1;
            self.N_end = self.N;
            
            % Options
            if nargin > 2
                for ll = 1:length(options)
                    aux = options{ll}{2}; %#ok<NASGU>
                    eval(['self.', options{ll}{1},'= aux;']);
                end
            end
        end
        
        % ---------------------------------------------------------------
        % All remaining methods are stored in extra files in the @Cpca
        % directory.
        % ---------------------------------------------------------------
    end
end
