classdef TestSettings
%Class that defines the test settings for the CCA model. For the default
%values, see 
    
    %% PROPERTIES SECTION
    %--------------------
    properties
        N = 1E3; % Number of points (1 x 1)
        D = 2; % Number of latent dimensions (1 x 1)
        M = 4; % Number of observed dimensions (1 x 1)
        K = [7.5, 10.0]'; % Matrix of prior values to be tested (D X 1)
        L = [1.0, 1.0]'; % Tensor of pendulum lengths to be tested (D X 1)
        c = [0.0, 0.0]'; % Tensor of pendulum offsets to be tested (2 X 1)
        noise = 0.1; % Observation noise (1 x 1)
    end
        
    %% METHODS SECTION
    %------------------
    methods
        
        function self = testSettings(nCases)
        %TESTSETTINGS Constructor that creates an array of the standard
        %test setting for the experiments
            if nargin ~= 0
                % Preallocate array
                self(nCases) = testSettings;
                for testCase = 1:nCases
                    % Set instances
                    self(testCase) = self(testCase).setTestSettings();
                end
            end
        end
        
        function self = setTestSettings(self, N, D, K, L, c, noise)
        %TESTSETTINGS Constructor that creates the standard test setting
        %for most experiments
            if nargin > 1
                self.N = N .* ones(1, 1);
                self.D = D .* ones(1, 1);
                self.M = 2 .* self.D .* ones(1, 1);
                self.K = K .* ones(self.D, 1);
                self.L = L .* ones(self.D, 1);
                self.c = c .* ones(2, 1);
                self.noise = noise .* ones(1, 1);
            end
        end
    end
end