function values = pack_as_array(self, packable_type, list_type)
% pack_as_array.m
% Joins all variables within a list into a package
%#ok<*NASGU>
%#ok<*AGROW>

    list = self.name_map([list_type, '_variables']);
    subset_ini = self.name_map([list_type, '_subset_ini']);
    subset_end = self.name_map([list_type, '_subset_end']);

    n_entries = size(list, 2);
    values = [];
    
    
    aux_prefix = 'aux = ';
    prefix = 'self.';
    suffix = ';';
    for entry = 1:n_entries
        % Parse evaluation string
        xvar_name = list{entry};
        
        % get subset indices to be used in the packing
        switch length(subset_ini{entry})
            case 3
                subset = ['(', num2str(subset_ini{entry}(1)), ...
                          ':', num2str(subset_end{entry}(1)), ...
                          ',', num2str(subset_ini{entry}(2)), ...
                          ':', num2str(subset_end{entry}(2)), ...
                          ',', num2str(subset_ini{entry}(3)), ...
                          ':', num2str(subset_end{entry}(3)), ...
                          ')'];
            case 2
                subset = ['(', num2str(subset_ini{entry}(1)), ...
                          ':', num2str(subset_end{entry}(1)), ...
                          ',', num2str(subset_ini{entry}(2)), ...
                          ':', num2str(subset_end{entry}(2)), ...
                          ')'];

            case 1
                subset = ['(', num2str(subset_ini{entry}(1)), ...
                          ':', num2str(subset_end{entry}(1)), ...
                          ')'];

            otherwise
                fprintf('Error! Too many dimensions!');
                keyboard;
        end
        

        if strcmp(packable_type, 'variables')

            et_pre = 'log(';
            x_type = '.xv';
            et_suf = ')';
            
        elseif strcmp(packable_type, 'gradients')
            
            et_pre = '';
            x_type = '.dx';
            et_suf = ['.*', prefix, xvar_name, '.xv', subset];
            
        elseif strcmp(packable_type, 'lower_bounds')
            
            et_pre = '';
            x_type = '.lb';
            et_suf = '';
            
        elseif strcmp(packable_type, 'upper_bounds')
            
            et_pre = '';
            x_type = '.ub';
            et_suf = '';
            
        end
    
        % Apply transformation to variables
        if eval([prefix, xvar_name, '.et']) % check exponential transf.
            transform_pre = et_pre;
            transform_suf = et_suf;
        else % use raw variable
            transform_pre = '';
            transform_suf = '';
        end
        
        % Parsing string for evaluation
        eval([aux_prefix, transform_pre, prefix, xvar_name, x_type,...
              subset, transform_suf, suffix]);
        
        [entry_r, entry_c, entry_p] = size(aux);               
        aux = reshape(aux, 1, entry_r * entry_c * entry_p);
        values = [values, aux];
    end
    % Transpose to be compatible with minimise function
    values = values';
end