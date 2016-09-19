function self = unpack_from_array(self, values, packable, list_type) %#ok<INUSL>
%unpack_from_array
% Unpacks variables (or parameters) from a value array into a string of
% commands to be evaluated so that the values are assigned to the name in
% the list within each object.

    list = self.name_map([list_type, '_variables']);
    subset_ini = self.name_map([list_type, '_subset_ini']);
    subset_end = self.name_map([list_type, '_subset_end']);
    n_entries = size(list, 2);

    % To reshape, we will parse a string with the following elements:
    %
    % prefix = prefix to reference the object
    % xvar_name = the list itself
    % x_type = identifier of the xvar entry type (variable, gradient, etc)
    % subset = subset of indices that will be considered
    % reshape_pre = assignment operator and reshape command
    % transform_pre = prefix of transformation to be applied on the array
    % array_str = variables in array form
    % transform_suf = prefix of transformation to be applied on the array
    % shape_str = new shape that the elements of the array should take
    % reshape_suf = closing brackets of the reshape operation
    
    prefix = 'self.';
    assign = '=';
    reshape_pre = 'reshape(';
    reshape_suf = ')';
    suffix = ';';
    
    start = 1;
    % Generate a string to be evaluated in order to update the
    % object's shared 
    for entry = 1:n_entries
        xvar_name = list{entry};
        
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
        
        if strcmp(packable, 'variables')
            et_pre = 'exp(';
            x_type = '.xv';
            et_suf = ')';
            
        elseif strcmp(packable, 'gradients')
            et_pre = '';
            x_type = '.dx';
            et_suf = ['./', prefix, xvar_name, '.xv', subset];
            
        elseif strcmp(packable, 'lower_bounds')
            x_type = '.lb';
            
        elseif strcmp(packable, 'upper_bounds')
            x_type = '.ub';
        end

        % Parsing index bookkeeping elements
        
        [entry_r, entry_c, entry_p] = ...
            size(eval([prefix, xvar_name, x_type, subset]));

        shift = entry_r * entry_c * entry_p - 1;
        
        array_str = strcat('values(', num2str(start),...
                           ':', num2str(start + shift),')');
                           
        shape_str = strcat(',[', num2str(entry_r), ...
                           ',', num2str(entry_c), ...
                           ',', num2str(entry_p), ']');
    
        % Apply transformation to variables
        if eval([prefix, xvar_name, '.et']) % check exponential transf.
            transform_pre = et_pre;
            transform_suf = et_suf;
        else % use raw variable
            transform_pre = '';
            transform_suf = '';
        end
        
        % Parsing string for evaluation
        eval_str = strcat(prefix, xvar_name, x_type, subset,...
                          assign, transform_pre, reshape_pre,...
                          array_str, shape_str, ...
                          reshape_suf, transform_suf, suffix);

        start = start + entry_r * entry_c;
        eval(eval_str);
    end
end