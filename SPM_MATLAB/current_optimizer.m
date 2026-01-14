clc;
clearvars;
clear global;
clear all;
tic;


% Initial C-rates (starting with high values)
C_rate = [4.00, 3.75, 3.5, 3.25];
Current_SOH = 0.9; % SOH between 0-1

% Define the maximum and minimum allowed C-rates
max_C_rate = 4.0;
min_C_rate = 0.05;

% Tolerance for stopping criteria
tolerance = 0.075; 

% Maximum iterations
max_iter = 100;

% Initialize previous C-rate
prev_C_rate = C_rate;

for iter = 1:max_iter
    % Calculate anodic potential and plating current
    eSPM(C_rate, Current_SOH);
    
    % Load the generated MAT file (data should be saved in a variable, e.g., 'data')
    loadedData = load('Anodic_potential_Charging.mat');  
    % Assume that the MAT file contains a variable 'data' which is a numeric matrix
    % with columns: SOC, Vn_p_1, Ipl_2.
    data = loadedData.Cycle10;  

    % Extract SOC and plating current columns
    SOC = data.SOC;
    plating_current = data.Ipl_2;

    % Filter data for SOC <= 0.8
    valid_indices = SOC <= 0.8;
    SOC = SOC(valid_indices);
    plating_current = plating_current(valid_indices);

    % Determine SOC intervals and indices for each interval
    intervals = [0, 0.20; 0.20, 0.40; 0.40, 0.60; 0.60, 0.80];
    indices = zeros(4, 2);

    for i = 1:4
        % Find indices for the current SOC interval
        start_idx = find(SOC >= intervals(i, 1), 1, 'first');
        end_idx = find(SOC < intervals(i, 2), 1, 'last');

        % Handle cases where find returns empty
        if isempty(start_idx)
            start_idx = 1;
        end

        if isempty(end_idx)
            end_idx = length(SOC);
        end

        indices(i, 1) = start_idx;
        indices(i, 2) = end_idx;
    end

    % Initialize flag to check if any adjustment is needed
    adjustment_needed = false;

    for i = 1:4
        % Check if any value in the current SOC interval has non-zero plating current
        if any(plating_current(indices(i, 1):indices(i, 2)) ~= 0)
            adjustment_needed = true;
            % Reduce the C-rate for the interval with non-zero plating current
            C_rate(i) = C_rate(i) - tolerance;
            C_rate(C_rate < min_C_rate) = min_C_rate; % Ensure C-rate does not go below minimum
        end
    end

    % If there were no adjustments and all plating currents are zero, exit the loop
    if ~adjustment_needed && all(plating_current == 0)
        break;
    end

    % Ensure descending order
    C_rate = sort(C_rate, 'descend');

    % Check for convergence (using norm for array comparison)
    if iter > 1 && norm(prev_C_rate - C_rate) < tolerance && all(plating_current == 0)
        break;
    end

    % Update previous C-rate
    prev_C_rate = C_rate;

    % Logging current values
    disp(['Iteration: ', num2str(iter)]);
    disp(['Current C-rates: ', num2str(C_rate)]);
    disp(['Plating Currents: ', num2str(plating_current')]);
end

% Display the optimized C-rates
disp('Optimized C-rates:');
disp(C_rate);

toc;

