%% Digital Signal Processing Project: I - ECG Bayesian Optimization
clear; close all; clc;

%% 1. Generate the Golden Ground Truth

fprintf('Generating Golden Signal...\n');
Fs = 360; 
load('100.mat');
raw_data = ecg_raw(:);

max_allowable_order = floor(length(raw_data) / 3) - 1;
N_golden = min(1500, max_allowable_order);
if mod(N_golden, 2) ~= 0, N_golden = N_golden - 1; end

fprintf('Using Golden FIR Order: %d\n', N_golden);
bp_freqs = [0.5 100] / (Fs/2);
b_golden_bp = fir1(N_golden, bp_freqs, 'bandpass', kaiser(N_golden+1, 8));

Q_golden = 100;
[b_golden_notch, a_golden_notch] = iirnotch(60/(Fs/2), (60/Q_golden)/(Fs/2));

temp_golden_full = filtfilt(b_golden_bp, 1, raw_data);
golden_ecg_full = filtfilt(b_golden_notch, a_golden_notch, temp_golden_full);

start_idx = 1 * Fs;
end_idx = 9 * Fs;
golden_ecg = golden_ecg_full(start_idx:end_idx);
t = (0:length(golden_ecg)-1)' / Fs;

%% 2. Generate Realistic Synthetic Noise

fprintf('Injecting Synthetic Noise...\n');
noise_baseline = 0.2 * sin(2*pi*0.2*t); 
noise_powerline = 0.1 * sin(2*pi*60*t);
raw_white_noise = randn(size(golden_ecg));
b_emg = fir1(50, [20 150]/(Fs/2), 'bandpass');
noise_muscle_shaped = filter(b_emg, 1, raw_white_noise);
noise_muscle = 0.05 * noise_muscle_shaped; 
synthetic_noisy_ecg = golden_ecg + noise_baseline + noise_powerline + noise_muscle;

%% 3. Setup Bayesian Optimization Space

fprintf('\n--- Initializing Bayesian Optimization ---\n');
fprintf('Objective: Minimize RMSE (Root Mean Square Error)\n');
fprintf('Constraint: 1 FIR, 2 IIRs. Strict Real-Time max delay of ~200ms.\n\n');

% Define the search space variables
vars = [
    optimizableVariable('FIR_Location', {'HP', 'Notch', 'LP'}, 'Type', 'categorical')
    optimizableVariable('HP_type', {'butter', 'cheby1', 'cheby2'}, 'Type', 'categorical')
    optimizableVariable('LP_type', {'butter', 'cheby1', 'cheby2'}, 'Type', 'categorical')
    optimizableVariable('Window', {'hamming', 'hann', 'blackman'}, 'Type', 'categorical')
    optimizableVariable('N_fir', [20, 150], 'Type', 'integer') % Max 150 caps delay at ~200ms
    optimizableVariable('N_iir_hp', [2, 6], 'Type', 'integer') % Kept low to avoid SOS NaN issues
    optimizableVariable('N_iir_lp', [2, 10], 'Type', 'integer')
    optimizableVariable('fc_hp', [0.3, 0.9], 'Type', 'real')
    optimizableVariable('fc_lp', [35, 100], 'Type', 'real')
    optimizableVariable('Q_notch', [10, 120], 'Type', 'real')
    optimizableVariable('Rp', [0.1, 1.0], 'Type', 'real')
    optimizableVariable('Rs', [30, 80], 'Type', 'real')
];

% Define the objective function
objFunc = @(params) evaluateFilter(params, Fs, synthetic_noisy_ecg, golden_ecg);

% Run Bayesian Optimization
rng('default');
results = bayesopt(objFunc, vars, ...
    'MaxObjectiveEvaluations', 150, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'PlotFcn', {@plotMinObjective});

%% 4. Extract and Apply the Winning Configuration
best_params = results.XAtMinObjective;
best_rmse = results.MinObjective;

fprintf('\nOPTIMIZATION COMPLETE\n');
fprintf('Lowest RMSE achieved: %.5f\n', best_rmse);
disp('Winning Parameters:');
disp(best_params);

% Reconstruct the best signal using the best parameters
[best_filtered_signal, delay] = buildAndFilter(best_params, Fs, synthetic_noisy_ecg);

%% 5. Display the Best Results
fprintf('Plotting Results...\n');
fig = figure('Name', 'Bayesian Optimization Results', 'Position', [100, 100, 1000, 600], 'Color', 'w');

% Time Domain Plot
subplot(2,1,1);
plot(t, synthetic_noisy_ecg, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5); hold on;
plot(t, golden_ecg, 'k', 'LineWidth', 1.5);
plot(t, best_filtered_signal, 'r--', 'LineWidth', 1.2);
title(sprintf('Time Domain Comparison (Best RMSE: %.5f | IRT Delay: %d samples)', best_rmse, delay));
xlabel('Time (s)'); ylabel('Amplitude (mV)');
legend('Synthetic Noisy', 'Golden Truth', 'Optimal Filter Output', 'Location', 'best');
xlim([3 8]); 
grid on;

% Error Signal Plot
subplot(2,1,2);
error_plot = golden_ecg - best_filtered_signal;
plot(t, error_plot, 'b', 'LineWidth', 1);
title('Residual Error (Golden Truth - Filtered Output)');
xlabel('Time (s)'); ylabel('Amplitude Error (mV)');
xlim([3 8]);
grid on;
ylim([-max(abs(golden_ecg))/2, max(abs(golden_ecg))/2]);

%% Save Bayesian Results to File
fprintf('Saving Bayesian Results...\n');
output_dir = 'Output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save workspace variables
mat_filename = fullfile(output_dir, 'Bayesian_Winning_Data.mat');
save(mat_filename, 'best_params', 'best_rmse', 'best_filtered_signal', 'results', 'delay');

% Save readable text log
txt_filename = fullfile(output_dir, 'Bayesian_Winning_Params.txt');
fid = fopen(txt_filename, 'w');
fprintf(fid, '--- Bayesian Optimization ---\n');
fprintf(fid, 'Lowest RMSE: %.5f\n', best_rmse);
fprintf(fid, 'Calculated FIR Delay: %d samples\n\n', delay);
fprintf(fid, 'Winning Parameters:\n');

% Format the Bayesian parameter table
varNames = best_params.Properties.VariableNames;
for i = 1:length(varNames)
    varName = varNames{i};
    val = best_params.(varName);
    
    % Handle categorical variables
    if iscategorical(val)
        fprintf(fid, '%15s: %s\n', varName, char(val));
    % Handle numeric variables
    elseif isnumeric(val)
        fprintf(fid, '%15s: %g\n', varName, val);
    else
        fprintf(fid, '%15s: %s\n', varName, string(val));
    end
end

fclose(fid);
fprintf('Bayesian results successfully saved to /Output directory.\n');

%% LOCAL FUNCTIONS %%

function cost = evaluateFilter(params, Fs, noisy_sig, golden_ecg)
    % Evaluates the filter pipeline and returns the RMSE
    try
        [aligned_sig, ~] = buildAndFilter(params, Fs, noisy_sig);
        
        eval_start = round(3 * Fs);
        eval_end = length(golden_ecg) - round(0.5 * Fs);
        
        error_sig = golden_ecg(eval_start:eval_end) - aligned_sig(eval_start:eval_end);
        cost = sqrt(mean(error_sig.^2));
        
        % Penalize unstable filters
        if isnan(cost) || isinf(cost)
            cost = 10; 
        end
    catch
        % Heavy penalty if parameters cause MATLAB to error out
        cost = 10; 
    end
end

function [aligned_sig, delay] = buildAndFilter(params, Fs, noisy_sig)
    % Extracts parameters and runs the 1-FIR, 2-IIR pipeline
    
    loc = char(params.FIR_Location);
    N_fir = params.N_fir;
    if mod(N_fir, 2) ~= 0, N_fir = N_fir - 1; end % Enforce even
    
    % Generate Window
    win_name = char(params.Window);
    if strcmp(win_name, 'hamming'), w = hamming(N_fir+1);
    elseif strcmp(win_name, 'hann'), w = hann(N_fir+1);
    else, w = blackman(N_fir+1); end
    
    %% 1. High Pass Generation
    if strcmp(loc, 'HP')
        b_hp = fir1(N_fir, params.fc_hp/(Fs/2), 'high', w);  a_hp = 1;
    else
        hp_t = char(params.HP_type);
        if strcmp(hp_t, 'butter')
            [b_hp, a_hp] = butter(params.N_iir_hp, params.fc_hp/(Fs/2), 'high');
        elseif strcmp(hp_t, 'cheby1')
            [b_hp, a_hp] = cheby1(params.N_iir_hp, params.Rp, params.fc_hp/(Fs/2), 'high');
        else
            [b_hp, a_hp] = cheby2(params.N_iir_hp, params.Rs, params.fc_hp/(Fs/2), 'high');
        end
    end
    
    %% 2. Notch Generation
    if strcmp(loc, 'Notch')
        bw = max(2, 60/params.Q_notch);
        b_notch = fir1(N_fir, [60-bw/2, 60+bw/2]/(Fs/2), 'stop', w);  a_notch = 1;
    else
        [b_notch, a_notch] = iirnotch(60/(Fs/2), (60/params.Q_notch)/(Fs/2));
    end
    
    %% 3. Low Pass Generation
    if strcmp(loc, 'LP')
        b_lp = fir1(N_fir, params.fc_lp/(Fs/2), 'low', w);  a_lp = 1;
    else
        lp_t = char(params.LP_type);
        if strcmp(lp_t, 'butter')
            [b_lp, a_lp] = butter(params.N_iir_lp, params.fc_lp/(Fs/2), 'low');
        elseif strcmp(lp_t, 'cheby1')
            [b_lp, a_lp] = cheby1(params.N_iir_lp, params.Rp, params.fc_lp/(Fs/2), 'low');
        else
            [b_lp, a_lp] = cheby2(params.N_iir_lp, params.Rs, params.fc_lp/(Fs/2), 'low');
        end
    end
    
    %% Processing Data
    sig1 = filter(b_hp, a_hp, noisy_sig);
    sig2 = filter(b_notch, a_notch, sig1);
    out_sig = filter(b_lp, a_lp, sig2);
    
    %% Delay Alignment
    
    delay = floor(N_fir / 2);
    aligned_sig = [out_sig(delay+1:end); zeros(delay, 1)];
end