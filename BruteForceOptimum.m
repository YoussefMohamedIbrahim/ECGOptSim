%% Digital Signal Processing Project: I - ECG Denoising Optimization
clear; close all; clc;

%% Generate the Golden Ground Truth
fprintf('Generating Golden Signal...\n');
Fs = 360; 
load('100.mat');
raw_data = ecg_raw(:);

max_allowable_order = floor(length(raw_data) / 3) - 1;
N_golden = min(1500, max_allowable_order);

if mod(N_golden, 2) ~= 0
    N_golden = N_golden - 1; 
end

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

%% Generate Realistic Synthetic Noise
fprintf('Injecting Synthetic Noise...\n');

% Baseline Wander
noise_baseline = 0.2 * sin(2*pi*0.2*t); 

% Powerline Interference
noise_powerline = 0.1 * sin(2*pi*60*t);

% Muscle Noise
raw_white_noise = randn(size(golden_ecg));
b_emg = fir1(50, [20 150]/(Fs/2), 'bandpass');
noise_muscle_shaped = filter(b_emg, 1, raw_white_noise);
noise_muscle = 0.05 * noise_muscle_shaped; 

synthetic_noisy_ecg = golden_ecg + noise_baseline + noise_powerline + noise_muscle;

%% 3. The Optimization Sweep
fprintf('Running Sequential Optimization (Objective: Minimize RMSE)...\n');

hpf_types = {'butter', 'cheby1', 'cheby2'};
windows = {'hamming', 'hann', 'blackman'};

hp_orders = 2:6;
fc_hps = 0.3:0.05:0.8;
notch_Qs = 20:10:100;
lp_orders = 30:20:150;
lp_cutoffs = 60:5:100;

total_iterations = length(hpf_types) * length(hp_orders) * length(fc_hps) * ...
                   length(notch_Qs) * length(lp_orders) * length(lp_cutoffs) * length(windows);

fprintf('Total combinations to evaluate: %d\n', total_iterations);

Rp = 0.5; % Passband ripple in dB
Rs = 50;  % Stopband attenuation in dB

best_cost = inf;
best_params = struct();
iteration_count = 0;

tic;

for i_type = 1:length(hpf_types)
    for i_hp_ord = hp_orders
        for fc_hp = fc_hps
            for Q = notch_Qs
                for N_lp = lp_orders
                    for fc_lp = lp_cutoffs
                        for i_win = 1:length(windows)
                            
                            % Progress Tracker Update
                            iteration_count = iteration_count + 1;
                            if mod(iteration_count, 5000) == 0
                                fprintf('  Progress: %d / %d combinations evaluated (%.1f%%)\n', ...
                                    iteration_count, total_iterations, (iteration_count/total_iterations)*100);
                            end
                            
                            hp_type = hpf_types{i_type};
                            win_name = windows{i_win};
                            
                            try
                                % Design Filters
                                if strcmp(hp_type, 'butter')
                                    [b, a] = butter(i_hp_ord, fc_hp/(Fs/2), 'high');
                                elseif strcmp(hp_type, 'cheby1')
                                    [b, a] = cheby1(i_hp_ord, Rp, fc_hp/(Fs/2), 'high');
                                else
                                    [b, a] = cheby2(i_hp_ord, Rs, fc_hp/(Fs/2), 'high');
                                end
                                
                                [bn, an] = iirnotch(60/(Fs/2), (60/Q)/(Fs/2));
                                
                                if strcmp(win_name, 'hamming'), w = hamming(N_lp+1);
                                elseif strcmp(win_name, 'hann'), w = hann(N_lp+1);
                                else, w = blackman(N_lp+1); end
                                bl = fir1(N_lp, fc_lp/(Fs/2), 'low', w);
                                
                                % Filter and Align
                                f_sig = filter(b, a, synthetic_noisy_ecg);
                                f_sig = filter(bn, an, f_sig);
                                f_sig = filter(bl, 1, f_sig);
                                
                                delay = N_lp / 2;
                                aligned = [f_sig(delay+1:end); zeros(delay, 1)];
                                
                                % Evaluate using RMSE
                                eval_start = round(3 * Fs);
                                eval_end = length(golden_ecg) - round(0.5 * Fs);
                                error_sig = golden_ecg(eval_start:eval_end) - aligned(eval_start:eval_end);
                                
                                cost = sqrt(mean(error_sig.^2));
                                
                                if cost < best_cost
                                    best_cost = cost;
                                    best_params.hp_type = hp_type;
                                    best_params.N_hp = i_hp_ord;
                                    best_params.fc_hp = fc_hp;
                                    best_params.Rp = Rp;
                                    best_params.Rs = Rs;
                                    best_params.Q = Q;
                                    best_params.N_lp = N_lp;
                                    best_params.fc_lp = fc_lp;
                                    best_params.window = win_name;
                                    best_params.rmse = cost;
                                    best_params.sig = aligned;
                                end
                            catch
                                continue;
                            end
                        end
                    end
                end
            end
        end
    end
end

best_rmse = best_params.rmse;
best_filtered_signal = best_params.sig;
toc;

fprintf('\nOPTIMIZATION COMPLETE\n');
fprintf('Lowest RMSE achieved: %.5f\n', best_rmse);
disp('Winning Parameters:');
disp(best_params);

%% Display the Best Results
fprintf('Plotting Results...\n');

fig = figure('Name', 'Optimization Results', 'Position', [100, 100, 1000, 600], 'Color', 'w');

% Time Domain Plot
subplot(2,1,1);
plot(t, synthetic_noisy_ecg, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5); hold on;
plot(t, golden_ecg, 'k', 'LineWidth', 1.5);
plot(t, best_filtered_signal, 'r--', 'LineWidth', 1.2);
title(sprintf('Time Domain Comparison (Best RMSE: %.5f)', best_rmse));
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

%% Save Brute Force Results to File
fprintf('Saving Brute Force Results...\n');
output_dir = 'Output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save workspace variables
mat_filename = fullfile(output_dir, 'BruteForce_Winning_Data.mat');
save(mat_filename, 'best_params', 'best_rmse', 'best_filtered_signal');

% Save text log
txt_filename = fullfile(output_dir, 'BruteForce_Winning_Params.txt');
fid = fopen(txt_filename, 'w');
fprintf(fid, '--- Brute Force Optimization ---\n');
fprintf(fid, 'Lowest RMSE: %.5f\n\n', best_rmse);
fprintf(fid, 'Winning Parameters:\n');
fields = fieldnames(best_params);
for i = 1:numel(fields)
    val = best_params.(fields{i});
    if isnumeric(val)
        if length(val) == 1
            fprintf(fid, '%15s: %g\n', fields{i}, val);
        else
            fprintf(fid, '%15s: [%dx%d %s array]\n', fields{i}, size(val,1), size(val,2), class(val));
        end
    elseif ischar(val) || isstring(val)
        fprintf(fid, '%15s: %s\n', fields{i}, val);
    end
end
fclose(fid);
fprintf('Brute Force results successfully saved to /Output directory.\n');