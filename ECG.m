%% Digital Signal Processing Project: I - ECG Signal Denoising
clear; close all; clc;

%% Configuration

OPTIMIZATION_METHOD = 'Bayesian'; % Options: 'BruteForce' or 'Bayesian'

%% Setup Output Directories
data_dir = 'Output';
output_dir = sprintf('Output_%s', OPTIMIZATION_METHOD);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
fprintf('Using %s Optimization Parameters.\n', OPTIMIZATION_METHOD);
fprintf('All PDFs and Reports will be saved to the "%s" directory.\n\n', output_dir);

%% Dynamic Filter Loading & Design
Fs = 360; 

if strcmp(OPTIMIZATION_METHOD, 'BruteForce')
    fprintf('Loading Brute Force Optimization Results...\n');
    load(fullfile(data_dir, 'BruteForce_Winning_Data.mat'), 'best_params');
    
    % A. High-Pass Filter
    if strcmp(best_params.hp_type, 'butter')
        [b_hp, a_hp] = butter(best_params.N_hp, best_params.fc_hp/(Fs/2), 'high');
    elseif strcmp(best_params.hp_type, 'cheby1')
        [b_hp, a_hp] = cheby1(best_params.N_hp, best_params.Rp, best_params.fc_hp/(Fs/2), 'high');
    else
        [b_hp, a_hp] = cheby2(best_params.N_hp, best_params.Rs, best_params.fc_hp/(Fs/2), 'high');
    end
    
    % B. Notch Filter
    [b_notch, a_notch] = iirnotch(60/(Fs/2), (60/best_params.Q)/(Fs/2));
    
    % C. Low-Pass Filter
    if strcmp(best_params.window, 'hamming'), w = hamming(best_params.N_lp+1);
    elseif strcmp(best_params.window, 'hann'), w = hann(best_params.N_lp+1);
    else, w = blackman(best_params.N_lp+1); end
    [b_lp, a_lp] = fir1(best_params.N_lp, best_params.fc_lp/(Fs/2), 'low', w);
    
    % Linear Phase Delay
    delay_samples = floor(best_params.N_lp / 2);

elseif strcmp(OPTIMIZATION_METHOD, 'Bayesian')
    fprintf('Loading Bayesian Optimization Results...\n');
    load(fullfile(data_dir, 'Bayesian_Winning_Data.mat'), 'best_params', 'delay');
    
    % Extract variables from the Bayesian table
    loc = char(best_params.FIR_Location);
    N_fir = best_params.N_fir;
    if mod(N_fir, 2) ~= 0, N_fir = N_fir - 1; end
    
    win_name = char(best_params.Window);
    if strcmp(win_name, 'hamming'), w = hamming(N_fir+1);
    elseif strcmp(win_name, 'hann'), w = hann(N_fir+1);
    else, w = blackman(N_fir+1); end
    
    % A. High-Pass Generation
    if strcmp(loc, 'HP')
        b_hp = fir1(N_fir, best_params.fc_hp/(Fs/2), 'high', w);  a_hp = 1;
    else
        hp_t = char(best_params.HP_type);
        if strcmp(hp_t, 'butter')
            [b_hp, a_hp] = butter(best_params.N_iir_hp, best_params.fc_hp/(Fs/2), 'high');
        elseif strcmp(hp_t, 'cheby1')
            [b_hp, a_hp] = cheby1(best_params.N_iir_hp, best_params.Rp, best_params.fc_hp/(Fs/2), 'high');
        else
            [b_hp, a_hp] = cheby2(best_params.N_iir_hp, best_params.Rs, best_params.fc_hp/(Fs/2), 'high');
        end
    end
    
    % B. Notch Generation
    if strcmp(loc, 'Notch')
        bw = max(2, 60/best_params.Q_notch);
        b_notch = fir1(N_fir, [60-bw/2, 60+bw/2]/(Fs/2), 'stop', w);  a_notch = 1;
    else
        [b_notch, a_notch] = iirnotch(60/(Fs/2), (60/best_params.Q_notch)/(Fs/2));
    end
    
    % C. Low-Pass Generation
    if strcmp(loc, 'LP')
        b_lp = fir1(N_fir, best_params.fc_lp/(Fs/2), 'low', w);  a_lp = 1;
    else
        lp_t = char(best_params.LP_type);
        if strcmp(lp_t, 'butter')
            [b_lp, a_lp] = butter(best_params.N_iir_lp, best_params.fc_lp/(Fs/2), 'low');
        elseif strcmp(lp_t, 'cheby1')
            [b_lp, a_lp] = cheby1(best_params.N_iir_lp, best_params.Rp, best_params.fc_lp/(Fs/2), 'low');
        else
            [b_lp, a_lp] = cheby2(best_params.N_iir_lp, best_params.Rs, best_params.fc_lp/(Fs/2), 'low');
        end
    end
    
    % Linear Phase Delay
    delay_samples = delay;
else
    error('Invalid OPTIMIZATION_METHOD. Please choose ''BruteForce'' or ''Bayesian''.');
end

%% Generate and Export Filter Characteristics

filters = {
    [OPTIMIZATION_METHOD '_HPF'], b_hp, a_hp;
    [OPTIMIZATION_METHOD '_Notch'], b_notch, a_notch;
    [OPTIMIZATION_METHOD '_LPF'], b_lp, a_lp
};

fprintf('Generating Filter Characteristic PDFs...\n');
for i = 1:size(filters, 1)
    name = filters{i, 1}; b = filters{i, 2}; a = filters{i, 3};
    
    fig_filt = figure('Name', [name ' Characteristics'], 'Position', [100, 100, 900, 600], ...
                      'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none');
    
    subplot(2,3,1); zplane(b, a); title('Pole-Zero Plot');
    subplot(2,3,2); impz(b, a, 50); title('Impulse Response');
    subplot(2,3,3); stepz(b, a, 50); title('Step Response');
    
    [H, f_hz] = freqz(b, a, 1024, Fs);
    
    subplot(2,3,[4 5]); 
    plot(f_hz, 20*log10(abs(H)), 'LineWidth', 1.5, 'Color', 'b'); 
    title('Magnitude Response'); xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)'); grid on;
    
    subplot(2,3,6); 
    plot(f_hz, unwrap(angle(H)) * (180/pi), 'LineWidth', 1.5, 'Color', 'r'); 
    title('Phase Response'); xlabel('Frequency (Hz)'); ylabel('Phase (Degrees)'); grid on;
    
    set(findall(fig_filt, 'type', 'axes'), 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k', 'GridAlpha', 0.2);
    set(findall(fig_filt, 'type', 'text'), 'Color', 'k');
    set(findall(fig_filt, 'type', 'legend'), 'Color', 'w', 'TextColor', 'k', 'EdgeColor', 'k');
    
    exportgraphics(fig_filt, fullfile(output_dir, [name '_Analysis.pdf']), 'ContentType', 'vector', 'BackgroundColor', 'w');
    close(fig_filt);
end

%% Process ECG Records

records = {'100', '106'};

% Open a text file to write the analysis report
log_filename = fullfile(output_dir, sprintf('%s_Analysis_Report.txt', OPTIMIZATION_METHOD));
fid_log = fopen(log_filename, 'w');
fprintf(fid_log, '=== Digital Signal Processing Project: I ===\n');
fprintf(fid_log, 'Optimization Method: %s\n', OPTIMIZATION_METHOD);
fprintf(fid_log, '============================================\n\n');

for r = 1:length(records)
    rec_name = records{r};
    fprintf('\n--- Processing Record %s ---\n', rec_name);
    
    % Load Data
    load([rec_name, '.mat']);
    N = length(ecg_raw);
    tm = (0:N-1) / Fs;
    ecg_raw = ecg_raw(:);
    
    % Apply Filters
    ecg_hp = filter(b_hp, a_hp, ecg_raw);
    ecg_notch = filter(b_notch, a_notch, ecg_hp);
    ecg_clean = filter(b_lp, a_lp, ecg_notch);
    
    %% Quantitative Analysis
    ecg_clean_aligned = [ecg_clean(delay_samples+1:end); zeros(delay_samples, 1)];
    
    noise_removed = ecg_raw - ecg_clean_aligned;
    ignore_idx = 2 * Fs;
    
    % Calculations
    snr_true = 10 * log10(var(ecg_clean_aligned(ignore_idx:end)) / var(noise_removed(ignore_idx:end)));
    rmse_val = sqrt(mean(noise_removed(ignore_idx:end).^2)); % RMSE of the removed noise/signal difference
    delay_ms = (delay_samples/Fs)*1000;
    
    % Print to Command Window
    fprintf('SNR: %.2f dB\n', snr_true);
    fprintf('RMSE (Raw vs Clean): %.4f mV\n', rmse_val);
    fprintf('Phase Delay: %d samples (%.2f ms)\n', delay_samples, delay_ms);
    
    % Write to Log File
    fprintf(fid_log, '--- Record %s ---\n', rec_name);
    fprintf(fid_log, 'SNR: %.2f dB\n', snr_true);
    fprintf(fid_log, 'RMSE (Raw vs Clean): %.4f mV\n', rmse_val);
    fprintf(fid_log, 'Phase Delay: %d samples (%.2f ms)\n\n', delay_samples, delay_ms);
    
    %% Visualization & PDF Export
    
    % Time-Domain Comparison
    fig_time = figure('Name', ['Time Domain - Record ' rec_name], 'Position', [100, 100, 800, 500], ...
                      'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none');
    subplot(2,1,1);
    plot(tm, ecg_raw, 'b'); title(['Original Corrupted ECG (Record ' rec_name ')']); 
    xlabel('Time (s)'); ylabel('Amplitude (mV)'); xlim([0 5]); grid on;
    subplot(2,1,2);
    plot(tm, ecg_clean, 'r'); title(sprintf('Cleaned ECG (%s Optimized)', OPTIMIZATION_METHOD)); 
    xlabel('Time (s)'); ylabel('Amplitude (mV)'); xlim([0 5]); grid on;
    
    set(findall(fig_time, 'type', 'axes'), 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k', 'GridAlpha', 0.2);
    set(findall(fig_time, 'type', 'text'), 'Color', 'k');
    
    exportgraphics(fig_time, fullfile(output_dir, [OPTIMIZATION_METHOD '_Record_' rec_name '_TimeDomain.pdf']), 'ContentType', 'vector', 'BackgroundColor', 'w');

    % Superimposed Comparison
    fig_super = figure('Name', ['Superimposed - Record ' rec_name], 'Position', [150, 150, 800, 400], ...
                      'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none');
    
    plot(tm, ecg_raw, 'b', 'LineWidth', 0.8); 
    hold on;
    plot(tm, ecg_clean_aligned, 'r', 'LineWidth', 1.5); 
    
    title(sprintf('Superimposed Comparison (Record %s - %s)', rec_name, OPTIMIZATION_METHOD)); 
    xlabel('Time (s)'); ylabel('Amplitude (mV)'); 
    xlim([3 5]);
    grid on;
    
    lgd_super = legend('Original (Corrupted)', 'Filtered (Aligned)', 'Location', 'northeast');
    set(lgd_super, 'TextColor', 'k', 'EdgeColor', 'k', 'Color', 'w');
    
    set(findall(fig_super, 'type', 'axes'), 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k', 'GridAlpha', 0.2);
    set(findall(fig_super, 'type', 'text'), 'Color', 'k');
    
    exportgraphics(fig_super, fullfile(output_dir, [OPTIMIZATION_METHOD '_Record_' rec_name '_Superimposed.pdf']), ...
                   'ContentType', 'vector', 'BackgroundColor', 'w');
    close(fig_super);
    
    % Power Spectral Density (Welch)
    fig_psd = figure('Name', ['PSD - Record ' rec_name], 'Position', [150, 150, 800, 400], ...
                     'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none');
    [pxx_raw, f_raw] = pwelch(ecg_raw, [], [], [], Fs);
    [pxx_clean, f_clean] = pwelch(ecg_clean, [], [], [], Fs);
    
    plot(f_raw, 10*log10(pxx_raw), 'b', 'LineWidth', 1); hold on;
    plot(f_clean, 10*log10(pxx_clean), 'r', 'LineWidth', 1.5);
    
    title(sprintf('Power Spectral Density (Record %s - %s)', rec_name, OPTIMIZATION_METHOD)); 
    xlabel('Frequency (Hz)'); ylabel('Power/Frequency (dB/Hz)');
    
    lgd_psd = legend('Original', 'Filtered', 'Location', 'northeast'); 
    set(lgd_psd, 'TextColor', 'k', 'EdgeColor', 'k', 'Color', 'w');
    xlim([0 180]); grid on;
    
    set(findall(fig_psd, 'type', 'axes'), 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k', 'GridAlpha', 0.2);
    set(findall(fig_psd, 'type', 'text'), 'Color', 'k');
    
    exportgraphics(fig_psd, fullfile(output_dir, [OPTIMIZATION_METHOD '_Record_' rec_name '_PSD.pdf']), 'ContentType', 'vector', 'BackgroundColor', 'w');
    
    % Spectrogram (STFT)
    fig_spec = figure('Name', ['Spectrogram - Record ' rec_name], 'Position', [200, 200, 800, 600], ...
                      'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none');
    subplot(2,1,1);
    spectrogram(ecg_raw, 256, 250, 256, Fs, 'yaxis');
    title(['Original ECG Spectrogram (Record ' rec_name ')']); ylim([0 180]);
    subplot(2,1,2);
    spectrogram(ecg_clean, 256, 250, 256, Fs, 'yaxis');
    title(sprintf('Cleaned ECG Spectrogram (%s)', OPTIMIZATION_METHOD)); ylim([0 180]);
    
    set(findall(fig_spec, 'type', 'axes'), 'XColor', 'k', 'YColor', 'k'); 
    set(findall(fig_spec, 'type', 'text'), 'Color', 'k');
    h_cb = findall(fig_spec, 'type', 'colorbar');
    set(h_cb, 'Color', 'k');
    
    exportgraphics(fig_spec, fullfile(output_dir, [OPTIMIZATION_METHOD '_Record_' rec_name '_Spectrogram.pdf']), 'Resolution', 300, 'BackgroundColor', 'w');
    
    close(fig_time); close(fig_psd); close(fig_spec);
end

% Close the log file
fclose(fid_log);

fprintf('\nAll PDFs and the Analysis Report successfully exported to /%s.\n', output_dir);