%% compare_five_models_MI_FUSION_FRAMEWORK_REFACTORED.m
% ============================================================
%  Five-model unified framework for MI-based MCS prediction
%
%  (1) FIXED      : fixed MI-EESM aggregation
%  (2) FEATURE    : degeneration-feature driven MLP
%  (3) E2E        : end-to-end MI neural network
%  (4) OURSNOB    : adaptive MI aggregation (beta,w), no b
%  (5) OURSFUSION : physics-data fusion network
% ============================================================

clc; clear; close all;

%% ===================== Parallel Pool =====================
try
    p = gcp('nocreate');
    if isempty(p), parpool('threads'); end
catch
    p = gcp('nocreate');
    if isempty(p), parpool; end
end

%% ===================== METHOD REGISTRY =====================
METHODS = get_method_registry();

%% ===================== CFG =====================
cfg = struct();

% ---------------- file / columns ----------------
cfg.filename = 'd0c1-bfe4-18f0_0.xlsx';
cfg.mcs_col  = 'mcs';
cfg.bf_col   = 'beamforming_en';
cfg.nf_col   = 'noise_floor';

cfg.csi_cols = { ...
    'csi_matrix_r0_c0','csi_matrix_r0_c1','csi_matrix_r0_c2','csi_matrix_r0_c3', ...
    'csi_matrix_r1_c0','csi_matrix_r1_c1','csi_matrix_r1_c2','csi_matrix_r1_c3'};

cfg.Nr = 2;
cfg.Nt = 4;
cfg.K  = 122;

% ---------------- split ----------------
cfg.run_all_splits = true;
cfg.split_modes_main     = {'random','nf_ood','snr_gap_ood'};
cfg.split_modes_appendix = {'gap_ood'};
cfg.split_mode = 'random';

cfg.train_ratio  = 0.8;
cfg.nf_train_quantile  = 0.5;
cfg.gap_train_quantile = 0.5;
cfg.snr_gap_q = 0.7;
cfg.gap_hard_q = 0.7;
cfg.snr_gap_minValN = 50;

% ---------------- loss / regularization ----------------
cfg.useFeatureOrdinalMAE = true;
cfg.ordMAEWeight = single(0.20);

cfg.useAuxOrdinalMAE = true;
cfg.auxMAEWeight = single(0.10);

cfg.beta_min = single(0.05);
cfg.beta_max = single(8.00);

cfg.w_min = single(0.40);
cfg.w_max = single(2.20);

cfg.b_min = single(-1.50);
cfg.b_max = single(1.50);

cfg.useBetaWReg = true;
cfg.betaRegWeight = single(1e-3);
cfg.wRegWeight    = single(1e-3);
cfg.bRegWeight    = single(1e-3);

cfg.betaCenter = single(1.0);
cfg.wCenter    = single(1.0);
cfg.bCenter    = single(0.0);

cfg.use_b_in_ours_nob = false;
cfg.use_b_in_fusion   = true;

% ---------------- few-shot ----------------
cfg.seeds = [0:2];
cfg.train_subset_fracs = [1.00];
cfg.minTrainN = 50;

% ---------------- training ----------------
cfg.numEpochs = 80;
cfg.batchSize = 128;
cfg.lr = 1e-3;

cfg.eps1 = single(1e-6);
cfg.eps2 = single(1e-12);

% ---------------- x_deg ----------------
cfg.domain_shift_ref = 'MI';   % 'MI' | 'gamma1'
cfg.xdeg_dim = 25;

% ---------------- threshold anchor ----------------
cfg.theta_anchor_mode = 'first_zero';

% ---------------- fusion architecture ----------------
cfg.ours_data_hidden1 = 64;
cfg.ours_data_hidden2 = 32;
cfg.ours_phys_hidden1 = 16;
cfg.ours_phys_hidden2 = 8;
cfg.ours_fuse_dim     = 32;

cfg.ours_use_prior_logit_fusion = true;
cfg.ours_prior_logit_weight = single(0.08);

cfg.ours_use_phy_align = true;
cfg.ours_phyAlignWeight = single(0.07);

cfg.ours_use_risk_aux = false;
cfg.ours_riskWeight = single(0.08);

cfg.ours_use_weighted_ce = true;
cfg.ours_weighted_ce_lambda = single(0.15);

cfg.ours_use_gate = false;
cfg.ours_init_scale = single(0.02);

% ---------------- structure switches for ablation ----------------
cfg.ours_use_data_branch  = true;
cfg.ours_use_phys_feature = true;   % whether h_phy participates in fusion

% ---------------- early stop ----------------
cfg.useEarlyStop = true;
cfg.patience    = 12;
cfg.minDeltaAcc = 1e-4;

% ---------------- low-MCS ----------------
cfg.lowMode = 'quantile';
cfg.lowMCS_m0 = 1;
cfg.lowQuantile = 0.30;

% ---------------- margin stability analysis ----------------
cfg.do_margin_stability_analysis = true;
cfg.margin_use_frac = 1.0;
cfg.margin_use_seed = cfg.seeds(1);
cfg.margin_split_list = {'random','nf_ood','gap_ood','snr_gap_ood'};

cfg.margin_eps_list = [0.02 0.05 0.08 0.12];   % MI 扰动预算（每个子载波 i_k 的幅度）
cfg.margin_nbins    = 8;                       % margin 分桶数
cfg.margin_mc       = 30;                      % Monte Carlo 次数
cfg.margin_use_quantile_bins = true;

%% ===================== High-SNR Conditional Mechanism =====================
cfg.highsnr.enable        = true;
cfg.highsnr.q_hi          = 0.80;   % top 80% 作为 high-SNR
cfg.highsnr.q_lo          = 0.20;   % bottom 20% 作为 low-SNR 对照
cfg.highsnr.nbins         = 6;      % Gap 分桶数
cfg.highsnr.proxy_mode    = 'auto'; % 'auto' | 'snr_db' | 'rho' | 'mi_bar'
cfg.highsnr.min_per_bin   = 10;     % 每个 bin 最少样本数
cfg.highsnr.make_plots    = true;
cfg.highsnr.save_mat      = true;

% ---------------- uncertainty / calibration ----------------
cfg.evalECE_bins = 15;
cfg.evalSelective_pts = 40;
cfg.evalAURC_useMAE = false;

% ---------------- timing ----------------
cfg.time_infer_reps = 2;
cfg.time_print_each_run = true;

% ---------------- fixed baseline ----------------
cfg.fixed_beta_list  = [0.25 0.5 1 2 4];
cfg.fixed_w_list     = [0.75 1.0 1.25 1.5];
cfg.fixed_select_by  = 'val_acc';

% ---------------- analysis plots ----------------
cfg.do_fewshot_triptych = false;
cfg.fewshot_plot_split_list = {'random'};
cfg.fewshot_use_logx = true;

cfg.do_ood_risk_plots = true;

cfg.do_beta_risk_analysis = true;
cfg.beta_risk_use_frac = 1.0;
cfg.beta_risk_use_seed = cfg.seeds(1);
cfg.beta_risk_split_list = {'random','nf_ood','gap_ood','snr_gap_ood'};
cfg.beta_risk_nbins = 10;
cfg.beta_risk_use_binary = false;

cfg.do_ess_entropy_analysis = true;
cfg.ess_use_frac = 1.0;
cfg.ess_use_seed = cfg.seeds(1);
cfg.ess_split_list = {'random'};
cfg.ess_nbins = 10;

% ---------------- new analysis / ablation ----------------
cfg.run_fusion_ablation = true;
cfg.ablation_split_list = {'random','gap_ood','snr_gap_ood','nf_ood'};
cfg.ablation_frac_list  = [1.00];
cfg.ablation_seed_list  = cfg.seeds;

cfg.ablation_specs = { ...
    'FULL',            struct(), ...
    'DATA_ONLY',       struct( ...
        'ours_use_phys_feature', false, ...
        'ours_use_prior_logit_fusion', false, ...
        'ours_use_phy_align', false, ...
        'use_b_in_fusion', false), ...
    'PHYS_ONLY',       struct( ...
        'ours_use_data_branch', false), ...
    'NO_PRIOR_ALL',    struct( ...
        'ours_use_prior_logit_fusion', false, ...
        'ours_use_phy_align', false), ...
    'NO_B',            struct('use_b_in_fusion', false), ...
    'NO_DEEPFADE',     struct('xdeg_zero_idx', 18:25) ...
    };

cfg.do_split_label_distribution = true;
cfg.do_confusion_analysis = true;
cfg.confusion_use_frac = 1.0;
cfg.confusion_use_seed = cfg.seeds(1);
cfg.confusion_split_list = {'random','gap_ood','snr_gap_ood','nf_ood'};
cfg.confusion_norm = 'row';   % 'row' | 'count'

cfg.do_classwise_analysis = true;
cfg.classwise_use_frac = 1.0;
cfg.classwise_use_seed = cfg.seeds(1);
cfg.classwise_split_list = {'random','gap_ood','snr_gap_ood','nf_ood'};

cfg.do_reliability_analysis = true;
cfg.reliability_use_frac = 1.0;
cfg.reliability_use_seed = cfg.seeds(1);
cfg.reliability_split_list = {'random','gap_ood','snr_gap_ood','nf_ood'};
cfg.reliability_methods = {'FEATURE','E2E','OURSNOB','OURSFUSION'};

cfg.do_risk_coverage_plot = true;
cfg.rc_use_frac = 1.0;
cfg.rc_use_seed = cfg.seeds(1);
cfg.rc_split_list = {'random','gap_ood','snr_gap_ood','nf_ood'};
cfg.rc_methods = {'FEATURE','E2E','OURSNOB','OURSFUSION'};
cfg.rc_npts = 50;

cfg.do_bootstrap_ci = true;
cfg.bootstrap_B = 1000;
cfg.bootstrap_use_frac = 1.0;
cfg.bootstrap_split_list = {'random','gap_ood','snr_gap_ood','nf_ood'};
cfg.bootstrap_methods = {'FEATURE','E2E','OURSNOB','OURSFUSION'};

% ---------------- outputs ----------------
cfg.output_root = 'runs_out';
cfg.run_tag = char(datetime('now','Format','yyyyMMdd_HHmmss'));
cfg.run_dir = fullfile(cfg.output_root, cfg.run_tag);

if ~exist(cfg.output_root, 'dir'), mkdir(cfg.output_root); end
if ~exist(cfg.run_dir, 'dir'), mkdir(cfg.run_dir); end

cfg.save_fig = true;
cfg.fig_dir = fullfile(cfg.run_dir, 'figs_out');
cfg.save_analysis_fig = true;
cfg.analysis_fig_dir = fullfile(cfg.run_dir, 'figs_analysis');

if ~exist(cfg.fig_dir,'dir'), mkdir(cfg.fig_dir); end
if ~exist(cfg.analysis_fig_dir,'dir'), mkdir(cfg.analysis_fig_dir); end

cfg.export_excel = true;
cfg.excel_out = fullfile(cfg.run_dir, 'results_five_methods.xlsx');
cfg.export_overwrite = true;

cfg.script_name = 'compare_five_models_MI_FUSION_FRAMEWORK_REFACTORED.m';
if exist(cfg.script_name, 'file')
    copyfile(cfg.script_name, fullfile(cfg.run_dir, cfg.script_name));
end

fprintf('\n====================================================\n');
fprintf('Run output directory:\n%s\n', cfg.run_dir);
fprintf('====================================================\n\n');

diary(fullfile(cfg.run_dir, 'run_log.txt'));
diary on;

%% ===================== 0) Read Excel =====================
T = readtable(cfg.filename);
N = height(T);
K = cfg.K;

y_raw = str2double(string(T.(cfg.mcs_col)));
bf    = str2double(string(T.(cfg.bf_col)));
n_dBm = str2double(string(T.(cfg.nf_col)));

assert(~any(isnan(y_raw)), 'mcs 列存在 NaN。');
assert(~any(isnan(bf)),    'beamforming_en 列存在 NaN。');
assert(~any(isnan(n_dBm)), 'noise_floor 列存在 NaN。');

Pn_mW = 10.^(n_dBm/10);

[levels, ~, y_enc] = unique(y_raw, 'sorted');
y = double(y_enc) - 1;
M = numel(levels);

fprintf('Mapped "%s" to ordinal classes: M=%d\n', cfg.mcs_col, M);
disp(table((0:M-1)', levels, 'VariableNames', {'class','level_value'}));
save(fullfile(cfg.run_dir, 'meta_basic.mat'), 'cfg', 'levels', 'M');

Cplx = complexity_estimate(cfg, M);
disp('===== Approx Params/MACs(FLOPs proxy), inference per sample =====');
disp(Cplx);

%% ===================== 1) Pipeline timing =====================
t_pre_all = tic;

fprintf('Parsing CSI columns...\n');
t_parse = tic;
for ci = 1:numel(cfg.csi_cols)
    colname = cfg.csi_cols{ci};
    T.(colname) = parse_csi_column_to_cellvec(T.(colname), K, colname);
end
time_parse = toc(t_parse);

fprintf('Building H...\n');
t_H = tic;
H = complex(zeros(cfg.Nr, cfg.Nt, K, N));
map = [ ...
    1 1; 1 2; 1 3; 1 4; ...
    2 1; 2 2; 2 3; 2 4];

for j = 1:N
    for idxCol = 1:numel(cfg.csi_cols)
        rx = map(idxCol,1);
        tx = map(idxCol,2);
        v  = T.(cfg.csi_cols{idxCol}){j};
        v  = check_vec(v, K, j, sprintf('r%dc%d', rx-1, tx-1));
        H(rx, tx, :, j) = reshape(v, 1, 1, K);
    end
end
time_buildH = toc(t_H);

fprintf('Precomputing MI + spectral statistics + x_deg raw components...\n');
t_preMI = tic;
[MI, eta, re, lk, GapK, gamma1, sigma2_ratio, log_cond] = ...
    precompute_MI_deg_FAST(H, Pn_mW, cfg);
time_preMI = toc(t_preMI);

time_pre_all = toc(t_pre_all);

fprintf('[PIPELINE TIME] parse=%.3fs | buildH=%.3fs | preMI=%.3fs | pre_all=%.3fs\n', ...
    time_parse, time_buildH, time_preMI, time_pre_all);

save(fullfile(cfg.run_dir, 'run_config.mat'), 'cfg', ...
    'time_parse', 'time_buildH', 'time_preMI', 'time_pre_all');

%% ===================== 2) EXP LOOP =====================
Results = struct([]);
AblationResults = struct([]);
row = 0;

if cfg.run_all_splits
    split_list = [cfg.split_modes_main, cfg.split_modes_appendix];
else
    split_list = {cfg.split_mode};
end

for sm = 1:numel(split_list)
    cfg.split_mode = split_list{sm};

    fprintf('\n\n###############################\n');
    fprintf('### SPLIT MODE = %s\n', cfg.split_mode);
    fprintf('###############################\n');

    for ff = 1:numel(cfg.train_subset_fracs)
        frac = cfg.train_subset_fracs(ff);

        for ss = 1:numel(cfg.seeds)
            seed = cfg.seeds(ss);

            row = row + 1;
            fprintf('\n====================================================\n');
            fprintf('RUN #%d | split=%s | train_frac=%.2f | seed=%d\n', ...
                row, cfg.split_mode, frac, seed);
            fprintf('====================================================\n');

            runCfg = cfg;
            runCfg.rng_seed = seed;
            runCfg.train_subset_frac = frac;

            [tr0, va] = make_split(runCfg, N, bf, n_dBm, GapK);

            tr = tr0;
            if frac < 1
                rng(seed);
                ntr = max(runCfg.minTrainN, round(frac*numel(tr0)));
                ntr = min(ntr, numel(tr0));
                tr = tr0(randperm(numel(tr0), ntr));
            end

            MI_tr = MI(:, tr); y_tr = y(tr);
            MI_va = MI(:, va); y_va = y(va);

            [Xdeg_tr, Xdeg_va, DegRef] = build_xdeg_rebuild( ...
                MI, gamma1, sigma2_ratio, log_cond, re, bf, n_dBm, tr, va, runCfg);

            [Xdeg_trs, Xdeg_vas] = zscore_train_only(Xdeg_tr, Xdeg_va);
            [MI_trs, MI_vas] = zscore_MI_train_only(MI_tr, MI_va);

            d_xdeg = size(Xdeg_trs, 2);

            fprintf('Train=%d (orig %d) | Val=%d | M=%d | x_deg_dim=%d\n', ...
                numel(tr), numel(tr0), numel(va), M, d_xdeg);

            runCfg.low_m0_train = low_threshold_m0(double(y_tr(:)), M, runCfg);

            ResultOne = struct();
            ResultOne.split = runCfg.split_mode;
            ResultOne.frac  = frac;
            ResultOne.seed  = seed;

            ResultOne.pipe_pre_all_s = time_pre_all;
            ResultOne.pipe_parse_s   = time_parse;
            ResultOne.pipe_buildH_s  = time_buildH;
            ResultOne.pipe_preMI_s   = time_preMI;

            ResultOne.y_val        = y_va(:);
            ResultOne.val_idx      = va(:);
            ResultOne.MI_val       = MI_va;
            ResultOne.Xdeg_val     = Xdeg_va;
            ResultOne.Gap_mean_val = mean(double(GapK(:,va)),1)';
            ResultOne.VarMI_val    = var(double(MI(:,va)),0,1)';
            ResultOne.Eta_mean_val = mean(double(eta(:,va)),1)';
            ResultOne.Re_mean_val  = mean(double(re(:,va)),1)';
            ResultOne.Lk_var_val   = var(double(lk(:,va)),0,1)';
            ResultOne.DegRef       = DegRef;

            %% ---------------- (1) FIXED ----------------
            fprintf('\n================= METHOD (1) FIXED =================\n');
            t_train = tic;
            [best0, Hist0, Time0] = train_fixed_mieesm_grid(MI_tr, y_tr, MI_va, y_va, M, runCfg);
            time_train = toc(t_train);

            [acc0, mae0, accpm1_0, Met0, Det0] = eval_fixed(best0, MI_va, y_va, M, runCfg);
            time_infer_total = time_infer_repeated(@() eval_fixed_details(best0, MI_va, M, runCfg), runCfg.time_infer_reps);
            time_infer_perSample = time_infer_total / numel(y_va);

            GP0 = goodput_proxy_metrics(y_va, Det0.yhat, levels);

            ResultOne = pack_method_result(ResultOne, 'FIXED', ...
                time_train, Time0.epoch_s_mean, time_infer_total, time_infer_perSample, ...
                acc0, mae0, accpm1_0, Met0, GP0, Det0, Hist0);
            ResultOne.FIXED_best_beta = best0.beta;
            ResultOne.FIXED_best_w    = best0.w;

            %% ---------------- (2) FEATURE ----------------
            fprintf('\n================= METHOD (2) FEATURE =================\n');
            layersFeat = [
                featureInputLayer(d_xdeg,'Normalization','none','Name','feat_in')
                fullyConnectedLayer(64,'Name','fc1')
                reluLayer('Name','relu1')
                fullyConnectedLayer(32,'Name','fc2')
                reluLayer('Name','relu2')
                fullyConnectedLayer(16,'Name','fc3')
                reluLayer('Name','relu3')
                fullyConnectedLayer(M,'Name','fc_out')
            ];
            netFeat = dlnetwork(layerGraph(layersFeat));

            t_train = tic;
            [bestF, HistF, TimeF] = train_feature_mlp(netFeat, Xdeg_trs, y_tr, Xdeg_vas, y_va, M, runCfg);
            time_train = toc(t_train);

            [accF, maeF, accpm1_F, MetF, DetF] = eval_feature_mlp(bestF, Xdeg_vas, y_va, M, runCfg);
            time_infer_total = time_infer_repeated(@() eval_feature_mlp_details(bestF, Xdeg_vas), runCfg.time_infer_reps);
            time_infer_perSample = time_infer_total / numel(y_va);

            GPF = goodput_proxy_metrics(y_va, DetF.yhat, levels);

            ResultOne = pack_method_result(ResultOne, 'FEATURE', ...
                time_train, TimeF.epoch_s_mean, time_infer_total, time_infer_perSample, ...
                accF, maeF, accpm1_F, MetF, GPF, DetF, HistF);

            %% ---------------- (3) E2E ----------------
            fprintf('\n================= METHOD (3) E2E =================\n');
            layersE2E = [
                featureInputLayer(K,'Normalization','none')
                fullyConnectedLayer(128)
                reluLayer
                fullyConnectedLayer(64)
                reluLayer
                fullyConnectedLayer(M)
            ];
            netE2E = dlnetwork(layerGraph(layersE2E));

            t_train = tic;
            [best2, Hist2, Time2] = train_e2e_MIonly(netE2E, MI_trs, y_tr, MI_vas, y_va, M, runCfg);
            time_train = toc(t_train);

            [acc2, mae2, accpm1_2, Met2, Det2] = eval_e2e_MIonly(best2, MI_vas, y_va, M, runCfg);
            time_infer_total = time_infer_repeated(@() eval_e2e_MIonly_details(best2, MI_vas), runCfg.time_infer_reps);
            time_infer_perSample = time_infer_total / numel(y_va);

            GP2 = goodput_proxy_metrics(y_va, Det2.yhat, levels);

            ResultOne = pack_method_result(ResultOne, 'E2E', ...
                time_train, Time2.epoch_s_mean, time_infer_total, time_infer_perSample, ...
                acc2, mae2, accpm1_2, Met2, GP2, Det2, Hist2);

            %% ---------------- (4) OURSNOB ----------------
            fprintf('\n================= METHOD (4) OURSNOB =================\n');
            layersPhysNoB = [
                featureInputLayer(d_xdeg,'Normalization','none','Name','xdeg_in')
                fullyConnectedLayer(runCfg.ours_phys_hidden1,'Name','fc1')
                reluLayer('Name','relu1')
                fullyConnectedLayer(runCfg.ours_phys_hidden2,'Name','fc2')
                reluLayer('Name','relu2')
                fullyConnectedLayer(2,'Name','phys_out')
            ];
            netPhysNoB = dlnetwork(layerGraph(layersPhysNoB));

            if M >= 3
                deltas_raw_nob = dlarray(single(0.2*ones(M-2,1)));
            else
                deltas_raw_nob = dlarray(single([]));
            end

            t_train = tic;
            [best3, Hist3, Time3] = train_ours_nob_ordinal( ...
                netPhysNoB, deltas_raw_nob, ...
                MI_tr, Xdeg_trs, y_tr, MI_va, Xdeg_vas, y_va, M, runCfg);
            time_train = toc(t_train);

            [acc3, mae3, accpm1_3, Met3, Det3] = eval_ours_nob_ordinal(best3, MI_va, Xdeg_vas, y_va, M, runCfg);
            [~, ~, ~, beta_va_nob, w_va_nob] = eval_ours_nob_ordinal_details_full(best3, MI_va, Xdeg_vas, M, runCfg);

            time_infer_total = time_infer_repeated(@() eval_ours_nob_ordinal_details(best3, MI_va, Xdeg_vas, M, runCfg), runCfg.time_infer_reps);
            time_infer_perSample = time_infer_total / numel(y_va);

            GP3 = goodput_proxy_metrics(y_va, Det3.yhat, levels);

            ResultOne = pack_method_result(ResultOne, 'OURSNOB', ...
                time_train, Time3.epoch_s_mean, time_infer_total, time_infer_perSample, ...
                acc3, mae3, accpm1_3, Met3, GP3, Det3, Hist3);
            ResultOne.OURSNOB_beta_val = beta_va_nob(:);
            ResultOne.OURSNOB_w_val    = w_va_nob(:);
            ResultOne.OURSNOB_yhat_val = Det3.yhat(:);

            %% ---------------- (5) OURSFUSION ----------------
            fprintf('\n================= METHOD (5) OURSFUSION =================\n');
            layersData = [
                featureInputLayer(K,'Normalization','none','Name','mi_in')
                fullyConnectedLayer(runCfg.ours_data_hidden1,'Name','fc1')
                reluLayer('Name','relu1')
                fullyConnectedLayer(runCfg.ours_data_hidden2,'Name','fc2')
                reluLayer('Name','relu2')
            ];
            netData = dlnetwork(layerGraph(layersData));

            layersPhys = [
                featureInputLayer(d_xdeg,'Normalization','none','Name','xdeg_in')
                fullyConnectedLayer(runCfg.ours_phys_hidden1,'Name','fc1')
                reluLayer('Name','relu1')
                fullyConnectedLayer(runCfg.ours_phys_hidden2,'Name','fc2')
                reluLayer('Name','relu2')
                fullyConnectedLayer(3,'Name','phys_out')
            ];
            netPhys = dlnetwork(layerGraph(layersPhys));

            if M >= 3
                deltas_raw = dlarray(single(0.2*ones(M-2,1)));
            else
                deltas_raw = dlarray(single([]));
            end

            fuseParams = init_ours_fusion_params(runCfg, M);

            t_train = tic;
            [best4, Hist4, Time4] = train_ours_pg_fusion( ...
                netData, netPhys, deltas_raw, fuseParams, ...
                MI_trs, MI_tr, Xdeg_trs, y_tr, ...
                MI_vas, MI_va, Xdeg_vas, y_va, M, runCfg);
            time_train = toc(t_train);

            [acc4, mae4, accpm1_4, Met4, Det4] = eval_ours_pg_fusion(best4, MI_vas, MI_va, Xdeg_vas, y_va, M, runCfg);
            [~, ~, ~, beta_va_fusion, w_va_fusion, b_va_fusion] = eval_ours_pg_fusion_details_full(best4, MI_vas, MI_va, Xdeg_vas, M, runCfg);
            
            theta_fusion = gather(extractdata(stripdims(build_theta_anchor(best4.deltas, M, runCfg.eps1))));

            time_infer_total = time_infer_repeated(@() eval_ours_pg_fusion_details(best4, MI_vas, MI_va, Xdeg_vas, M, runCfg), runCfg.time_infer_reps);
            time_infer_perSample = time_infer_total / numel(y_va);

            GP4 = goodput_proxy_metrics(y_va, Det4.yhat, levels);

            ResultOne = pack_method_result(ResultOne, 'OURSFUSION', ...
                time_train, Time4.epoch_s_mean, time_infer_total, time_infer_perSample, ...
                acc4, mae4, accpm1_4, Met4, GP4, Det4, Hist4);
            ResultOne.OURSFUSION_beta_val = beta_va_fusion(:);
            ResultOne.OURSFUSION_w_val    = w_va_fusion(:);
            ResultOne.OURSFUSION_b_val    = b_va_fusion(:);
            ResultOne.OURSFUSION_yhat_val = Det4.yhat(:);
            ResultOne.OURSFUSION_theta = theta_fusion(:);
            
                        % -------- extra bookkeeping for plots --------
            ResultOne.y_train = y_tr(:);
            ResultOne.train_idx = tr(:);

            if cfg.run_fusion_ablation && should_run_fusion_ablation(runCfg, cfg)
                AblationResults = run_fusion_ablation_suite( ...
                    AblationResults, runCfg, ...
                    MI_trs, MI_tr, Xdeg_trs, y_tr, ...
                    MI_vas, MI_va, Xdeg_vas, y_va, ...
                    M, levels);
            end

            Results = append_result_struct(Results, ResultOne);

            print_run_metrics(ResultOne, METHODS, runCfg);
        end
    end
end

%% ===================== SUMMARY =====================
fprintf('\n================ SUMMARY ================\n');

Summ = summarize_methods(Results, METHODS);
disp(Summ);

Summ_by_frac = build_split_frac_metric_table(Results, METHODS);
disp(Summ_by_frac);

Summ_wide_by_frac = build_split_frac_wide_table(Results, METHODS);
disp(Summ_wide_by_frac);

MainTab = build_main_paper_table(Results, cfg, METHODS);
disp(MainTab);

SigTab = paired_significance_table(Results, METHODS);
disp(SigTab);

if cfg.do_fewshot_triptych
    fprintf('\n================ FEW-SHOT TRIPTYCH =================\n');
    plot_fewshot_triptych(Results, cfg, METHODS);
end

if cfg.do_ood_risk_plots
    fprintf('\n================ OOD RISK ANALYSIS =================\n');
    plot_main_accuracy_by_split(Results, cfg, METHODS);
    plot_main_risk_by_split(Results, cfg, METHODS, 'highMiss');
    plot_complexity_performance_tradeoff(Results, cfg, METHODS);
end

if cfg.do_beta_risk_analysis
    fprintf('\n================ beta_t vs under-prediction risk ANALYSIS =================\n');
    BetaRiskTab = run_beta_risk_analysis(Results, cfg);
    disp(BetaRiskTab);
else
    BetaRiskTab = table();
end

if cfg.do_ess_entropy_analysis
    fprintf('\n================ ESS / ENTROPY ANALYSIS =================\n');
    ESSTab = run_ess_entropy_analysis(Results, cfg);
    disp(ESSTab);
else
    ESSTab = table();
end

if cfg.do_margin_stability_analysis
    fprintf('\n================ MARGIN STABILITY ANALYSIS =================\n');
    MarginOut = run_margin_stability_analysis(Results, cfg);
    if isfield(MarginOut, 'SummaryTab')
        disp(MarginOut.SummaryTab);
    end
else
    MarginOut = struct();
end

if cfg.highsnr.enable
    fprintf('\n================ HIGH-SNR CONDITIONAL MECHANISM =================\n');
    HighSNROut = run_highsnr_analysis_from_results(Results, cfg);
    if isfield(HighSNROut, 'CorrTab')
        disp(HighSNROut.CorrTab);
    end
else
    HighSNROut = struct();
end

if cfg.do_split_label_distribution
    fprintf('\n================ SPLIT LABEL DISTRIBUTION =================\n');
    SplitLabelTab = build_split_label_distribution_table(Results, levels);
    disp(SplitLabelTab);
    plot_split_label_distribution(Results, levels, cfg);
else
    SplitLabelTab = table();
end

if cfg.do_confusion_analysis
    fprintf('\n================ CONFUSION ANALYSIS =================\n');
    plot_confusion_analysis(Results, levels, cfg, METHODS);
end

if cfg.do_classwise_analysis
    fprintf('\n================ CLASS-WISE ANALYSIS =================\n');
    ClasswiseTab = build_classwise_metric_table(Results, levels, cfg, METHODS);
    disp(ClasswiseTab);
    plot_classwise_recall(Results, levels, cfg, METHODS);
else
    ClasswiseTab = table();
end

if cfg.do_reliability_analysis
    fprintf('\n================ RELIABILITY ANALYSIS =================\n');
    ReliabilityTab = build_reliability_table(Results, cfg);
    disp(ReliabilityTab);
    plot_reliability_analysis(Results, cfg, METHODS);
else
    ReliabilityTab = table();
end

if cfg.do_risk_coverage_plot
    fprintf('\n================ RISK-COVERAGE ANALYSIS =================\n');
    plot_risk_coverage_analysis(Results, cfg, METHODS);
end

if cfg.run_fusion_ablation
    fprintf('\n================ FUSION ABLATION =================\n');
    AblationTab = summarize_fusion_ablation(AblationResults);
    disp(AblationTab);
    plot_fusion_ablation(AblationResults, cfg);
else
    AblationTab = table();
end

if cfg.do_bootstrap_ci
    fprintf('\n================ BOOTSTRAP CI =================\n');
    BootstrapTab = build_bootstrap_ci_table(Results, cfg);
    disp(BootstrapTab);
else
    BootstrapTab = table();
end

if cfg.export_excel
    fprintf('\n================ EXPORT ALL RESULTS TO EXCEL =================\n');
    export_all_results_to_excel( ...
    Results, Summ, Summ_by_frac, Summ_wide_by_frac, cfg, ...
    MainTab, SigTab, BetaRiskTab, ESSTab, METHODS, ...
    SplitLabelTab, ClasswiseTab, ReliabilityTab, AblationTab, BootstrapTab, AblationResults, HighSNROut);
    fprintf('Excel exported: %s\n', cfg.excel_out);
end

save(fullfile(cfg.run_dir, 'final_workspace.mat'), ...
    'Results', 'Summ', 'cfg', 'levels', 'M', ...
    'MainTab', 'SigTab', 'Summ_by_frac', 'Summ_wide_by_frac', ...
    'BetaRiskTab', 'ESSTab', 'HighSNROut', 'MarginOut', ...
    'time_parse', 'time_buildH', 'time_preMI', 'time_pre_all', ...
    '-v7.3');

fprintf('\nAll outputs saved to:\n%s\n', cfg.run_dir);
fprintf('\nDone.\n');

diary off;

%% ============================================================
%  METHOD REGISTRY / PACK / PRINT
% ============================================================
function METHODS = get_method_registry()
METHODS = struct([]);

METHODS(1).key   = 'FIXED';
METHODS(1).label = 'MI-FIX';

METHODS(2).key   = 'FEATURE';
METHODS(2).label = 'DEG-MLP';

METHODS(3).key   = 'E2E';
METHODS(3).label = 'MI-DNN';

METHODS(4).key   = 'OURSNOB';
METHODS(4).label = 'MI-ADA';

METHODS(5).key   = 'OURSFUSION';
METHODS(5).label = 'MI-FUSE';
end

function R = pack_method_result(R, key, train_s, epoch_s_mean, infer_total_s, infer_perSample_s, ...
    acc, mae, accpm1, Met, GP, Det, Hist)

R.([key '_train_s']) = train_s;
R.([key '_epoch_s_mean']) = epoch_s_mean;
R.([key '_infer_total_s']) = infer_total_s;
R.([key '_infer_us_perSample']) = 1e6 * infer_perSample_s;

R.([key '_acc']) = acc;
R.([key '_mae']) = mae;
R.([key '_accpm1']) = accpm1;

R.([key '_lowMiss']) = Met.lowMiss;
R.([key '_highMiss']) = Met.highMiss;
R.([key '_ocGap']) = Met.ocGap;
R.([key '_nll']) = Met.nll;
R.([key '_brier']) = Met.brier;
R.([key '_ece']) = Met.ece;
R.([key '_auroc_err']) = Met.auroc_err;
R.([key '_aurc']) = Met.aurc;

R.([key '_goodput']) = GP.goodput;
R.([key '_goodput_norm']) = GP.goodput_norm;
R.([key '_outage']) = GP.outage;
R.([key '_underuse']) = GP.underuse;

R.(['Det_' key]) = Det;
R.(['Hist_' key]) = Hist;
end

function print_run_metrics(R, METHODS, cfg)
for i = 1:numel(METHODS)
    key = METHODS(i).key;
    lbl = METHODS(i).label;

    fprintf('[VAL] %-10s : Acc=%.4f MAE=%.4f Acc_pm1=%.4f | lowMiss=%.4f highMiss=%.4f | ECE=%.4f | GPn=%.4f\n', ...
        lbl, ...
        R.([key '_acc']), ...
        R.([key '_mae']), ...
        R.([key '_accpm1']), ...
        R.([key '_lowMiss']), ...
        R.([key '_highMiss']), ...
        R.([key '_ece']), ...
        R.([key '_goodput_norm']));
end

if cfg.time_print_each_run
    for i = 1:numel(METHODS)
        key = METHODS(i).key;
        lbl = METHODS(i).label;
        fprintf('[TIME] %-10s train=%.3fs (epoch=%.4fs) infer=%.4fs (perSample=%.3fus)\n', ...
            lbl, ...
            R.([key '_train_s']), ...
            R.([key '_epoch_s_mean']), ...
            R.([key '_infer_total_s']), ...
            R.([key '_infer_us_perSample']));
    end
end
end

%% ============================================================
%  SUMMARY / EXPORT / TABLES
% ============================================================
function Summ = summarize_methods(Results, METHODS)
if isempty(Results), Summ = table(); return; end

Summ = table();
for i = 1:numel(METHODS)
    key = METHODS(i).key;
    lbl = matlab.lang.makeValidName(METHODS(i).label);

    acc = collect_field(Results, [key '_acc']);
    mae = collect_field(Results, [key '_mae']);
    lowm = collect_field(Results, [key '_lowMiss']);
    ece = collect_field(Results, [key '_ece']);
    gpn = collect_field(Results, [key '_goodput_norm']);
    outg = collect_field(Results, [key '_outage']);
    trt = collect_field(Results, [key '_train_s']);
    infs = collect_field(Results, [key '_infer_us_perSample']);

    Summ.([lbl '_Acc_mean']) = mean(acc,'omitnan');
    Summ.([lbl '_Acc_std'])  = std(acc,'omitnan');
    Summ.([lbl '_MAE_mean']) = mean(mae,'omitnan');
    Summ.([lbl '_lowMiss_mean']) = mean(lowm,'omitnan');
    Summ.([lbl '_ECE_mean']) = mean(ece,'omitnan');
    Summ.([lbl '_goodput_norm_mean']) = mean(gpn,'omitnan');
    Summ.([lbl '_outage_mean']) = mean(outg,'omitnan');
    Summ.([lbl '_train_s_mean']) = mean(trt,'omitnan');
    Summ.([lbl '_infer_us_mean']) = mean(infs,'omitnan');
end
end

function export_all_results_to_excel(Results, Summ, Summ_by_frac, Summ_wide_by_frac, cfg, MainTab, SigTab, BetaRiskTab, ESSTab, METHODS, SplitLabelTab, ClasswiseTab, ReliabilityTab, AblationTab, BootstrapTab, AblationResults, HighSNROut)
excelFile = cfg.excel_out;

if cfg.export_overwrite
    if exist(excelFile, 'file'), delete(excelFile); end
end

T_runs = flatten_results_for_export(Results, METHODS);
safe_writetable(T_runs, excelFile, 'runs_all');

if ~isempty(Summ), safe_writetable(Summ, excelFile, 'summary_all'); end

T_split = build_split_metric_table(Results, METHODS);
safe_writetable(T_split, excelFile, 'summary_by_split');

T_drop = build_ood_drop_table(Results, METHODS);
if ~isempty(T_drop), safe_writetable(T_drop, excelFile, 'ood_drop_vs_random'); end

[T_acc, T_gp, T_low, T_high, T_ece, T_auroc] = build_ood_metric_tables(Results, METHODS);
if ~isempty(T_acc),   safe_writetable(T_acc,   excelFile, 'ood_metric_acc'); end
if ~isempty(T_gp),    safe_writetable(T_gp,    excelFile, 'ood_metric_goodput'); end
if ~isempty(T_low),   safe_writetable(T_low,   excelFile, 'ood_metric_lowMiss'); end
if ~isempty(T_high),  safe_writetable(T_high,  excelFile, 'ood_metric_highMiss'); end
if ~isempty(T_ece),   safe_writetable(T_ece,   excelFile, 'ood_metric_ece'); end
if ~isempty(T_auroc), safe_writetable(T_auroc, excelFile, 'ood_metric_auroc'); end

if ~isempty(MainTab), safe_writetable(MainTab, excelFile, 'main_paper_table'); end
if ~isempty(SigTab),  safe_writetable(SigTab,  excelFile, 'paired_significance'); end
if ~isempty(BetaRiskTab), safe_writetable(BetaRiskTab, excelFile, 'beta_risk_analysis'); end
if ~isempty(ESSTab),      safe_writetable(ESSTab,      excelFile, 'ess_entropy_analysis'); end
if ~isempty(Summ_by_frac), safe_writetable(Summ_by_frac, excelFile, 'summary_by_split_frac'); end
if ~isempty(Summ_wide_by_frac), safe_writetable(Summ_wide_by_frac, excelFile, 'summary_wide_split_frac'); end

if ~isempty(SplitLabelTab),  safe_writetable(SplitLabelTab,  excelFile, 'split_label_dist'); end
if ~isempty(ClasswiseTab),   safe_writetable(ClasswiseTab,   excelFile, 'classwise_metrics'); end
if ~isempty(ReliabilityTab), safe_writetable(ReliabilityTab, excelFile, 'reliability'); end
if ~isempty(AblationTab),    safe_writetable(AblationTab,    excelFile, 'fusion_ablation'); end
if ~isempty(BootstrapTab),   safe_writetable(BootstrapTab,   excelFile, 'bootstrap_ci'); end

if ~isempty(AblationResults)
    T_ab_runs = flatten_ablation_results(AblationResults);
    safe_writetable(T_ab_runs, excelFile, 'ablation_runs');
end

if isfield(HighSNROut,'CorrTab') && ~isempty(HighSNROut.CorrTab)
    safe_writetable(HighSNROut.CorrTab, excelFile, 'highsnr_corr');
end
if isfield(HighSNROut,'SubsetSummary') && ~isempty(HighSNROut.SubsetSummary)
    safe_writetable(HighSNROut.SubsetSummary, excelFile, 'highsnr_subset_summary');
end
if isfield(HighSNROut,'BinTab') && ~isempty(HighSNROut.BinTab)
    safe_writetable(HighSNROut.BinTab, excelFile, 'highsnr_gap_binning');
end
end

function T = flatten_results_for_export(Results, METHODS)
if isempty(Results), T = table(); return; end
n = numel(Results);

split = strings(n,1);
frac  = nan(n,1);
seed  = nan(n,1);

pipe_pre_all_s = nan(n,1);
pipe_parse_s   = nan(n,1);
pipe_buildH_s  = nan(n,1);
pipe_preMI_s   = nan(n,1);

metricNames = { ...
    'train_s','epoch_s_mean','infer_total_s','infer_us_perSample', ...
    'acc','mae','accpm1','lowMiss','highMiss','ocGap', ...
    'nll','brier','ece','auroc_err','aurc', ...
    'goodput','goodput_norm','outage','underuse'};

S = struct();
for m = 1:numel(METHODS)
    label = matlab.lang.makeValidName(METHODS(m).label);
    key   = METHODS(m).key;
    for k = 1:numel(metricNames)
        S.([label '_' metricNames{k}]) = nan(n,1);
    end
    for i = 1:n
        if isfield(Results(i),[key '_' metricNames{1}])
            for kk = 1:numel(metricNames)
                fnR = [key '_' metricNames{kk}];
                fnT = [label '_' metricNames{kk}];
                if isfield(Results(i), fnR)
                    S.(fnT)(i) = Results(i).(fnR);
                end
            end
        end
    end
end

FIXED_best_beta = nan(n,1);
FIXED_best_w    = nan(n,1);

for i = 1:n
    r = Results(i);

    if isfield(r,'split'), split(i) = string(r.split); end
    if isfield(r,'frac'),  frac(i)  = r.frac; end
    if isfield(r,'seed'),  seed(i)  = r.seed; end

    if isfield(r,'pipe_pre_all_s'), pipe_pre_all_s(i) = r.pipe_pre_all_s; end
    if isfield(r,'pipe_parse_s'),   pipe_parse_s(i)   = r.pipe_parse_s; end
    if isfield(r,'pipe_buildH_s'),  pipe_buildH_s(i)  = r.pipe_buildH_s; end
    if isfield(r,'pipe_preMI_s'),   pipe_preMI_s(i)   = r.pipe_preMI_s; end

    if isfield(r,'FIXED_best_beta'), FIXED_best_beta(i) = r.FIXED_best_beta; end
    if isfield(r,'FIXED_best_w'),    FIXED_best_w(i)    = r.FIXED_best_w; end
end

T = table(split, frac, seed, pipe_pre_all_s, pipe_parse_s, pipe_buildH_s, pipe_preMI_s);
fn = fieldnames(S);
for i = 1:numel(fn)
    T.(fn{i}) = S.(fn{i});
end
T.FIXED_best_beta = FIXED_best_beta;
T.FIXED_best_w    = FIXED_best_w;
end

function T = build_split_metric_table(Results, METHODS)
if isempty(Results), T = table(); return; end

splitNames = unique(string({Results.split}), 'stable');
nRows = numel(splitNames) * numel(METHODS);

split_col   = strings(nRows,1);
method_col  = strings(nRows,1);
nRuns_col   = nan(nRows,1);
acc_mean    = nan(nRows,1);
acc_std     = nan(nRows,1);
mae_mean    = nan(nRows,1);
low_mean    = nan(nRows,1);
high_mean   = nan(nRows,1);
ece_mean    = nan(nRows,1);
auroc_mean  = nan(nRows,1);
aurc_mean   = nan(nRows,1);
goodput_mean = nan(nRows,1);
outage_mean  = nan(nRows,1);
underuse_mean = nan(nRows,1);
train_mean = nan(nRows,1);
infer_mean = nan(nRows,1);

rows = 0;
for s = 1:numel(splitNames)
    idxS = strcmp({Results.split}, splitNames{s});
    Rsub = Results(idxS);

    for m = 1:numel(METHODS)
        rows = rows + 1;
        key = METHODS(m).key;
        lbl = METHODS(m).label;

        acc   = collect_field(Rsub, [key '_acc']);
        mae   = collect_field(Rsub, [key '_mae']);
        lowm  = collect_field(Rsub, [key '_lowMiss']);
        highm = collect_field(Rsub, [key '_highMiss']);
        ece   = collect_field(Rsub, [key '_ece']);
        auc   = collect_field(Rsub, [key '_auroc_err']);
        aurc  = collect_field(Rsub, [key '_aurc']);
        gp    = collect_field(Rsub, [key '_goodput_norm']);
        outg  = collect_field(Rsub, [key '_outage']);
        under = collect_field(Rsub, [key '_underuse']);
        trt   = collect_field(Rsub, [key '_train_s']);
        infs  = collect_field(Rsub, [key '_infer_us_perSample']);

        split_col(rows)  = splitNames(s);
        method_col(rows) = string(lbl);
        nRuns_col(rows)  = sum(isfinite(acc));

        acc_mean(rows)   = mean(acc,'omitnan');
        acc_std(rows)    = std(acc,'omitnan');
        mae_mean(rows)   = mean(mae,'omitnan');
        low_mean(rows)   = mean(lowm,'omitnan');
        high_mean(rows)  = mean(highm,'omitnan');
        ece_mean(rows)   = mean(ece,'omitnan');
        auroc_mean(rows) = mean(auc,'omitnan');
        aurc_mean(rows)  = mean(aurc,'omitnan');
        goodput_mean(rows) = mean(gp,'omitnan');
        outage_mean(rows)  = mean(outg,'omitnan');
        underuse_mean(rows)= mean(under,'omitnan');
        train_mean(rows) = mean(trt,'omitnan');
        infer_mean(rows) = mean(infs,'omitnan');
    end
end

T = table(split_col, method_col, nRuns_col, ...
    acc_mean, acc_std, mae_mean, low_mean, high_mean, ...
    ece_mean, auroc_mean, aurc_mean, ...
    goodput_mean, outage_mean, underuse_mean, ...
    train_mean, infer_mean, ...
    'VariableNames', {'split','method','nRuns', ...
    'acc_mean','acc_std','mae_mean','lowMiss_mean','highMiss_mean', ...
    'ece_mean','auroc_mean','aurc_mean', ...
    'goodput_norm_mean','outage_mean','underuse_mean', ...
    'train_s_mean','infer_us_mean'});
end

function T = build_ood_drop_table(Results, METHODS)
T = table();
if isempty(Results), return; end

splitNames = unique(string({Results.split}), 'stable');
if ~any(strcmp(splitNames, "random")), return; end

idxRand = strcmp({Results.split}, 'random');
Rrand = Results(idxRand);

baseAcc = nan(1, numel(METHODS));
for m = 1:numel(METHODS)
    baseAcc(m) = mean(collect_field(Rrand, [METHODS(m).key '_acc']), 'omitnan');
end

oodSplits = splitNames(~strcmp(splitNames, "random"));
nRows = numel(oodSplits) * numel(METHODS);

split_col = strings(nRows,1);
method_col = strings(nRows,1);
random_acc_mean = nan(nRows,1);
ood_acc_mean = nan(nRows,1);
acc_drop_vs_random = nan(nRows,1);

rows = 0;
for s = 1:numel(oodSplits)
    idxS = strcmp({Results.split}, oodSplits{s});
    Rsub = Results(idxS);

    for m = 1:numel(METHODS)
        rows = rows + 1;
        key = METHODS(m).key;
        lbl = METHODS(m).label;
        accMean = mean(collect_field(Rsub, [key '_acc']), 'omitnan');

        split_col(rows) = oodSplits(s);
        method_col(rows) = string(lbl);
        random_acc_mean(rows) = baseAcc(m);
        ood_acc_mean(rows) = accMean;
        acc_drop_vs_random(rows) = baseAcc(m) - accMean;
    end
end

T = table(split_col, method_col, random_acc_mean, ood_acc_mean, acc_drop_vs_random, ...
    'VariableNames', {'split','method','random_acc_mean','ood_acc_mean','acc_drop_vs_random'});
end

function [T_acc, T_gp, T_low, T_high, T_ece, T_auroc] = build_ood_metric_tables(Results, METHODS)
T_acc   = build_metric_matrix_table(Results, METHODS, 'acc');
T_gp    = build_metric_matrix_table(Results, METHODS, 'goodput_norm');
T_low   = build_metric_matrix_table(Results, METHODS, 'lowMiss');
T_high  = build_metric_matrix_table(Results, METHODS, 'highMiss');
T_ece   = build_metric_matrix_table(Results, METHODS, 'ece');
T_auroc = build_metric_matrix_table(Results, METHODS, 'auroc_err');
end

function T = build_metric_matrix_table(Results, METHODS, metricName)
if isempty(Results), T = table(); return; end

splitNames = unique(string({Results.split}), 'stable');
T = table();
T.split = splitNames(:);

for m = 1:numel(METHODS)
    key = METHODS(m).key;
    lbl = matlab.lang.makeValidName(METHODS(m).label);
    col = nan(numel(splitNames),1);

    for s = 1:numel(splitNames)
        idxS = strcmp({Results.split}, splitNames{s});
        Rsub = Results(idxS);
        vals = collect_field(Rsub, [key '_' metricName]);
        col(s) = mean(vals, 'omitnan');
    end
    T.(lbl) = col;
end
end

function MainTab = build_main_paper_table(Results, cfg, METHODS)
if isempty(Results), MainTab = table(); return; end

idxUse = arrayfun(@(r) abs(r.frac - 1.0) < 1e-12, Results);
R = Results(idxUse);

mainSplits = string(cfg.split_modes_main);
idxKeep = ismember(string({R.split}), mainSplits);
R = R(idxKeep);

splitNames = unique(string({R.split}), 'stable');
nRows = numel(splitNames) * numel(METHODS);

split_col = strings(nRows,1);
method_col = strings(nRows,1);
acc_mean = nan(nRows,1);
acc_std = nan(nRows,1);
highMiss_mean = nan(nRows,1);
ece_mean = nan(nRows,1);
goodput_norm_mean = nan(nRows,1);
train_s_mean = nan(nRows,1);
infer_us_mean = nan(nRows,1);
lowMiss_mean = nan(nRows,1);

rows = 0;
for s = 1:numel(splitNames)
    idxS = strcmp({R.split}, splitNames{s});
    Rs = R(idxS);

    for m = 1:numel(METHODS)
        key = METHODS(m).key;
        lbl = METHODS(m).label;
        rows = rows + 1;

        split_col(rows) = splitNames(s);
        method_col(rows) = string(lbl);

        acc_mean(rows) = mean(collect_field(Rs,[key '_acc']),'omitnan');
        acc_std(rows)  = std(collect_field(Rs,[key '_acc']),'omitnan');
        highMiss_mean(rows) = mean(collect_field(Rs,[key '_highMiss']),'omitnan');
        ece_mean(rows) = mean(collect_field(Rs,[key '_ece']),'omitnan');
        goodput_norm_mean(rows) = mean(collect_field(Rs,[key '_goodput_norm']),'omitnan');
        train_s_mean(rows) = mean(collect_field(Rs,[key '_train_s']),'omitnan');
        infer_us_mean(rows) = mean(collect_field(Rs,[key '_infer_us_perSample']),'omitnan');
        lowMiss_mean(rows) = mean(collect_field(Rs,[key '_lowMiss']),'omitnan');
    end
end

MainTab = table(split_col, method_col, acc_mean, acc_std, highMiss_mean, ...
    ece_mean, goodput_norm_mean, train_s_mean, infer_us_mean, lowMiss_mean, ...
    'VariableNames', {'split','method','acc_mean','acc_std','highMiss_mean', ...
    'ece_mean','goodput_norm_mean','train_s_mean','infer_us_mean','lowMiss_mean'});
end

function SigTab = paired_significance_table(Results, METHODS)
if isempty(Results), SigTab = table(); return; end

idxUse = arrayfun(@(r) abs(r.frac - 1.0) < 1e-12, Results);
R = Results(idxUse);

splitNames = unique(string({R.split}), 'stable');
pairs = {
    'OURSFUSION','OURSNOB';
    'OURSFUSION','E2E';
    'OURSFUSION','FEATURE';
    'OURSFUSION','FIXED';
    'OURSNOB','E2E'
};

maxRows = numel(splitNames) * size(pairs,1);

split_col = strings(maxRows,1);
compare_col = strings(maxRows,1);
delta_acc_mean = nan(maxRows,1);
p_acc = nan(maxRows,1);
delta_highMiss_mean = nan(maxRows,1);
p_highMiss = nan(maxRows,1);
delta_goodput_mean = nan(maxRows,1);
p_goodput = nan(maxRows,1);

rows = 0;
for s = 1:numel(splitNames)
    idxS = strcmp({R.split}, splitNames{s});
    Rs = R(idxS);

    for p = 1:size(pairs,1)
        A = pairs{p,1};
        B = pairs{p,2};

        a_acc = collect_field(Rs, [A '_acc']);
        b_acc = collect_field(Rs, [B '_acc']);
        a_hm  = collect_field(Rs, [A '_highMiss']);
        b_hm  = collect_field(Rs, [B '_highMiss']);
        a_gp  = collect_field(Rs, [A '_goodput_norm']);
        b_gp  = collect_field(Rs, [B '_goodput_norm']);

        rows = rows + 1;
        split_col(rows) = splitNames(s);

        labelA = method_key_to_label(A, METHODS);
        labelB = method_key_to_label(B, METHODS);
        compare_col(rows) = labelA + "_vs_" + labelB;

        ok_acc = isfinite(a_acc) & isfinite(b_acc);
        ok_hm  = isfinite(a_hm)  & isfinite(b_hm);
        ok_gp  = isfinite(a_gp)  & isfinite(b_gp);

        if nnz(ok_acc) >= 2
            delta_acc_mean(rows) = mean(a_acc(ok_acc) - b_acc(ok_acc), 'omitnan');
            p_acc(rows) = signrank(a_acc(ok_acc), b_acc(ok_acc));
        end

        if nnz(ok_hm) >= 2
            delta_highMiss_mean(rows) = mean(a_hm(ok_hm) - b_hm(ok_hm), 'omitnan');
            p_highMiss(rows) = signrank(a_hm(ok_hm), b_hm(ok_hm));
        end

        if nnz(ok_gp) >= 2
            delta_goodput_mean(rows) = mean(a_gp(ok_gp) - b_gp(ok_gp), 'omitnan');
            p_goodput(rows) = signrank(a_gp(ok_gp), b_gp(ok_gp));
        end
    end
end

SigTab = table(split_col(1:rows), compare_col(1:rows), ...
    delta_acc_mean(1:rows), p_acc(1:rows), ...
    delta_highMiss_mean(1:rows), p_highMiss(1:rows), ...
    delta_goodput_mean(1:rows), p_goodput(1:rows), ...
    'VariableNames', {'split','compare','delta_acc_mean','p_acc', ...
    'delta_highMiss_mean','p_highMiss','delta_goodput_mean','p_goodput'});
end

function T = build_split_frac_metric_table(Results, METHODS)
if isempty(Results), T = table(); return; end

splitNames = unique(string({Results.split}), 'stable');
fracVals   = unique([Results.frac]);

maxRows = numel(splitNames) * numel(fracVals) * numel(METHODS);

split_col   = strings(maxRows,1);
frac_col    = nan(maxRows,1);
method_col  = strings(maxRows,1);
nRuns_col   = nan(maxRows,1);

acc_mean    = nan(maxRows,1);
acc_std     = nan(maxRows,1);
mae_mean    = nan(maxRows,1);
mae_std     = nan(maxRows,1);
accpm1_mean = nan(maxRows,1);
low_mean    = nan(maxRows,1);
high_mean   = nan(maxRows,1);
ece_mean    = nan(maxRows,1);
auroc_mean  = nan(maxRows,1);
aurc_mean   = nan(maxRows,1);
goodput_mean  = nan(maxRows,1);
outage_mean   = nan(maxRows,1);
underuse_mean = nan(maxRows,1);
train_mean  = nan(maxRows,1);
infer_mean  = nan(maxRows,1);

rows = 0;
for s = 1:numel(splitNames)
    for f = 1:numel(fracVals)
        idxSF = strcmp({Results.split}, splitNames{s}) & ...
                arrayfun(@(r) abs(r.frac - fracVals(f)) < 1e-12, Results);
        Rsub = Results(idxSF);
        if isempty(Rsub), continue; end

        for m = 1:numel(METHODS)
            key = METHODS(m).key;
            lbl = METHODS(m).label;

            acc   = collect_field(Rsub, [key '_acc']);
            mae   = collect_field(Rsub, [key '_mae']);
            accp1 = collect_field(Rsub, [key '_accpm1']);
            lowm  = collect_field(Rsub, [key '_lowMiss']);
            highm = collect_field(Rsub, [key '_highMiss']);
            ece   = collect_field(Rsub, [key '_ece']);
            auc   = collect_field(Rsub, [key '_auroc_err']);
            aurc  = collect_field(Rsub, [key '_aurc']);
            gp    = collect_field(Rsub, [key '_goodput_norm']);
            outg  = collect_field(Rsub, [key '_outage']);
            under = collect_field(Rsub, [key '_underuse']);
            trt   = collect_field(Rsub, [key '_train_s']);
            infs  = collect_field(Rsub, [key '_infer_us_perSample']);

            rows = rows + 1;
            split_col(rows) = splitNames(s);
            frac_col(rows)  = fracVals(f);
            method_col(rows)= string(lbl);
            nRuns_col(rows) = sum(isfinite(acc));

            acc_mean(rows)    = mean(acc,'omitnan');
            acc_std(rows)     = std(acc,'omitnan');
            mae_mean(rows)    = mean(mae,'omitnan');
            mae_std(rows)     = std(mae,'omitnan');
            accpm1_mean(rows) = mean(accp1,'omitnan');
            low_mean(rows)    = mean(lowm,'omitnan');
            high_mean(rows)   = mean(highm,'omitnan');
            ece_mean(rows)    = mean(ece,'omitnan');
            auroc_mean(rows)  = mean(auc,'omitnan');
            aurc_mean(rows)   = mean(aurc,'omitnan');
            goodput_mean(rows)= mean(gp,'omitnan');
            outage_mean(rows) = mean(outg,'omitnan');
            underuse_mean(rows)=mean(under,'omitnan');
            train_mean(rows)  = mean(trt,'omitnan');
            infer_mean(rows)  = mean(infs,'omitnan');
        end
    end
end

T = table( ...
    split_col(1:rows), frac_col(1:rows), method_col(1:rows), nRuns_col(1:rows), ...
    acc_mean(1:rows), acc_std(1:rows), mae_mean(1:rows), mae_std(1:rows), ...
    accpm1_mean(1:rows), low_mean(1:rows), high_mean(1:rows), ...
    ece_mean(1:rows), auroc_mean(1:rows), aurc_mean(1:rows), ...
    goodput_mean(1:rows), outage_mean(1:rows), underuse_mean(1:rows), ...
    train_mean(1:rows), infer_mean(1:rows), ...
    'VariableNames', { ...
    'split','frac','method','nRuns', ...
    'acc_mean','acc_std','mae_mean','mae_std','accpm1_mean', ...
    'lowMiss_mean','highMiss_mean','ece_mean','auroc_mean','aurc_mean', ...
    'goodput_norm_mean','outage_mean','underuse_mean','train_s_mean','infer_us_mean'});
end

function T = build_split_frac_wide_table(Results, METHODS)
if isempty(Results), T = table(); return; end

splitNames = unique(string({Results.split}), 'stable');
fracVals   = unique([Results.frac]);

maxRows = numel(splitNames) * numel(fracVals);
split_col = strings(maxRows,1);
frac_col  = nan(maxRows,1);

metrics = {'acc','mae','highMiss','ece','goodput_norm'};
S = struct();
for i = 1:numel(METHODS)
    lbl = matlab.lang.makeValidName(METHODS(i).label);
    for j = 1:numel(metrics)
        S.([lbl '_' metrics{j}]) = nan(maxRows,1);
    end
end

rows = 0;
for s = 1:numel(splitNames)
    for f = 1:numel(fracVals)
        idxSF = strcmp({Results.split}, splitNames{s}) & ...
                arrayfun(@(r) abs(r.frac - fracVals(f)) < 1e-12, Results);
        Rsub = Results(idxSF);
        if isempty(Rsub), continue; end

        rows = rows + 1;
        split_col(rows) = splitNames(s);
        frac_col(rows) = fracVals(f);

        for m = 1:numel(METHODS)
            key = METHODS(m).key;
            lbl = matlab.lang.makeValidName(METHODS(m).label);

            S.([lbl '_acc'])(rows) = mean(collect_field(Rsub, [key '_acc']), 'omitnan');
            S.([lbl '_mae'])(rows) = mean(collect_field(Rsub, [key '_mae']), 'omitnan');
            S.([lbl '_highMiss'])(rows) = mean(collect_field(Rsub, [key '_highMiss']), 'omitnan');
            S.([lbl '_ece'])(rows) = mean(collect_field(Rsub, [key '_ece']), 'omitnan');
            S.([lbl '_goodput_norm'])(rows) = mean(collect_field(Rsub, [key '_goodput_norm']), 'omitnan');
        end
    end
end

T = table(split_col(1:rows), frac_col(1:rows), 'VariableNames', {'split','frac'});
fn = fieldnames(S);
for i = 1:numel(fn)
    T.(fn{i}) = S.(fn{i})(1:rows);
end
end

%% ============================================================
%  PLOTS
% ============================================================
function plot_main_accuracy_by_split(Results, cfg, METHODS)
if isempty(Results), return; end

splitNames = unique(string({Results.split}), 'stable');
x = 1:numel(splitNames);

% ---------- compute mean/std first ----------
mu_all = nan(numel(splitNames), numel(METHODS));
sd_all = nan(numel(splitNames), numel(METHODS));

for m = 1:numel(METHODS)
    for s = 1:numel(splitNames)
        idxS = strcmp({Results.split}, splitNames{s});
        Rsub = Results(idxS);
        vals = collect_field(Rsub, [METHODS(m).key '_acc']);
        mu_all(s,m) = mean(vals, 'omitnan');
        sd_all(s,m) = std(vals, 'omitnan');
    end
end

% ---------- pretty split labels ----------
xTickLabels = pretty_split_labels(splitNames);

% ---------- figure ----------
fig = figure('Color','w','Position',[120 120 860 500]);
hold on; box off;

% Main methods first in legend order
drawOrder = {'FEATURE','E2E','OURSFUSION','OURSNOB','FIXED'};

hMap = containers.Map();
for ii = 1:numel(drawOrder)
    key = drawOrder{ii};
    m = find(strcmp({METHODS.key}, key), 1);
    if isempty(m), continue; end

    sty = method_style_twc(METHODS(m).key);

    h = errorbar(x, mu_all(:,m), sd_all(:,m), ...
        'LineStyle', '-', ...
        'Color', sty.color, ...
        'LineWidth', sty.lineWidth, ...
        'Marker', sty.marker, ...
        'MarkerSize', sty.markerSize, ...
        'MarkerFaceColor', sty.markerFaceColor, ...
        'MarkerEdgeColor', sty.color, ...
        'CapSize', sty.capSize);

    hMap(METHODS(m).key) = h;
end

% ---------- axes ----------
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 14;
ax.LineWidth = 1.0;
ax.TickDir = 'out';
ax.TickLength = [0.012 0.012];
ax.XColor = [0.15 0.15 0.15];
ax.YColor = [0.15 0.15 0.15];
ax.XLim = [0.8, numel(splitNames)+0.2];
ax.XTick = x;
ax.XTickLabel = xTickLabels;

% dynamic y-limit, but keep nice margins
ymin = min(mu_all(:) - sd_all(:), [], 'omitnan');
ymax = max(mu_all(:) + sd_all(:), [], 'omitnan');

if ~isfinite(ymin), ymin = 0.6; end
if ~isfinite(ymax), ymax = 1.0; end

ymin = max(0, floor((ymin - 0.02)*20)/20);
ymax = min(1.0, ceil((ymax + 0.01)*20)/20);

if ymax - ymin < 0.15
    ymin = max(0, ymin - 0.03);
end

ax.YLim = [ymin, ymax];

% only light y-grid
grid on;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.GridColor = [0.86 0.86 0.86];
ax.GridAlpha = 1.0;
ax.MinorGridAlpha = 0;
ax.Layer = 'top';

ylabel('Accuracy', 'FontName','Times New Roman', 'FontSize', 17);
xlabel('Split protocol', 'FontName','Times New Roman', 'FontSize', 17);
title('Accuracy across distribution splits', ...
    'FontName','Times New Roman', 'FontSize', 16, 'FontWeight', 'normal');

% ---------- legend ----------
legendHandles = [];
legendLabels  = {};

legendKeyOrder = {'FEATURE','E2E','OURSFUSION','OURSNOB','FIXED'};
for ii = 1:numel(legendKeyOrder)
    k = legendKeyOrder{ii};
    if isKey(hMap, k)
        legendHandles(end+1) = hMap(k); %#ok<AGROW>
        legendLabels{end+1} = method_key_to_label_local(k); %#ok<AGROW>
    end
end

lgd = legend(legendHandles, legendLabels, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'NumColumns', 3);
lgd.Box = 'off';
lgd.FontName = 'Times New Roman';
lgd.FontSize = 13;

hold off;

if cfg.save_analysis_fig
    save_fig_paper(fig, cfg.analysis_fig_dir, 'Fig_Main_Accuracy_Across_Splits');
end
end

function labels = pretty_split_labels(splitNames)
labels = cell(numel(splitNames),1);
for i = 1:numel(splitNames)
    s = lower(strtrim(char(splitNames(i))));
    switch s
        case 'random'
            labels{i} = 'Random';
        case 'nf_ood'
            labels{i} = 'NF-OOD';
        case 'gap_ood'
            labels{i} = 'Gap-OOD';
        case 'snr_gap_ood'
            labels{i} = 'SNR+Gap-OOD';
        otherwise
            s = strrep(s, '_', '-');
            labels{i} = s;
    end
end
end

function sty = method_style_twc(methodKey)
% TWC-style visual hierarchy:
% - highlight MI-FUSE
% - keep DEG-MLP / MI-DNN clear
% - de-emphasize MI-FIX / MI-ADA

switch upper(methodKey)
    case 'FIXED'
        sty.color = [0.72 0.72 0.72];
        sty.marker = 'o';
        sty.lineWidth = 1.4;
        sty.markerSize = 6.5;
        sty.markerFaceColor = [0.72 0.72 0.72];
        sty.capSize = 7;

    case 'FEATURE'
        sty.color = [0.20 0.60 0.24];
        sty.marker = '^';
        sty.lineWidth = 2.1;
        sty.markerSize = 7.5;
        sty.markerFaceColor = [0.20 0.60 0.24];
        sty.capSize = 7;

    case 'E2E'
        sty.color = [0.93 0.49 0.19];
        sty.marker = 'd';
        sty.lineWidth = 2.1;
        sty.markerSize = 7.5;
        sty.markerFaceColor = [0.93 0.49 0.19];
        sty.capSize = 7;

    case 'OURSNOB'
        sty.color = [0.32 0.44 0.84];
        sty.marker = 's';
        sty.lineWidth = 1.6;
        sty.markerSize = 7.0;
        sty.markerFaceColor = [0.32 0.44 0.84];
        sty.capSize = 7;

    case 'OURSFUSION'
        sty.color = [0.65 0.12 0.20];
        sty.marker = 'p';
        sty.lineWidth = 2.6;
        sty.markerSize = 8.5;
        sty.markerFaceColor = [0.65 0.12 0.20];
        sty.capSize = 7;

    otherwise
        sty.color = [0.2 0.2 0.2];
        sty.marker = 'o';
        sty.lineWidth = 1.8;
        sty.markerSize = 7;
        sty.markerFaceColor = [0.2 0.2 0.2];
        sty.capSize = 7;
end
end

function lbl = method_key_to_label_local(key)
switch upper(key)
    case 'FIXED'
        lbl = 'MI-FIX';
    case 'FEATURE'
        lbl = 'DEG-MLP';
    case 'E2E'
        lbl = 'MI-DNN';
    case 'OURSNOB'
        lbl = 'MI-ADA';
    case 'OURSFUSION'
        lbl = 'MI-FUSE';
    otherwise
        lbl = key;
end
end

function plot_main_risk_by_split(Results, cfg, METHODS, riskName)
if nargin < 4 || isempty(riskName), riskName = 'highMiss'; end
if isempty(Results), return; end

splitNames = unique(string({Results.split}), 'stable');
x = 1:numel(splitNames);
xTickLabels = pretty_split_labels(splitNames);

switch lower(riskName)
    case 'highmiss'
        ylab = 'High-MCS miss rate';
        ttl  = 'Directional risk across distribution splits';
        fieldSuffix = 'highMiss';
        fileTag = 'HighMiss';
    case 'lowmiss'
        ylab = 'Low-MCS miss rate';
        ttl  = 'Low-rate risk across distribution splits';
        fieldSuffix = 'lowMiss';
        fileTag = 'LowMiss';
    otherwise
        error('Unknown riskName: %s', riskName);
end

mu_all = nan(numel(splitNames), numel(METHODS));
sd_all = nan(numel(splitNames), numel(METHODS));

for m = 1:numel(METHODS)
    for s = 1:numel(splitNames)
        idxS = strcmp({Results.split}, splitNames{s});
        Rsub = Results(idxS);
        vals = collect_field(Rsub, [METHODS(m).key '_' fieldSuffix]);
        mu_all(s,m) = mean(vals, 'omitnan');
        sd_all(s,m) = std(vals, 'omitnan');
    end
end

fig = figure('Color','w','Position',[120 120 860 500]);
hold on; box off;

drawOrder = {'FEATURE','E2E','OURSFUSION','OURSNOB','FIXED'};
hMap = containers.Map();

for ii = 1:numel(drawOrder)
    key = drawOrder{ii};
    m = find(strcmp({METHODS.key}, key), 1);
    if isempty(m), continue; end

    sty = method_style_twc(METHODS(m).key);

    h = errorbar(x, mu_all(:,m), sd_all(:,m), ...
        'LineStyle', '-', ...
        'Color', sty.color, ...
        'LineWidth', sty.lineWidth, ...
        'Marker', sty.marker, ...
        'MarkerSize', sty.markerSize, ...
        'MarkerFaceColor', sty.markerFaceColor, ...
        'MarkerEdgeColor', sty.color, ...
        'CapSize', sty.capSize);

    hMap(METHODS(m).key) = h;
end

ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 14;
ax.LineWidth = 1.0;
ax.TickDir = 'out';
ax.TickLength = [0.012 0.012];
ax.XColor = [0.15 0.15 0.15];
ax.YColor = [0.15 0.15 0.15];
ax.XLim = [0.8, numel(splitNames)+0.2];
ax.XTick = x;
ax.XTickLabel = xTickLabels;

ymin = min(mu_all(:) - sd_all(:), [], 'omitnan');
ymax = max(mu_all(:) + sd_all(:), [], 'omitnan');
if ~isfinite(ymin), ymin = 0; end
if ~isfinite(ymax), ymax = 0.2; end

ymin = max(0, floor((ymin - 0.01)*50)/50);
ymax = ceil((ymax + 0.01)*50)/50;
ax.YLim = [ymin, ymax];

grid on;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.GridColor = [0.86 0.86 0.86];
ax.GridAlpha = 1.0;
ax.Layer = 'top';

ylabel(ylab, 'FontName','Times New Roman', 'FontSize', 17);
xlabel('Split protocol', 'FontName','Times New Roman', 'FontSize', 17);
title(ttl, 'FontName','Times New Roman', 'FontSize', 16, 'FontWeight', 'normal');

legendHandles = [];
legendLabels  = {};
legendKeyOrder = {'FEATURE','E2E','OURSFUSION','OURSNOB','FIXED'};
for ii = 1:numel(legendKeyOrder)
    k = legendKeyOrder{ii};
    if isKey(hMap, k)
        legendHandles(end+1) = hMap(k); %#ok<AGROW>
        legendLabels{end+1} = method_key_to_label_local(k); %#ok<AGROW>
    end
end

lgd = legend(legendHandles, legendLabels, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'NumColumns', 3);
lgd.Box = 'off';
lgd.FontName = 'Times New Roman';
lgd.FontSize = 13;

hold off;

if cfg.save_analysis_fig
    save_fig_paper(fig, cfg.analysis_fig_dir, ['Fig_Main_' fileTag '_Across_Splits']);
end
end

function plot_complexity_performance_tradeoff(Results, cfg, METHODS)
if isempty(Results), return; end

xv = nan(numel(METHODS),1);
yv = nan(numel(METHODS),1);
cv = nan(numel(METHODS),1);

for m = 1:numel(METHODS)
    xv(m) = mean(collect_field(Results, [METHODS(m).key '_infer_us_perSample']), 'omitnan');
    yv(m) = mean(collect_field(Results, [METHODS(m).key '_acc']), 'omitnan');
    cv(m) = mean(collect_field(Results, [METHODS(m).key '_goodput_norm']), 'omitnan');
end

figure('Color','w','Position',[100 100 700 500]); hold on;
for m = 1:numel(METHODS)
    sty = method_style(METHODS(m).key);
    h = scatter(xv(m), yv(m), 220, cv(m), 'filled', 'LineWidth', 1.2);
    h.Marker = sty.marker;
    text(xv(m), yv(m), ['  ' METHODS(m).label], ...
        'FontSize', 10, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end

cb = colorbar;
cb.Label.String = 'Normalized goodput';
xlabel('Inference time per sample (us)');
ylabel('Accuracy');
title('Complexity-performance tradeoff');
beautify_ax(gca);

if cfg.save_analysis_fig
    save_fig_paper(gcf, cfg.analysis_fig_dir, 'Fig_Complexity_Performance_Tradeoff');
end
end

function plot_fewshot_triptych(Results, cfg, METHODS)
if isempty(Results), return; end

targetSplits = string(cfg.fewshot_plot_split_list);

for ss = 1:numel(targetSplits)
    splitNow = targetSplits(ss);

    idxS = strcmp(string({Results.split}), splitNow);
    Rsub = Results(idxS);
    if isempty(Rsub), continue; end

    fracVals = unique([Rsub.frac]);
    fracVals = sort(fracVals(:));

    nF = numel(fracVals);
    nM = numel(METHODS);

    AccMean = nan(nF, nM);
    AccStd  = nan(nF, nM);
    GPMean  = nan(nF, nM);
    GPStd   = nan(nF, nM);

    for f = 1:nF
        idxF = arrayfun(@(r) abs(r.frac - fracVals(f)) < 1e-12, Rsub);
        Rf = Rsub(idxF);

        for m = 1:nM
            key = METHODS(m).key;
            acc = collect_field(Rf, [key '_acc']);
            gp  = collect_field(Rf, [key '_goodput_norm']);

            AccMean(f,m) = mean(acc, 'omitnan');
            AccStd(f,m)  = std(acc, 'omitnan');
            GPMean(f,m)  = mean(gp, 'omitnan');
            GPStd(f,m)   = std(gp, 'omitnan');
        end
    end

    figH = figure('Color','w','Position',[80 80 1500 430]);

    subplot(1,3,1); hold on;
    for m = 1:nM
        sty = method_style(METHODS(m).key);
        errorbar(fracVals, AccMean(:,m), AccStd(:,m), ...
            'LineWidth', 1.8, 'Color', sty.color, ...
            'Marker', sty.marker, 'MarkerSize', 7, ...
            'MarkerFaceColor', sty.color, 'CapSize', 8);
    end
    if cfg.fewshot_use_logx, set(gca, 'XScale', 'log'); end
    xlabel('Train subset fraction');
    ylabel('Accuracy');
    title(sprintf('Accuracy mean +- std | split=%s', char(splitNow)));
    legend({METHODS.label}, 'Location', 'best');
    beautify_ax(gca);

    subplot(1,3,2); hold on;
    for m = 1:nM
        sty = method_style(METHODS(m).key);
        errorbar(fracVals, GPMean(:,m), GPStd(:,m), ...
            'LineWidth', 1.8, 'Color', sty.color, ...
            'Marker', sty.marker, 'MarkerSize', 7, ...
            'MarkerFaceColor', sty.color, 'CapSize', 8);
    end
    if cfg.fewshot_use_logx, set(gca, 'XScale', 'log'); end
    xlabel('Train subset fraction');
    ylabel('Normalized goodput');
    title(sprintf('Goodput mean +- std | split=%s', char(splitNow)));
    beautify_ax(gca);

    subplot(1,3,3); hold on;
    for m = 1:nM
        sty = method_style(METHODS(m).key);
        plot(fracVals, AccStd(:,m), ...
            'LineWidth', 1.8, 'Color', sty.color, ...
            'Marker', sty.marker, 'MarkerSize', 7, ...
            'MarkerFaceColor', sty.color);
    end
    if cfg.fewshot_use_logx, set(gca, 'XScale', 'log'); end
    xlabel('Train subset fraction');
    ylabel('Seed std of accuracy');
    title(sprintf('Accuracy stability (seed std) | split=%s', char(splitNow)));
    beautify_ax(gca);

    sgtitle(sprintf('Few-shot comparison across five methods | split=%s', char(splitNow)));

    if cfg.save_analysis_fig
        fn = sprintf('FewShot_Triptych_%s', char(splitNow));
        fn = regexprep(fn, '[^\w\d]+', '_');
        exportgraphics(figH, fullfile(cfg.analysis_fig_dir, [fn '.png']), 'Resolution', 300);
    end
end
end

%% ============================================================
%  COMMON HELPERS
% ============================================================
function safe_writetable(T, excelFile, sheetName)
if isempty(T) || ~istable(T), return; end
sheetName = make_valid_sheet_name(sheetName);
writetable(T, excelFile, 'Sheet', sheetName);
end

function s = make_valid_sheet_name(s)
s = char(string(s));
s = regexprep(s, '[:\\/?*\[\]]', '_');
if strlength(string(s)) > 31
    s = char(extractBetween(string(s), 1, 31));
end
end

function x = collect_field(Rsub, fieldName)
n = numel(Rsub);
x = nan(n,1);
for i = 1:n
    if isfield(Rsub(i), fieldName)
        v = Rsub(i).(fieldName);
        if isnumeric(v) && isscalar(v), x(i) = v; end
    end
end
end

function sty = method_style(key)
switch upper(string(key))
    case "FIXED"
        sty.color  = [0.45 0.45 0.45];
        sty.marker = 'o';
    case "FEATURE"
        sty.color  = [0.20 0.60 0.20];
        sty.marker = 'p';
    case "E2E"
        sty.color  = [0.85 0.33 0.10];
        sty.marker = 'd';
    case "OURSNOB"
        sty.color  = [0.25 0.35 0.85];
        sty.marker = 's';
    case "OURSFUSION"
        sty.color  = [0.64 0.08 0.18];
        sty.marker = '^';
    otherwise
        sty.color  = [0.20 0.20 0.20];
        sty.marker = 'o';
end
end

function beautify_ax(ax)
if nargin < 1 || isempty(ax), ax = gca; end
set(ax, 'FontName', 'Times New Roman', 'FontSize', 11, ...
    'LineWidth', 1.0, 'Box', 'on', 'TickDir', 'out');
grid(ax, 'on');
ax.GridAlpha = 0.15;
ax.MinorGridAlpha = 0.08;
end

function save_fig_paper(figHandle, outDir, baseName)
if nargin < 1 || isempty(figHandle), figHandle = gcf; end
if nargin < 2 || isempty(outDir), return; end
if nargin < 3 || isempty(baseName), baseName = 'figure'; end

if ~exist(outDir, 'dir'), mkdir(outDir); end
baseName = regexprep(baseName, '[^\w\d]+', '_');
saveas(figHandle, fullfile(outDir, [baseName '.png']));
end

function [xb, yb] = make_binned_mean_curve(x, y, nbins)
x = double(x(:));
y = double(y(:));
ok = isfinite(x) & isfinite(y);
x = x(ok); y = y(ok);

if isempty(x)
    xb = nan; yb = nan; return;
end

edges = linspace(min(x), max(x), nbins+1);
xb = nan(nbins,1); yb = nan(nbins,1);

for b = 1:nbins
    if b < nbins
        in = (x >= edges(b)) & (x < edges(b+1));
    else
        in = (x >= edges(b)) & (x <= edges(b+1));
    end
    if any(in)
        xb(b) = mean(x(in));
        yb(b) = mean(y(in));
    end
end

ok2 = isfinite(xb) & isfinite(yb);
xb = xb(ok2);
yb = yb(ok2);
end

function hit = find_run(Results, splitNow, targetFrac, targetSeed)
hit = [];
for i = 1:numel(Results)
    ok1 = isfield(Results(i),'split') && strcmp(string(Results(i).split), string(splitNow));
    ok2 = isfield(Results(i),'frac')  && abs(Results(i).frac - targetFrac) < 1e-12;
    ok3 = isfield(Results(i),'seed')  && Results(i).seed == targetSeed;
    if ok1 && ok2 && ok3
        hit = i;
        return;
    end
end
end



%% ============================================================
%  FIXED-MI-EESM
% ============================================================
function [best, Hist, Time] = train_fixed_mieesm_grid(MI_tr, y_tr, MI_va, y_va, M, cfg)
beta_list = cfg.fixed_beta_list;
w_list    = cfg.fixed_w_list;

bestScore = -inf;
bestMAE = inf;

nComb = numel(beta_list) * numel(w_list);
beta_v = nan(nComb,1);
w_v    = nan(nComb,1);
acc_v  = nan(nComb,1);
mae_v  = nan(nComb,1);

rr = 0;
for ib = 1:numel(beta_list)
    for iw = 1:numel(w_list)
        beta = beta_list(ib);
        w    = w_list(iw);

        z_tr = fixed_mieesm_score(MI_tr, beta, w, cfg);
        theta = fit_thresholds_from_train_z_anchor(z_tr, y_tr, M);

        tmp.beta = beta;
        tmp.w    = w;
        tmp.theta = theta;

        [acc_va, mae_va, ~] = eval_fixed_basic(tmp, MI_va, y_va, M, cfg);

        rr = rr + 1;
        beta_v(rr) = beta;
        w_v(rr)    = w;
        acc_v(rr)  = acc_va;
        mae_v(rr)  = mae_va;

        if strcmpi(cfg.fixed_select_by, 'val_mae')
            if mae_va < bestMAE
                bestMAE = mae_va;
                best = tmp;
            end
        else
            if acc_va > bestScore
                bestScore = acc_va;
                best = tmp;
            end
        end
    end
end

Hist = table(beta_v, w_v, acc_v, mae_v, ...
    'VariableNames', {'beta','w','valAcc','valMAE'});

Time.epoch_s_mean = 0;
Time.epoch_s_std  = 0;
end

function z = fixed_mieesm_score(MI0, beta, w, cfg)
beta = single(beta);
w    = single(w);
z = mi_eesm(MI0, beta, cfg.eps2);
z = w .* z;

if isa(z,'dlarray'), z = extractdata(z); end
if isa(z,'gpuArray'), z = gather(z); end
z = double(z(:));
end

function theta = fit_thresholds_from_train_z_anchor(z, y, M)
z = double(z(:));
y = double(y(:));

cls_med = nan(M,1);
for m = 0:M-1
    idx = (y == m);
    if any(idx)
        cls_med(m+1) = median(z(idx));
    end
end
for m = 1:M
    if isnan(cls_med(m))
        cls_med(m) = median(z);
    end
end

theta_raw = zeros(M-1,1);
for m = 1:M-1
    theta_raw(m) = 0.5 * (cls_med(m) + cls_med(m+1));
end
theta_raw = cummax(theta_raw);

if isempty(theta_raw)
    theta = zeros(0,1);
else
    theta = theta_raw - theta_raw(1);
end
end

function [acc, mae, acc_pm1, Met, Det] = eval_fixed(best, MI0, y, M, cfg)
[z, P, yhat] = eval_fixed_details(best, MI0, M, cfg);
acc = mean(yhat == y(:));
mae = mean(abs(yhat - y(:)));
acc_pm1 = mean(abs(yhat - y(:)) <= 1);
Met = risk_unc_metrics(P, y(:), yhat, cfg);
Det.P = P; Det.ytrue = y(:); Det.yhat = yhat; Det.z = z;
end

function [acc, mae, acc_pm1] = eval_fixed_basic(best, MI0, y, M, cfg)
[~, ~, yhat] = eval_fixed_details(best, MI0, M, cfg);
acc = mean(yhat == y(:));
mae = mean(abs(yhat - y(:)));
acc_pm1 = mean(abs(yhat - y(:)) <= 1);
end

function [z, P, yhat] = eval_fixed_details(best, MI0, M, cfg)
z = fixed_mieesm_score(MI0, best.beta, best.w, cfg);
theta = double(best.theta(:));

N = numel(z);
P = zeros(N,M);
for n = 1:N
    P(n,:) = ordinal_probs_fixed(z(n), theta);
end

[~, ix] = max(P, [], 2);
yhat = ix - 1;
end

function P = ordinal_probs(z, theta, M, eps2)
B = size(z,2);

if M == 1
    if isa(z,'dlarray')
        P = dlarray(ones(B,1,'single'));
    else
        P = ones(B,1,'single');
    end
    return;
end

zcol = reshape(z,[B,1]);
thetaRow = reshape(theta,[1,M-1]);
F = sigmoid_dl(thetaRow - zcol);

if isa(F,'dlarray')
    P = dlarray(zeros(B,M,'single'));
else
    P = zeros(B,M,'like',F);
end

P(:,1) = max(F(:,1), eps2);
for m = 2:M-1
    P(:,m) = max(F(:,m)-F(:,m-1), eps2);
end
P(:,M) = max(1-F(:,M-1), eps2);
P = P ./ sum(P,2);
end

%% ============================================================
%  FEATURE-MLP
% ============================================================
function [best, Hist, Time] = train_feature_mlp(net, X_tr, y_tr, X_va, y_va, M, cfg)
numEpochs = cfg.numEpochs; bs = cfg.batchSize; lr = cfg.lr;
trN = numel(y_tr); iters = ceil(trN/bs);

avgG=[]; avgSqG=[];
bestScore=-inf; noImp=0; best.net = net;
Hist.epoch = (1:numEpochs)';
Hist.trainLoss = nan(numEpochs,1);
Hist.valAcc = nan(numEpochs,1);
Hist.valMAE = nan(numEpochs,1);
epochTimes = nan(numEpochs,1);

for e=1:numEpochs
    te=tic;
    p = randperm(trN);
    epochLoss = 0;

    for it=1:iters
        i1=(it-1)*bs+1; i2=min(it*bs,trN);
        idx = p(i1:i2);

        Xb = X_tr(idx, :);
        yb = int32(y_tr(idx));
        dlX = dlarray(single(Xb)', 'CB');

        [loss, grads] = dlfeval(@grads_feature_mlp, net, dlX, yb, M, cfg);
        epochLoss = epochLoss + double(gather(extractdata(loss)));
        [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, (e-1)*iters+it, lr);
    end

    epochLoss = epochLoss/iters;
    Hist.trainLoss(e)=epochLoss;

    curr.net = net;
    [acc_va, mae_va, ~] = eval_feature_mlp_basic(curr, X_va, y_va);
    Hist.valAcc(e)=acc_va;
    Hist.valMAE(e)=mae_va;
    epochTimes(e)=toc(te);

    if acc_va > bestScore + cfg.minDeltaAcc
        bestScore = acc_va; noImp = 0; best.net = net;
    else
        noImp = noImp + 1;
    end

    if cfg.useEarlyStop && noImp>=cfg.patience
        Hist = trim_hist(Hist, e);
        epochTimes = epochTimes(1:e);
        break;
    end
end

Hist = trim_hist(Hist, find(~isnan(Hist.trainLoss),1,'last'));
epochTimes = epochTimes(1:numel(Hist.epoch));
Time.epoch_s_mean = mean(epochTimes,'omitnan');
Time.epoch_s_std  = std(epochTimes,'omitnan');
end

function [loss, grads] = grads_feature_mlp(net, dlX, y, M, cfg)
logits = forward(net, dlX);
B = size(logits,2);

amax = max(logits,[],1);
E = exp(logits-amax);
P = E ./ sum(E,1);

idx = sub2ind([M,B], double(y(:)'+1), 1:B);
py = P(idx);
loss_ce = -mean(log(py + 1e-12));

mvec = dlarray(single((0:M-1)'));
yexp = sum(P .* mvec, 1);
ytrue = single(reshape(y, 1, []));
loss_ord = mean(abs(yexp - ytrue));

if isfield(cfg,'useFeatureOrdinalMAE') && cfg.useFeatureOrdinalMAE
    loss = loss_ce + single(cfg.ordMAEWeight) * loss_ord;
else
    loss = loss_ce;
end

grads = dlgradient(loss, net.Learnables);
end

function [acc, mae, acc_pm1, Met, Det] = eval_feature_mlp(best, X_s, y, M, cfg)
[~, P, yhat] = eval_feature_mlp_details(best, X_s);
acc = mean(yhat==y(:));
mae = mean(abs(yhat-y(:)));
acc_pm1 = mean(abs(yhat-y(:))<=1);
Met = risk_unc_metrics(P, y(:), yhat, cfg);
Det.P = P; Det.ytrue = y(:); Det.yhat = yhat;
end

function [acc, mae, acc_pm1] = eval_feature_mlp_basic(best, X_s, y)
[~, ~, yhat] = eval_feature_mlp_details(best, X_s);
acc = mean(yhat==y(:));
mae = mean(abs(yhat-y(:)));
acc_pm1 = mean(abs(yhat-y(:))<=1);
end

function [logits, P, yhat] = eval_feature_mlp_details(best, X_s)
dlX = dlarray(single(X_s)', 'CB');
logits = extractdata(forward(best.net, dlX));
logits = logits';

amax = max(logits,[],2);
E = exp(logits-amax);
P = E ./ sum(E,2);

[~,ix]=max(P,[],2);
yhat = ix-1;
end

%% ============================================================
%  E2E
% ============================================================
function [best, Hist, Time] = train_e2e_MIonly(net, MI_trs, y_tr, MI_vas, y_va, M, cfg)
numEpochs = cfg.numEpochs; bs = cfg.batchSize; lr = cfg.lr;
trN = numel(y_tr); iters = ceil(trN/bs);

avgG=[]; avgSqG=[];
bestScore=-inf; noImp=0; best.net = net;
Hist.epoch = (1:numEpochs)';
Hist.trainLoss = nan(numEpochs,1);
Hist.valAcc = nan(numEpochs,1);
Hist.valMAE = nan(numEpochs,1);
epochTimes = nan(numEpochs,1);

for e=1:numEpochs
    te=tic;
    p = randperm(trN);
    epochLoss = 0;

    for it=1:iters
        i1=(it-1)*bs+1; i2=min(it*bs,trN);
        idx = p(i1:i2);

        MIb = MI_trs(:, idx);
        yb  = int32(y_tr(idx));
        dlX = dlarray(single(MIb), 'CB');

        [loss, grads] = dlfeval(@grads_e2e_MIonly, net, dlX, yb, M);
        epochLoss = epochLoss + double(gather(extractdata(loss)));
        [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, (e-1)*iters+it, lr);
    end

    epochLoss = epochLoss/iters;
    Hist.trainLoss(e)=epochLoss;

    curr.net = net;
    [acc_va, mae_va, ~] = eval_e2e_MIonly_basic(curr, MI_vas, y_va);
    Hist.valAcc(e)=acc_va;
    Hist.valMAE(e)=mae_va;
    epochTimes(e)=toc(te);

    if acc_va > bestScore + cfg.minDeltaAcc
        bestScore = acc_va; noImp = 0; best.net = net;
    else
        noImp = noImp + 1;
    end

    if cfg.useEarlyStop && noImp>=cfg.patience
        Hist = trim_hist(Hist, e);
        epochTimes = epochTimes(1:e);
        break;
    end
end

Hist = trim_hist(Hist, find(~isnan(Hist.trainLoss),1,'last'));
epochTimes = epochTimes(1:numel(Hist.epoch));
Time.epoch_s_mean = mean(epochTimes,'omitnan');
Time.epoch_s_std  = std(epochTimes,'omitnan');
end

function [loss, grads] = grads_e2e_MIonly(net, dlX, y, M)
logits = forward(net, dlX);
B = size(logits,2);

amax = max(logits,[],1);
E = exp(logits-amax);
P = E ./ sum(E,1);

idx = sub2ind([M,B], double(y(:)'+1), 1:B);
py = P(idx);
loss = -mean(log(py + 1e-12));

grads = dlgradient(loss, net.Learnables);
end

function [acc, mae, acc_pm1, Met, Det] = eval_e2e_MIonly(best, MI_s, y, ~, cfg)
[~, P, yhat] = eval_e2e_MIonly_details(best, MI_s);
acc = mean(yhat==y(:));
mae = mean(abs(yhat-y(:)));
acc_pm1 = mean(abs(yhat-y(:))<=1);
Met = risk_unc_metrics(P, y(:), yhat, cfg);
Det.P = P; Det.ytrue = y(:); Det.yhat = yhat;
end

function [acc, mae, acc_pm1] = eval_e2e_MIonly_basic(best, MI_s, y)
[~, ~, yhat] = eval_e2e_MIonly_details(best, MI_s);
acc = mean(yhat==y(:));
mae = mean(abs(yhat-y(:)));
acc_pm1 = mean(abs(yhat-y(:))<=1);
end

function [logits, P, yhat] = eval_e2e_MIonly_details(best, MI_s)
dlX = dlarray(single(MI_s), 'CB');
logits = extractdata(forward(best.net, dlX));
logits = logits';

amax = max(logits,[],2);
E = exp(logits-amax);
P = E ./ sum(E,2);

[~,ix]=max(P,[],2);
yhat = ix-1;
end

%% ============================================================
%  OURS: Physics-guided fusion
%  data branch: MI -> h_data
%  physics branch: x_deg -> (beta,w,b) -> z_phys/prior
%  fusion head: h_data + gated h_phy -> logits
% ============================================================
function fuseParams = init_ours_fusion_params(cfg, M)
fd = cfg.ours_data_hidden2;
pd = 4; % [z_phys; beta; w; b]
init = cfg.ours_init_scale;

fuseParams.Wp = dlarray(single(init * randn(fd, pd)));
fuseParams.bp = dlarray(single(zeros(fd,1)));

if cfg.ours_use_gate
    fuseParams.Wg = dlarray(single(init * randn(fd, 2*fd)));
    fuseParams.bg = dlarray(single(zeros(fd,1)));
else
    fuseParams.Wg = dlarray(single([]));
    fuseParams.bg = dlarray(single([]));
end

fuseParams.Wc = dlarray(single(init * randn(M, fd)));
fuseParams.bc = dlarray(single(zeros(M,1)));

fuseParams.Wr = dlarray(single(init * randn(1, fd)));
fuseParams.br = dlarray(single(0));
end

function [best, Hist, Time] = train_ours_pg_fusion( ...
    netData, netPhys, deltas_raw, fuseParams, ...
    MI_trs, MI_tr_raw, Xdeg_tr, y_tr, ...
    MI_vas, MI_va_raw, Xdeg_va, y_va, M, cfg)

numEpochs = cfg.numEpochs;
bs = cfg.batchSize;
lr = cfg.lr;

trN = numel(y_tr);
iters = ceil(trN / bs);

avgGD = []; avgSqGD = [];
avgGP = []; avgSqGP = [];
avgGA = []; avgSqGA = [];

bestScore = -inf;
noImp = 0;

best.netData = netData;
best.netPhys = netPhys;
best.deltas  = deltas_raw;
best.fuseParams = fuseParams;

Hist.epoch = (1:numEpochs)';
Hist.trainLoss = nan(numEpochs,1);
Hist.valAcc = nan(numEpochs,1);
Hist.valMAE = nan(numEpochs,1);
epochTimes = nan(numEpochs,1);

for e = 1:numEpochs
    te = tic;
    p = randperm(trN);
    epochLoss = 0;

    for it = 1:iters
        i1 = (it-1)*bs + 1;
        i2 = min(it*bs, trN);
        idx = p(i1:i2);

        MIb_z = MI_trs(:, idx);
        MIb_r = MI_tr_raw(:, idx);
        Xdb   = Xdeg_tr(idx,:);
        yb    = int32(y_tr(idx(:)));

        dlMIz  = dlarray(single(MIb_z), 'CB');
        dlMIr  = dlarray(single(MIb_r), 'CB');
        dlXdeg = dlarray(single(Xdb)', 'CB');

        [loss, gData, gPhys, gAux] = dlfeval(@grads_ours_pg_fusion, ...
            netData, netPhys, deltas_raw, fuseParams, ...
            dlMIz, dlMIr, dlXdeg, yb, M, cfg);

        epochLoss = epochLoss + double(gather(extractdata(loss)));

        [netData, avgGD, avgSqGD] = adamupdate(netData, gData, avgGD, avgSqGD, (e-1)*iters+it, lr);
        [netPhys, avgGP, avgSqGP] = adamupdate(netPhys, gPhys, avgGP, avgSqGP, (e-1)*iters+it, lr);

        auxParams = {deltas_raw, ...
                     fuseParams.Wp, fuseParams.bp, ...
                     fuseParams.Wg, fuseParams.bg, ...
                     fuseParams.Wc, fuseParams.bc, ...
                     fuseParams.Wr, fuseParams.br};

        [auxParams, avgGA, avgSqGA] = adamupdate_cell(auxParams, gAux, avgGA, avgSqGA, (e-1)*iters+it, lr);

        deltas_raw     = auxParams{1};
        fuseParams.Wp  = auxParams{2};
        fuseParams.bp  = auxParams{3};
        fuseParams.Wg  = auxParams{4};
        fuseParams.bg  = auxParams{5};
        fuseParams.Wc  = auxParams{6};
        fuseParams.bc  = auxParams{7};
        fuseParams.Wr  = auxParams{8};
        fuseParams.br  = auxParams{9};
    end

    epochLoss = epochLoss / iters;
    Hist.trainLoss(e) = epochLoss;

    curr.netData = netData;
    curr.netPhys = netPhys;
    curr.deltas = deltas_raw;
    curr.fuseParams = fuseParams;

    [acc_va, mae_va, ~] = eval_ours_pg_fusion_basic(curr, MI_vas, MI_va_raw, Xdeg_va, y_va, M, cfg);
    Hist.valAcc(e) = acc_va;
    Hist.valMAE(e) = mae_va;
    epochTimes(e) = toc(te);

    if acc_va > bestScore + cfg.minDeltaAcc
        bestScore = acc_va;
        noImp = 0;
        best.netData = netData;
        best.netPhys = netPhys;
        best.deltas = deltas_raw;
        best.fuseParams = fuseParams;
    else
        noImp = noImp + 1;
    end

    if cfg.useEarlyStop && noImp >= cfg.patience
        Hist = trim_hist(Hist, e);
        epochTimes = epochTimes(1:e);
        break;
    end
end

Hist = trim_hist(Hist, find(~isnan(Hist.trainLoss),1,'last'));
epochTimes = epochTimes(1:numel(Hist.epoch));
Time.epoch_s_mean = mean(epochTimes,'omitnan');
Time.epoch_s_std  = std(epochTimes,'omitnan');
end

function [loss, gradsData, gradsPhys, gradsAux] = grads_ours_pg_fusion( ...
    netData, netPhys, deltas_raw, fuseParams, ...
    dlMIz, dlMIr, dlXdeg, y, M, cfg)

eps1 = dlarray(cfg.eps1);
eps2 = dlarray(cfg.eps2);

% ----- data branch -----
h_data = forward(netData, dlMIz);  % [fd, B], 'CB'
if isfield(cfg,'ours_use_data_branch') && ~cfg.ours_use_data_branch
    h_data = 0 .* h_data;
end

% ----- physics branch -----
o = forward(netPhys, dlXdeg);
beta = bounded_sigmoid_dl(o(1,:), cfg.beta_min, cfg.beta_max);
w    = bounded_sigmoid_dl(o(2,:), cfg.w_min,    cfg.w_max);
b    = bounded_sigmoid_dl(o(3,:), cfg.b_min,    cfg.b_max);

z_phys = w .* mi_eesm(dlMIr, beta, eps2);
if isfield(cfg,'use_b_in_fusion') && cfg.use_b_in_fusion
    z_phys = z_phys + b;
else
    b = 0 .* b;
end

theta   = build_theta_anchor(deltas_raw, M, eps1);
P_prior = ordinal_probs(z_phys, theta, M, eps2);              % [B,M]
prior_logits = dlarray(stripdims(log(P_prior + eps2))', 'CB'); % [M,B]

p_feat = [reshape(z_phys,1,[]); reshape(beta,1,[]); reshape(w,1,[]); reshape(b,1,[])];
h_phy = relu_dl(fc_dl(fuseParams.Wp, p_feat, fuseParams.bp));

if isfield(cfg,'ours_use_phys_feature') && ~cfg.ours_use_phys_feature
    h_phy = 0 .* h_data;
end

useGateNow = cfg.ours_use_gate && ...
    (~isfield(cfg,'ours_use_data_branch') || cfg.ours_use_data_branch) && ...
    (~isfield(cfg,'ours_use_phys_feature') || cfg.ours_use_phys_feature);

if useGateNow
    gh = concat_channel(h_data, h_phy);
    g = sigmoid_dl(fc_dl(fuseParams.Wg, gh, fuseParams.bg));
    h_fuse = h_data + g .* h_phy;
else
    h_fuse = h_data + h_phy;
end

logits = fc_dl(fuseParams.Wc, h_fuse, fuseParams.bc);

if isfield(cfg,'ours_use_prior_logit_fusion') && cfg.ours_use_prior_logit_fusion
    logits = logits + single(cfg.ours_prior_logit_weight) * prior_logits;
end

P = softmax_channel(logits, eps2);   % [M,B]

B = size(P,2);
yy = double(y(:)') + 1;
idx = sub2ind([M,B], yy, 1:B);
py = P(idx);

% ----- weighted CE -----
if cfg.ours_use_weighted_ce
    zc = z_phys - mean(z_phys, 'all');
    risk_w = sigmoid_dl(-zc);
    wt = 1 + single(cfg.ours_weighted_ce_lambda) * risk_w;
    loss_cls = -mean(wt .* log(py + eps2));
else
    loss_cls = -mean(log(py + eps2));
end

% ----- ordinal MAE -----
mvec = dlarray(single((0:M-1)'));
yexp = sum(P .* mvec, 1);
loss_ord = mean(abs(yexp - single(reshape(y,1,[]))));

loss = loss_cls;
if cfg.useAuxOrdinalMAE
    loss = loss + single(cfg.auxMAEWeight) * loss_ord;
end

% ----- physics alignment -----
if cfg.ours_use_phy_align
    % P: [M,B] -> [B,M]
    P_post = stripdims(P)';
    P_post = reshape(P_post, [], M);

    % P_prior should also be [B,M], force it explicitly
    P_pri  = stop_gradient(P_prior);
    P_pri  = stripdims(P_pri);
    P_pri  = reshape(P_pri, [], M);

    loss_phy = mean((P_post - P_pri).^2, 'all');
    loss = loss + single(cfg.ours_phyAlignWeight) * loss_phy;
end

% ----- risk auxiliary -----
if cfg.ours_use_risk_aux
    risk_logit = fc_dl(fuseParams.Wr, h_phy, fuseParams.br);   % [1,B]
    [~, yhat_tmp] = max(extractdata(P), [], 1);
    yhat_tmp = single(yhat_tmp - 1);
    risk_tgt = single(abs(yhat_tmp - single(y(:)')) > 0);
    risk_prob = sigmoid_dl(risk_logit);
    loss_risk = -mean(risk_tgt .* log(risk_prob + eps2) + ...
                     (1-risk_tgt) .* log(1-risk_prob + eps2));
    loss = loss + single(cfg.ours_riskWeight) * loss_risk;
end

% ----- regularization -----
if cfg.useBetaWReg
    loss = loss ...
        + single(cfg.betaRegWeight) * mean((beta(:) - single(cfg.betaCenter)).^2) ...
        + single(cfg.wRegWeight)    * mean((w(:)    - single(cfg.wCenter)).^2) ...
        + single(cfg.bRegWeight)    * mean((b(:)    - single(cfg.bCenter)).^2);
end

varsData = netData.Learnables.Value;
varsPhys = netPhys.Learnables.Value;
varsAux  = {deltas_raw, ...
            fuseParams.Wp, fuseParams.bp, ...
            fuseParams.Wg, fuseParams.bg, ...
            fuseParams.Wc, fuseParams.bc, ...
            fuseParams.Wr, fuseParams.br};

varsAll = [varsData; varsPhys; varsAux(:)];
gAll = dlgradient(loss, varsAll);

nD = numel(varsData);
nP = numel(varsPhys);

gradsData = netData.Learnables;
gradsData.Value = gAll(1:nD);

gradsPhys = netPhys.Learnables;
gradsPhys.Value = gAll(nD+1:nD+nP);

gradsAux = gAll(nD+nP+1:end);
end

function [acc, mae, acc_pm1, Met, Det] = eval_ours_pg_fusion(best, MIz, MIr, Xdeg, y, M, cfg)
[z, P, yhat] = eval_ours_pg_fusion_details(best, MIz, MIr, Xdeg, M, cfg);
acc = mean(yhat == y(:));
mae = mean(abs(yhat - y(:)));
acc_pm1 = mean(abs(yhat - y(:)) <= 1);
Met = risk_unc_metrics(P, y(:), yhat, cfg);
Det.P = P;
Det.ytrue = y(:);
Det.yhat = yhat;
Det.z = z;
end

function [acc, mae, acc_pm1] = eval_ours_pg_fusion_basic(best, MIz, MIr, Xdeg, y, M, cfg)
[~, ~, yhat] = eval_ours_pg_fusion_details(best, MIz, MIr, Xdeg, M, cfg);
acc = mean(yhat == y(:));
mae = mean(abs(yhat - y(:)));
acc_pm1 = mean(abs(yhat - y(:)) <= 1);
end

function [z, P, yhat] = eval_ours_pg_fusion_details(best, MIz, MIr, Xdeg, M, cfg)
dlMIz  = dlarray(single(MIz), 'CB');
dlMIr  = dlarray(single(MIr), 'CB');
dlXdeg = dlarray(single(Xdeg)', 'CB');

[~, z_phys, P_post, ~, ~, ~] = forward_ours_pg(best, dlMIz, dlMIr, dlXdeg, M, cfg);

P = gather(extractdata(stripdims(P_post)))';   % [B,M]
[~, yhat] = max(P, [], 2);
yhat = yhat - 1;
z = gather(extractdata(stripdims(z_phys(:))));
end

function [z, P, yhat, beta_s, w_s, b_s] = eval_ours_pg_fusion_details_full(best, MIz, MIr, Xdeg, M, cfg)
dlMIz  = dlarray(single(MIz), 'CB');
dlMIr  = dlarray(single(MIr), 'CB');
dlXdeg = dlarray(single(Xdeg)', 'CB');

[~, z_phys, P_post, beta, w, b] = forward_ours_pg(best, dlMIz, dlMIr, dlXdeg, M, cfg);

P = gather(extractdata(stripdims(P_post)))';   % [B,M]
[~, yhat] = max(P, [], 2);
yhat = yhat - 1;

beta_s = gather(extractdata(stripdims(beta(:))));
w_s    = gather(extractdata(stripdims(w(:))));
b_s    = gather(extractdata(stripdims(b(:))));
z      = gather(extractdata(stripdims(z_phys(:))));
end

function [h_fuse, z_phys, P_post, beta, w, b] = forward_ours_pg(best, dlMIz, dlMIr, dlXdeg, M, cfg)
eps1 = cfg.eps1;
eps2 = cfg.eps2;


h_data = forward(best.netData, dlMIz);
if isfield(cfg,'ours_use_data_branch') && ~cfg.ours_use_data_branch
    h_data = 0 .* h_data;
end

o = forward(best.netPhys, dlXdeg);
beta = bounded_sigmoid_dl(o(1,:), cfg.beta_min, cfg.beta_max);
w    = bounded_sigmoid_dl(o(2,:), cfg.w_min,    cfg.w_max);
b    = bounded_sigmoid_dl(o(3,:), cfg.b_min,    cfg.b_max);

z_phys = w .* mi_eesm(dlMIr, beta, eps2);
if isfield(cfg,'use_b_in_fusion') && cfg.use_b_in_fusion
    z_phys = z_phys + b;
else
    b = 0 .* b;
end

theta   = build_theta_anchor(best.deltas, M, eps1);
P_prior = ordinal_probs(z_phys, theta, M, eps2);
prior_logits = dlarray(stripdims(log(P_prior + eps2))', 'CB');

p_feat = [reshape(z_phys,1,[]); reshape(beta,1,[]); reshape(w,1,[]); reshape(b,1,[])];
h_phy = relu_dl(fc_dl(best.fuseParams.Wp, p_feat, best.fuseParams.bp));

if isfield(cfg,'ours_use_phys_feature') && ~cfg.ours_use_phys_feature
    h_phy = 0 .* h_data;
end

useGateNow = cfg.ours_use_gate && ...
    (~isfield(cfg,'ours_use_data_branch') || cfg.ours_use_data_branch) && ...
    (~isfield(cfg,'ours_use_phys_feature') || cfg.ours_use_phys_feature);

if useGateNow
    gh = concat_channel(h_data, h_phy);
    g = sigmoid_dl(fc_dl(best.fuseParams.Wg, gh, best.fuseParams.bg));
    h_fuse = h_data + g .* h_phy;
else
    h_fuse = h_data + h_phy;
end

logits = fc_dl(best.fuseParams.Wc, h_fuse, best.fuseParams.bc);

if isfield(cfg,'ours_use_prior_logit_fusion') && cfg.ours_use_prior_logit_fusion
    logits = logits + single(cfg.ours_prior_logit_weight) * prior_logits;
end

P_post = softmax_channel(logits, eps2);
end

function y = relu_dl(x)
y = max(x, 0);
end

function P = softmax_channel(logits, eps2)
amax = max(logits, [], 1);
E = exp(logits - amax);
P = E ./ (sum(E,1) + eps2);
end

function y = stop_gradient(x)
tmp = extractdata(stripdims(x));
y = dlarray(single(tmp));
end

%% ============================================================
%  PRECOMPUTE + 25D x_deg
% ============================================================
function [MI, eta, re, lk, GapK, gamma1, sigma2_ratio, log_cond] = precompute_MI_deg_FAST(H, Pn_mW, cfg)
K = cfg.K;
N = size(H,4);

MI   = zeros(K, N, 'single');
eta  = zeros(K, N, 'single');
re   = zeros(K, N, 'single');
lk   = zeros(K, N, 'single');
GapK = zeros(K, N, 'single');

gamma1       = zeros(K, N, 'single');
sigma2_ratio = zeros(K, N, 'single');
log_cond     = zeros(K, N, 'single');

I2 = eye(cfg.Nr);

fprintf('Precompute (parallel) started...\n');
t0 = tic;

parfor j = 1:N
    rho0 = 1 / (double(Pn_mW(j)) + eps);

    MI_col   = zeros(K,1,'single');
    eta_col  = zeros(K,1,'single');
    re_col   = zeros(K,1,'single');
    lk_col   = zeros(K,1,'single');
    Gap_col  = zeros(K,1,'single');

    g1_col   = zeros(K,1,'single');
    sr_col   = zeros(K,1,'single');
    lc_col   = zeros(K,1,'single');

    for i = 1:K
        Hij = H(:,:,i,j);
        Q = Hij * Hij';

        lam = eig2x2_hermitian(Q);
        lam1 = max(lam(1), eps);
        lam2 = max(lam(2), eps);

        MI_col(i) = single(max(log1p(rho0*lam1) + log1p(rho0*lam2), 0));

        s1 = sqrt(lam1);
        s2 = sqrt(lam2);
        s  = [s1; s2];
        sN = s / (norm(s) + eps);

        lam1n = single(sN(1)^2);
        lam2n = single(sN(2)^2);

        eta_col(i) = lam1n;
        re_col(i)  = exp(-(lam1n*log(lam1n+eps) + lam2n*log(lam2n+eps)));
        lk_col(i)  = log(single(sN(1)/(sN(2)+eps)));

        Qtil = Q + double(1e-3) * I2;
        lamT = eig2x2_hermitian((Qtil + Qtil')/2);
        logdetQ = log(max(lamT(1), eps)) + log(max(lamT(2), eps));
        trQ = real(trace(Qtil));
        Gap_col(i) = single(cfg.Nr * log(trQ/cfg.Nr + eps) - logdetQ);

        g1_col(i) = single(lam1 * rho0);
        sr_col(i) = single(s2 / (s1 + eps));
        lc_col(i) = single(log(s1 / (s2 + eps)));
    end

    MI(:,j)   = MI_col;
    eta(:,j)  = eta_col;
    re(:,j)   = re_col;
    lk(:,j)   = lk_col;
    GapK(:,j) = Gap_col;

    gamma1(:,j)       = g1_col;
    sigma2_ratio(:,j) = sr_col;
    log_cond(:,j)     = lc_col;
end

fprintf('Precompute (parallel) finished. Total %.1fs\n', toc(t0));
end

function lam = eig2x2_hermitian(Q)
a = real(Q(1,1));
d = real(Q(2,2));
b = Q(1,2);

trQ  = a + d;
detQ = a*d - abs(b)^2;
disc = max(trQ^2 - 4*detQ, 0);
root_disc = sqrt(disc);

lam1 = 0.5 * (trQ + root_disc);
lam2 = 0.5 * (trQ - root_disc);

lam = [lam1; lam2];
lam(lam < 0) = 0;
lam = sort(lam, 'descend');
end

function [Xtr, Xva, Ref] = build_xdeg_rebuild(MI, gamma1, sigma2_ratio, log_cond, re, bf, n_dBm, tr, va, cfg)
if strcmpi(cfg.domain_shift_ref, 'gamma1')
    Rtr = double(gamma1(:,tr));
else
    Rtr = double(MI(:,tr));
end

ref_vec = Rtr(:);
Ref.median_train = median(ref_vec);
Ref.iqr_train    = iqr(ref_vec);
Ref.p10_train    = prctile(ref_vec, 10);

if Ref.iqr_train < 1e-8
    Ref.iqr_train = 1e-8;
end

tmp_mi_train = double(MI(:,tr));
Ref.deep_thr_mi = prctile(tmp_mi_train(:), 15);

Xtr = build_xdeg_side(MI, gamma1, sigma2_ratio, log_cond, re, bf, n_dBm, tr, Ref, cfg);
Xva = build_xdeg_side(MI, gamma1, sigma2_ratio, log_cond, re, bf, n_dBm, va, Ref, cfg);
end

function X = build_xdeg_side(MI, gamma1, sigma2_ratio, log_cond, re, bf, n_dBm, idx, Ref, cfg)
Nside = numel(idx);
X = zeros(Nside, 25, 'single');

K = size(MI,1);
seg_edges = round(linspace(1, K+1, 5));

for t = 1:Nside
    j = idx(t);

    vMI = double(MI(:,j));
    vg1 = double(gamma1(:,j));
    vsr = double(sigma2_ratio(:,j));
    vlc = double(log_cond(:,j));
    vre = double(re(:,j));

    if strcmpi(cfg.domain_shift_ref, 'gamma1')
        vref = vg1;
    else
        vref = vMI;
    end

    med_now = median(vref);
    iqr_now = iqr(vref);
    p10_now = prctile(vref, 10);

    dMI = diff(vMI);

    X(t,1)  = single(mean(vg1));
    X(t,2)  = single(var(vg1, 0));
    X(t,3)  = single(prctile(vg1, 10));
    X(t,4)  = single(median(vg1));

    X(t,5)  = single(mean(vsr));
    X(t,6)  = single(var(vsr, 0));
    X(t,7)  = single(mean(vlc));
    X(t,8)  = single(mean(vre));

    X(t,9)  = single(med_now - Ref.median_train);
    X(t,10) = single(iqr_now / (Ref.iqr_train + 1e-8));
    X(t,11) = single(p10_now - Ref.p10_train);

    X(t,12) = single(mean(vMI));
    X(t,13) = single(prctile(vMI, 10));
    X(t,14) = single(var(vMI, 0));
    X(t,15) = single(var(dMI, 0));

    X(t,16) = single(bf(j));
    X(t,17) = single(n_dBm(j));

    thr = double(Ref.deep_thr_mi(1));
    mask = (vMI(:) < thr);

    X(t,18) = single(mean(mask));
    X(t,19) = single(longest_consecutive_ones(mask));
    X(t,20) = single(mean(max(0, thr - vMI(:))));
    X(t,21) = single(mean(abs(dMI)));
    X(t,22) = single(max(abs(dMI)));

    vMI_sort = sort(vMI, 'ascend');
    nw10 = min(10, numel(vMI_sort));
    nw20 = min(20, numel(vMI_sort));

    X(t,23) = single(mean(vMI_sort(1:nw10)));
    X(t,24) = single(mean(vMI_sort(1:nw20)));

    seg_means = zeros(4,1);
    for s = 1:4
        i1 = seg_edges(s);
        i2 = seg_edges(s+1)-1;
        seg_means(s) = mean(vMI(i1:i2));
    end
    X(t,25) = single(max(seg_means) - min(seg_means));
end
end

function L = longest_consecutive_ones(mask)
mask = logical(mask(:));
if isempty(mask)
    L = 0;
    return;
end

d = diff([false; mask; false]);
run_starts = find(d == 1);
run_ends   = find(d == -1) - 1;

if isempty(run_starts)
    L = 0;
else
    L = max(run_ends - run_starts + 1);
end
end

function [x_trs, x_vas] = zscore_train_only(x_tr, x_va)
mu = mean(x_tr,1);
sd = std(x_tr,[],1) + single(1e-6);
x_trs = (x_tr - mu) ./ sd;
x_vas = (x_va - mu) ./ sd;
end

function [MI_trs, MI_vas] = zscore_MI_train_only(MI_tr, MI_va)
muMI = mean(MI_tr,2);
sdMI = std(MI_tr,0,2) + single(1e-6);
MI_trs = (MI_tr - muMI) ./ sdMI;
MI_vas = (MI_va - muMI) ./ sdMI;
end

%% ============================================================
%  SPLIT
% ============================================================
function [tr, va] = make_split(cfg, N, ~, n_dBm, GapK)
rng(cfg.rng_seed);

switch lower(cfg.split_mode)
    case 'random'
        idx = randperm(N);
        ntr = round(cfg.train_ratio * N);
        tr  = idx(1:ntr);
        va  = idx(ntr+1:end);

    case 'nf_ood'
        q = quantile(n_dBm, cfg.nf_train_quantile);
        tr = find(n_dBm <= q);
        va = find(n_dBm > q);

    case 'gap_ood'
        meanGap = mean(double(GapK),1)';
        q = quantile(meanGap, cfg.gap_train_quantile);
        tr = find(meanGap <= q);
        va = find(meanGap > q);

    case 'snr_gap_ood'
        meanGap = mean(double(GapK),1)';
        qNoise = quantile(n_dBm, cfg.snr_gap_q);
        qGap   = quantile(meanGap, cfg.gap_hard_q);

        va = find((n_dBm > qNoise) & (meanGap > qGap));
        tr = setdiff((1:N)', va);

        if numel(va) < cfg.snr_gap_minValN
            qNoise = quantile(n_dBm, 0.6);
            qGap   = quantile(meanGap, 0.6);
            va = find((n_dBm > qNoise) & (meanGap > qGap));
            tr = setdiff((1:N)', va);
        end

    otherwise
        error('Unknown split_mode: %s', cfg.split_mode);
end

assert(~isempty(tr) && ~isempty(va), 'split failed: one side empty.');
end

%% ============================================================
%  COMMON
% ============================================================
function Hist = trim_hist(Hist, emax)
if isempty(emax) || ~isfinite(emax), return; end
Hist.epoch = Hist.epoch(1:emax);
Hist.trainLoss = Hist.trainLoss(1:emax);
Hist.valAcc = Hist.valAcc(1:emax);
Hist.valMAE = Hist.valMAE(1:emax);
end

function z = mi_eesm(MI, beta, eps2)
K = size(MI,1);
beta_row = reshape(beta, 1, []);
Z = -MI .* beta_row;
lse = logsumexp_dim1(Z, eps2) - log(single(K));
z = -(1 ./ (beta_row + eps2)) .* lse;
end

function theta = build_theta_anchor(deltas_raw, M, eps1)
if M <= 1
    theta = dlarray(zeros(0,1,'single'));
    return;
elseif M == 2
    theta = dlarray(single(0));
    return;
end

theta = dlarray(zeros(M-1,1,'single'));
theta(1) = dlarray(single(0));

inc = softplus_dl(deltas_raw) + eps1;
for k = 2:(M-1)
    theta(k) = theta(k-1) + inc(k-1);
end
end

function y = softplus_dl(x)
y = max(x,0) + log(1 + exp(-abs(x)));
end

function y = sigmoid_dl(x)
y = 1 ./ (1 + exp(-x));
end

function y = bounded_sigmoid_dl(x, lo, hi)
y = single(lo) + single(hi - lo) .* sigmoid_dl(x);
end

function s = logsumexp_dim1(A, eps2)
amax = max(A,[],1);
s = amax + log(sum(exp(A-amax),1) + eps2);
end

function [params, avgGrad, avgSqGrad] = adamupdate_cell(params, grads, avgGrad, avgSqGrad, iter, lr)
beta1=0.9; beta2=0.999; eps0=1e-8;
if isempty(avgGrad)
    avgGrad=cell(size(params));
    avgSqGrad=cell(size(params));
    for i=1:numel(params)
        avgGrad{i}=dlarray(zeros(size(params{i}),'like',params{i}));
        avgSqGrad{i}=dlarray(zeros(size(params{i}),'like',params{i}));
    end
end
for i=1:numel(params)
    if isempty(params{i}), continue; end
    g = grads{i};
    avgGrad{i}   = beta1*avgGrad{i}   + (1-beta1)*g;
    avgSqGrad{i} = beta2*avgSqGrad{i} + (1-beta2)*(g.^2);
    mhat = avgGrad{i}   ./ (1 - beta1^iter);
    vhat = avgSqGrad{i} ./ (1 - beta2^iter);
    params{i} = params{i} - lr*mhat./(sqrt(vhat)+eps0);
end
end

function tmean = time_infer_repeated(fhandle, nRep)
t = zeros(nRep,1);
for i=1:nRep
    tt = tic;
    fhandle();
    t(i) = toc(tt);
end
tmean = mean(t);
end

function v = check_vec(v, K, row, name)
if ~(isnumeric(v) && numel(v)==K)
    error("Row %d: %s wrong size (%s, %d elems), expected %d.", row, name, class(v), numel(v), K);
end
end

function cellvec = parse_csi_column_to_cellvec(col, K, colname)
N = numel(col);
cellvec = cell(N,1);
t0=tic;
for j=1:N
    if mod(j, max(1,floor(N/10)))==0
        fprintf('  parsing %s: %6d/%6d (%.1f%%), elapsed %.1fs\n', colname, j, N, 100*j/N, toc(t0));
    end
    s = col{j};
    if isempty(s), error('Row %d: %s empty.', j, colname); end
    s = strtrim(string(s));
    if ~startsWith(s,'['), s="["+s; end
    if ~endsWith(s,']'),   s=s+"]"; end
    v = str2num(char(s)); %#ok<ST2NM>
    if isempty(v), error('Row %d: %s parse failed. Example="%s"', j, colname, s); end
    v = v(:).';
    if numel(v) ~= K
        error('Row %d: %s has %d elems (expected %d).', j, colname, numel(v), K);
    end
    cellvec{j}=v;
end
end

function m0 = low_threshold_m0(ytrue, M, cfg)
if isfield(cfg,'lowMode') && strcmpi(cfg.lowMode,'quantile')
    thr = quantile(double(ytrue), cfg.lowQuantile);
    m0 = floor(thr);
else
    m0 = cfg.lowMCS_m0;
end
m0 = max(0, min(M-1, m0));
end

%% ============================================================
%  METRICS
% ============================================================
function Met = risk_unc_metrics(P, ytrue, yhat, cfg)
P = double(P);
ytrue = double(ytrue(:));
yhat  = double(yhat(:));
N = numel(ytrue);
M = size(P,2);

if isfield(cfg,'low_m0_train')
    m0 = cfg.low_m0_train;
else
    m0 = low_threshold_m0(ytrue, M, cfg);
end
m0 = max(0, min(M-1, m0));

oky = isfinite(ytrue) & (ytrue>=0) & (ytrue<=M-1);
lowTrue  = oky & (ytrue <= m0);
highTrue = oky & (ytrue >  m0);

if any(lowTrue)
    Met.lowMiss = mean(yhat(lowTrue) > m0);
else
    Met.lowMiss = NaN;
end
Met.lowMAE = mean(abs(yhat(lowTrue) - ytrue(lowTrue)), 'omitnan');

if any(highTrue)
    Met.highMiss = mean(yhat(highTrue) <= m0);
else
    Met.highMiss = NaN;
end

if any(highTrue)
    Met.ocGap = mean(max(ytrue(highTrue) - yhat(highTrue), 0));
else
    Met.ocGap = NaN;
end

eps2 = 1e-12;
Met.entropy_mean = mean(-sum(P .* log(P + eps2), 2));
Met.maxprob_mean = mean(max(P,[],2));

idx = sub2ind([N M], find(oky), ytrue(oky)+1);
py = P(idx);
Met.nll = mean(-log(py + eps2));

Y = zeros(N,M);
Y(idx) = 1;
Met.brier = mean(sum((P - Y).^2, 2));

Met.ece = ece_maxprob(P(oky,:), ytrue(oky), cfg.evalECE_bins);

conf = max(P,[],2);
score = 1 - conf;

score_ok = score(oky);
yhat_ok  = yhat(oky);
ytrue_ok = ytrue(oky);

err = double(yhat_ok ~= ytrue_ok);
Met.auroc_err = auroc_binary(score_ok, err);

if isfield(cfg,'evalAURC_useMAE') && cfg.evalAURC_useMAE
    riskPerSample = abs(yhat_ok - ytrue_ok);
else
    riskPerSample = err;
end
Met.aurc = aurc_risk_coverage(score_ok, riskPerSample, cfg.evalSelective_pts);
end

function ece = ece_maxprob(P, ytrue, nBins)
P = double(P); ytrue = double(ytrue(:));
conf = max(P,[],2);
[~, yhat] = max(P,[],2); yhat = yhat - 1;

ok = isfinite(ytrue) & (ytrue>=0) & (ytrue<=max(yhat));
conf = conf(ok); yhat = yhat(ok); ytrue = ytrue(ok);
N = numel(ytrue);
if N==0, ece = NaN; return; end

edges = linspace(0,1,nBins+1);
ece = 0;
for b=1:nBins
    in = (conf >= edges(b)) & (conf < edges(b+1));
    if b==nBins, in = (conf >= edges(b)) & (conf <= edges(b+1)); end
    nb = sum(in);
    if nb==0, continue; end
    acc_b  = mean(yhat(in) == ytrue(in));
    conf_b = mean(conf(in));
    ece = ece + (nb/N) * abs(acc_b - conf_b);
end
end

function auc = auroc_binary(score, label01)
score = double(score(:));
y = double(label01(:));
ok = isfinite(score) & isfinite(y);
score = score(ok); y = y(ok);

pos = (y==1); neg = (y==0);
nPos = sum(pos); nNeg = sum(neg);
if nPos==0 || nNeg==0, auc = NaN; return; end

[ss, ord] = sort(score, 'ascend');
y_sorted = y(ord);
r_sorted = (1:numel(ss))';
i = 1;
while i <= numel(ss)
    j = i;
    while j < numel(ss) && ss(j+1) == ss(i), j = j + 1; end
    if j > i, r_sorted(i:j) = mean(r_sorted(i:j)); end
    i = j + 1;
end
sumRankPos = sum(r_sorted(y_sorted==1));
auc = (sumRankPos - nPos*(nPos+1)/2) / (nPos*nNeg);
end

function aurc = aurc_risk_coverage(score, riskPerSample, nPts)
score = double(score(:));
risk  = double(riskPerSample(:));
ok = isfinite(score) & isfinite(risk);
score = score(ok); risk = risk(ok);

N = numel(score);
if N==0, aurc = NaN; return; end

[~, idx] = sort(score, 'ascend');
riskS = risk(idx);

covGrid = linspace(max(1/N, 1/nPts), 1, nPts);
riskAtCov = zeros(size(covGrid));
for i=1:numel(covGrid)
    k = max(1, round(covGrid(i)*N));
    riskAtCov(i) = mean(riskS(1:k));
end

aurc = trapz(covGrid, riskAtCov);
end

%% ============================================================
%  SUMMARY / EXPORT / TABLES
% ============================================================
function C = complexity_estimate(cfg, M)
K = cfg.K;
d_xdeg = cfg.xdeg_dim;

C.FIXED_params = 0;
C.FIXED_MAC = (3*K) + (3*M);

C.FEATURE_params = ...
    (d_xdeg*64 + 64) + ...
    (64*32 + 32) + ...
    (32*16 + 16) + ...
    (16*M + M);

C.FEATURE_MAC = ...
    d_xdeg*64 + ...
    64*32 + ...
    32*16 + ...
    16*M + M;

C.E2E_params = (K*128+128) + (128*64+64) + (64*M+M);
C.E2E_MAC    = K*128 + 128*64 + 64*M + M;

% OURS-NOB
p_nob = ...
    (d_xdeg*cfg.ours_phys_hidden1 + cfg.ours_phys_hidden1) + ...
    (cfg.ours_phys_hidden1*cfg.ours_phys_hidden2 + cfg.ours_phys_hidden2) + ...
    (cfg.ours_phys_hidden2*2 + 2) + ...
    max(M-2,0);

mac_nob = ...
    d_xdeg*cfg.ours_phys_hidden1 + ...
    cfg.ours_phys_hidden1*cfg.ours_phys_hidden2 + ...
    cfg.ours_phys_hidden2*2 + ...
    (3*K) + (3*M);

C.OURSNOB_params = p_nob;
C.OURSNOB_MAC    = mac_nob;

% OURS-FUSION
fd = cfg.ours_data_hidden2;

p_data = ...
    (K*cfg.ours_data_hidden1 + cfg.ours_data_hidden1) + ...
    (cfg.ours_data_hidden1*cfg.ours_data_hidden2 + cfg.ours_data_hidden2);

p_phys = ...
    (d_xdeg*cfg.ours_phys_hidden1 + cfg.ours_phys_hidden1) + ...
    (cfg.ours_phys_hidden1*cfg.ours_phys_hidden2 + cfg.ours_phys_hidden2) + ...
    (cfg.ours_phys_hidden2*3 + 3);

p_fuse = ...
    (fd*4 + fd) + ...
    (fd*(2*fd) + fd) + ...
    (M*fd + M) + ...
    (1*fd + 1);

p_ord = max(M-2,0);

C.OURSFUSION_params = p_data + p_phys + p_fuse + p_ord;

C.OURSFUSION_MAC = ...
    (K*cfg.ours_data_hidden1 + cfg.ours_data_hidden1*cfg.ours_data_hidden2) + ...
    (d_xdeg*cfg.ours_phys_hidden1 + cfg.ours_phys_hidden1*cfg.ours_phys_hidden2 + cfg.ours_phys_hidden2*3) + ...
    (fd*4 + fd*(2*fd) + M*fd) + ...
    (3*K) + (3*M);
end

%% ============================================================
%  GOODPUT
% ============================================================
function G = goodput_proxy_metrics(ytrue, yhat, levels)
ytrue = double(ytrue(:));
yhat  = double(yhat(:));

ok = isfinite(ytrue) & isfinite(yhat);
ytrue = ytrue(ok);
yhat  = yhat(ok);

yhat = max(min(yhat, numel(levels)-1), 0);

rate_true = double(levels(ytrue + 1));
rate_hat  = double(levels(yhat  + 1));

succ = (yhat <= ytrue);

G.goodput = mean(rate_hat .* succ);
G.goodput_norm = G.goodput / (mean(rate_true) + 1e-12);
G.outage = mean(~succ);

if any(succ)
    G.underuse = mean(rate_true(succ) - rate_hat(succ));
else
    G.underuse = NaN;
end
end

%% ============================================================
%  PLOTS
% ============================================================

%% ============================================================
%  BETA-RISK / ESS
% ============================================================
function BetaRiskTab = run_beta_risk_analysis(Results, cfg)
if isempty(Results), BetaRiskTab = table(); return; end

targetSplits = string(cfg.beta_risk_split_list);
targetFrac   = cfg.beta_risk_use_frac;
targetSeed   = cfg.beta_risk_use_seed;
nbins        = cfg.beta_risk_nbins;

maxRows = numel(targetSplits);
split_col          = strings(maxRows,1);
N_col              = nan(maxRows,1);
pearson_col        = nan(maxRows,1);
spearman_col       = nan(maxRows,1);
risk_mean_col      = nan(maxRows,1);
risk_posrate_col   = nan(maxRows,1);
risk_lowbeta_col   = nan(maxRows,1);
risk_highbeta_col  = nan(maxRows,1);
beta_low_col       = nan(maxRows,1);
beta_high_col      = nan(maxRows,1);
p_ranksum_col      = nan(maxRows,1);

rows = 0;
for ss = 1:numel(targetSplits)
    splitNow = targetSplits(ss);
    hit = find_run(Results, splitNow, targetFrac, targetSeed);
    if isempty(hit), continue; end

    R = Results(hit);
    if ~isfield(R,'OURSFUSION_beta_val') || ~isfield(R,'y_val'), continue; end

    ytrue = double(R.y_val(:));
    if isfield(R,'OURSFUSION_yhat_val')
        yhat = double(R.OURSFUSION_yhat_val(:));
    else
        yhat = double(R.Det_OURSFUSION.yhat(:));
    end
    beta_v = double(R.OURSFUSION_beta_val(:));
    risk = max(ytrue - yhat, 0);

    if cfg.beta_risk_use_binary
        risk = double(risk > 0);
        ylabStr = 'Under-prediction indicator';
        ttlRisk = 'binary under-risk';
    else
        ylabStr = 'Under-prediction risk';
        ttlRisk = 'under-prediction risk';
    end

    ok = isfinite(beta_v) & isfinite(risk);
    beta_v = beta_v(ok);
    risk   = risk(ok);
    if numel(beta_v) < 10, continue; end

    rows = rows + 1;
    split_col(rows) = splitNow;
    N_col(rows) = numel(beta_v);
    pearson_col(rows)  = corr(beta_v, risk, 'Type', 'Pearson',  'Rows', 'complete');
    spearman_col(rows) = corr(beta_v, risk, 'Type', 'Spearman', 'Rows', 'complete');
    risk_mean_col(rows) = mean(risk, 'omitnan');
    risk_posrate_col(rows) = mean(risk > 0, 'omitnan');

    qv = quantile(beta_v, [0.2 0.8]);
    idxLow  = beta_v <= qv(1);
    idxHigh = beta_v >= qv(2);

    beta_low_col(rows) = mean(beta_v(idxLow), 'omitnan');
    beta_high_col(rows)= mean(beta_v(idxHigh), 'omitnan');
    risk_lowbeta_col(rows) = mean(risk(idxLow), 'omitnan');
    risk_highbeta_col(rows)= mean(risk(idxHigh), 'omitnan');

    try
        if any(idxLow) && any(idxHigh)
            p_ranksum_col(rows) = ranksum(risk(idxLow), risk(idxHigh));
        end
    catch
        p_ranksum_col(rows) = NaN;
    end

    figure('Color','w','Position',[100 100 620 500]); hold on;
    hsc = scatter(beta_v, risk, 18, 'filled');
    try
        hsc.MarkerFaceAlpha = 0.22;
    catch
    end
    [xb, yb] = make_binned_mean_curve(beta_v, risk, nbins);
    plot(xb, yb, '-o', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('\beta_t'); ylabel(ylabStr);
    title(sprintf('\\beta_t vs %s | split=%s | \\rho_s=%.3f', ttlRisk, char(splitNow), spearman_col(rows)));
    grid on; beautify_ax(gca);
    if cfg.save_analysis_fig
        fn = sprintf('beta_vs_underRisk_scatter_%s', char(splitNow));
        fn = regexprep(fn, '[^\w\d]+', '_');
        saveas(gcf, fullfile(cfg.analysis_fig_dir, [fn '.png']));
    end
end

BetaRiskTab = table(split_col(1:rows), N_col(1:rows), ...
    pearson_col(1:rows), spearman_col(1:rows), ...
    risk_mean_col(1:rows), risk_posrate_col(1:rows), ...
    beta_low_col(1:rows), beta_high_col(1:rows), ...
    risk_lowbeta_col(1:rows), risk_highbeta_col(1:rows), ...
    p_ranksum_col(1:rows), ...
    'VariableNames', {'split','N','pearson','spearman', ...
    'risk_mean','risk_posrate', ...
    'beta_low_group_mean','beta_high_group_mean', ...
    'risk_lowbeta','risk_highbeta','p_ranksum'});
end

function ESSTab = run_ess_entropy_analysis(Results, cfg)
if isempty(Results), ESSTab = table(); return; end

targetSplits = string(cfg.ess_split_list);
targetFrac   = cfg.ess_use_frac;
targetSeed   = cfg.ess_use_seed;
nbins        = cfg.ess_nbins;

maxRows = numel(targetSplits);
split_col             = strings(maxRows,1);
N_col                 = nan(maxRows,1);
rho_beta_ess_col      = nan(maxRows,1);
rho_gap_ess_col       = nan(maxRows,1);
rho_ess_risk_col      = nan(maxRows,1);
ess_mean_col          = nan(maxRows,1);
ent_mean_col          = nan(maxRows,1);
maxw_mean_col         = nan(maxRows,1);

rows = 0;
for ss = 1:numel(targetSplits)
    splitNow = targetSplits(ss);
    hit = find_run(Results, splitNow, targetFrac, targetSeed);
    if isempty(hit), continue; end

    R = Results(hit);
    if ~isfield(R,'MI_val') || ~isfield(R,'OURSFUSION_beta_val') || ~isfield(R,'Gap_mean_val') || ~isfield(R,'y_val')
        continue;
    end
        
    if isfield(R,'OURSFUSION_yhat_val')
        yhat = double(R.OURSFUSION_yhat_val(:));
    else
        yhat = double(R.Det_OURSFUSION.yhat(:));
    end

    beta_v  = double(R.OURSFUSION_beta_val(:));

    ytrue   = double(R.y_val(:));
    MI_mat  = double(R.MI_val);
    gap_v   = double(R.Gap_mean_val(:));
    risk_v  = max(ytrue - yhat, 0);

    Nv = numel(beta_v);
    ESS_v  = nan(Nv,1);
    ENT_v  = nan(Nv,1);
    MAXW_v = nan(Nv,1);

    for n = 1:Nv
        beta_n = max(beta_v(n), 1e-8);
        tmp = -beta_n * MI_mat(:,n);
        tmp = tmp - max(tmp);
        a = exp(tmp);
        s = sum(a);
        if ~isfinite(s) || s <= 0, continue; end
        p = a ./ s;
        ESS_v(n)  = 1 / sum(p.^2);
        ENT_v(n)  = -sum(p .* log(p + 1e-12));
        MAXW_v(n) = max(p);
    end

    ok = isfinite(beta_v) & isfinite(gap_v) & isfinite(risk_v) & ...
         isfinite(ESS_v) & isfinite(ENT_v) & isfinite(MAXW_v);

    beta_v = beta_v(ok);
    gap_v  = gap_v(ok);
    risk_v = risk_v(ok);
    ESS_v  = ESS_v(ok);
    ENT_v  = ENT_v(ok);
    MAXW_v = MAXW_v(ok);

    if numel(beta_v) < 10, continue; end

    rows = rows + 1;
    split_col(rows) = splitNow;
    N_col(rows) = numel(beta_v);
    rho_beta_ess_col(rows) = corr(beta_v, ESS_v, 'Type', 'Spearman', 'Rows', 'complete');
    rho_gap_ess_col(rows)  = corr(gap_v,  ESS_v, 'Type', 'Spearman', 'Rows', 'complete');
    rho_ess_risk_col(rows) = corr(ESS_v,  risk_v,'Type', 'Spearman', 'Rows', 'complete');
    ess_mean_col(rows) = mean(ESS_v, 'omitnan');
    ent_mean_col(rows) = mean(ENT_v, 'omitnan');
    maxw_mean_col(rows)= mean(MAXW_v,'omitnan');

    figure('Color','w','Position',[100 100 620 500]); hold on;
    hsc = scatter(beta_v, ESS_v, 18, 'filled');
    try
        hsc.MarkerFaceAlpha = 0.22;
    catch
    end
    [xb, yb] = make_binned_mean_curve(beta_v, ESS_v, nbins);
    plot(xb, yb, '-o', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('\beta_t'); ylabel('ESS');
    title(sprintf('\\beta_t vs ESS | split=%s | \\rho_s=%.3f', char(splitNow), rho_beta_ess_col(rows)));
    grid on; beautify_ax(gca);
    if cfg.save_analysis_fig
        fn = sprintf('beta_vs_ESS_%s', char(splitNow));
        fn = regexprep(fn, '[^\w\d]+', '_');
        saveas(gcf, fullfile(cfg.analysis_fig_dir, [fn '.png']));
    end
end

ESSTab = table(split_col(1:rows), N_col(1:rows), ...
    rho_beta_ess_col(1:rows), rho_gap_ess_col(1:rows), rho_ess_risk_col(1:rows), ...
    ess_mean_col(1:rows), ent_mean_col(1:rows), maxw_mean_col(1:rows), ...
    'VariableNames', {'split','N', ...
    'rho_beta_ess','rho_gap_ess','rho_ess_risk', ...
    'ESS_mean','Entropy_mean','MaxWeight_mean'});
end

%% ============================================================
%  split-frac tables
% ============================================================
function y = fc_dl(W, x, b)
    if isa(W, 'dlarray')
        W0 = stripdims(W);
    else
        W0 = W;
    end

    if isa(x, 'dlarray')
        x0 = stripdims(x);
    else
        x0 = x;
    end

    if isa(b, 'dlarray')
        b0 = stripdims(b);
    else
        b0 = b;
    end

    y0 = W0 * x0 + b0;
    y  = dlarray(y0, 'CB');
end

function y = concat_channel(a, b)
% concatenate along channel dimension for CB arrays
    if isa(a,'dlarray'), a = stripdims(a); end
    if isa(b,'dlarray'), b = stripdims(b); end
    y = dlarray([a; b], 'CB');
end

function [best, Hist, Time] = train_ours_nob_ordinal( ...
    netPhys, deltas_raw, ...
    MI_tr_raw, Xdeg_tr, y_tr, ...
    MI_va_raw, Xdeg_va, y_va, M, cfg)

numEpochs = cfg.numEpochs;
bs = cfg.batchSize;
lr = cfg.lr;

trN = numel(y_tr);
iters = ceil(trN / bs);

avgGP = []; avgSqGP = [];
avgGA = []; avgSqGA = [];

bestScore = -inf;
noImp = 0;

best.netPhys = netPhys;
best.deltas  = deltas_raw;

Hist.epoch = (1:numEpochs)';
Hist.trainLoss = nan(numEpochs,1);
Hist.valAcc = nan(numEpochs,1);
Hist.valMAE = nan(numEpochs,1);
epochTimes = nan(numEpochs,1);

for e = 1:numEpochs
    te = tic;
    p = randperm(trN);
    epochLoss = 0;

    for it = 1:iters
        i1 = (it-1)*bs + 1;
        i2 = min(it*bs, trN);
        idx = p(i1:i2);

        MIb_r = MI_tr_raw(:, idx);
        Xdb   = Xdeg_tr(idx,:);
        yb    = int32(y_tr(idx(:)));

        dlMIr  = dlarray(single(MIb_r), 'CB');
        dlXdeg = dlarray(single(Xdb)', 'CB');

        [loss, gPhys, gAux] = dlfeval(@grads_ours_nob_ordinal, ...
            netPhys, deltas_raw, dlMIr, dlXdeg, yb, M, cfg);

        epochLoss = epochLoss + double(gather(extractdata(loss)));

        [netPhys, avgGP, avgSqGP] = adamupdate(netPhys, gPhys, avgGP, avgSqGP, (e-1)*iters+it, lr);

        auxParams = {deltas_raw};
        [auxParams, avgGA, avgSqGA] = adamupdate_cell(auxParams, gAux, avgGA, avgSqGA, (e-1)*iters+it, lr);
        deltas_raw = auxParams{1};
    end

    epochLoss = epochLoss / iters;
    Hist.trainLoss(e) = epochLoss;

    curr.netPhys = netPhys;
    curr.deltas  = deltas_raw;

    [acc_va, mae_va, ~] = eval_ours_nob_ordinal_basic(curr, MI_va_raw, Xdeg_va, y_va, M, cfg);
    Hist.valAcc(e) = acc_va;
    Hist.valMAE(e) = mae_va;
    epochTimes(e) = toc(te);

    if acc_va > bestScore + cfg.minDeltaAcc
        bestScore = acc_va;
        noImp = 0;
        best.netPhys = netPhys;
        best.deltas  = deltas_raw;
    else
        noImp = noImp + 1;
    end

    if cfg.useEarlyStop && noImp >= cfg.patience
        Hist = trim_hist(Hist, e);
        epochTimes = epochTimes(1:e);
        break;
    end
end

Hist = trim_hist(Hist, find(~isnan(Hist.trainLoss),1,'last'));
epochTimes = epochTimes(1:numel(Hist.epoch));
Time.epoch_s_mean = mean(epochTimes,'omitnan');
Time.epoch_s_std  = std(epochTimes,'omitnan');
end

function [loss, gradsPhys, gradsAux] = grads_ours_nob_ordinal( ...
    netPhys, deltas_raw, dlMIr, dlXdeg, y, M, cfg)

eps1 = dlarray(cfg.eps1);
eps2 = dlarray(cfg.eps2);

o = forward(netPhys, dlXdeg);
beta = bounded_sigmoid_dl(o(1,:), cfg.beta_min, cfg.beta_max);
w    = bounded_sigmoid_dl(o(2,:), cfg.w_min,    cfg.w_max);

z = w .* mi_eesm(dlMIr, beta, eps2);

theta = build_theta_anchor(deltas_raw, M, eps1);
P = ordinal_probs(z, theta, M, eps2);   % [B,M]

B = size(P,1);
yy = double(y(:)) + 1;
idx = sub2ind([B,M], (1:B)', yy);
py = P(idx);

loss_cls = -mean(log(py + eps2));

mvec = dlarray(single(0:M-1));
yexp = sum(P .* reshape(mvec,1,[]), 2);
loss_ord = mean(abs(yexp - single(y(:))));

loss = loss_cls;
if cfg.useAuxOrdinalMAE
    loss = loss + single(cfg.auxMAEWeight) * loss_ord;
end

if cfg.useBetaWReg
    loss = loss ...
        + single(cfg.betaRegWeight) * mean((beta(:) - single(cfg.betaCenter)).^2) ...
        + single(cfg.wRegWeight)    * mean((w(:)    - single(cfg.wCenter)).^2);
end

varsPhys = netPhys.Learnables.Value;
varsAux  = {deltas_raw};
varsAll = [varsPhys; varsAux(:)];

gAll = dlgradient(loss, varsAll);

nP = numel(varsPhys);
gradsPhys = netPhys.Learnables;
gradsPhys.Value = gAll(1:nP);
gradsAux = gAll(nP+1:end);
end

function [acc, mae, acc_pm1, Met, Det] = eval_ours_nob_ordinal(best, MIr, Xdeg, y, M, cfg)
[z, P, yhat] = eval_ours_nob_ordinal_details(best, MIr, Xdeg, M, cfg);
acc = mean(yhat == y(:));
mae = mean(abs(yhat - y(:)));
acc_pm1 = mean(abs(yhat - y(:)) <= 1);
Met = risk_unc_metrics(P, y(:), yhat, cfg);
Det.P = P;
Det.ytrue = y(:);
Det.yhat = yhat;
Det.z = z;
end

function [acc, mae, acc_pm1] = eval_ours_nob_ordinal_basic(best, MIr, Xdeg, y, M, cfg)
[~, ~, yhat] = eval_ours_nob_ordinal_details(best, MIr, Xdeg, M, cfg);
acc = mean(yhat == y(:));
mae = mean(abs(yhat - y(:)));
acc_pm1 = mean(abs(yhat - y(:)) <= 1);
end

function [z, P, yhat] = eval_ours_nob_ordinal_details(best, MIr, Xdeg, M, cfg)
dlMIr  = dlarray(single(MIr), 'CB');
dlXdeg = dlarray(single(Xdeg)', 'CB');

[z_dl, P_dl, ~, ~] = forward_ours_nob(best, dlMIr, dlXdeg, M, cfg);

P = gather(extractdata(P_dl));      % [B,M]
[~, yhat] = max(P, [], 2);
yhat = yhat - 1;
z = gather(extractdata(z_dl(:)));
end

function [z, P, yhat, beta_s, w_s] = eval_ours_nob_ordinal_details_full(best, MIr, Xdeg, M, cfg)
dlMIr  = dlarray(single(MIr), 'CB');
dlXdeg = dlarray(single(Xdeg)', 'CB');

[z_dl, P_dl, beta, w] = forward_ours_nob(best, dlMIr, dlXdeg, M, cfg);

P = gather(extractdata(P_dl));      % [B,M]
[~, yhat] = max(P, [], 2);
yhat = yhat - 1;

beta_s = gather(extractdata(beta(:)));
w_s    = gather(extractdata(w(:)));
z      = gather(extractdata(z_dl(:)));
end

function [z, P, beta, w] = forward_ours_nob(best, dlMIr, dlXdeg, M, cfg)
eps1 = cfg.eps1;
eps2 = cfg.eps2;

o = forward(best.netPhys, dlXdeg);
beta = bounded_sigmoid_dl(o(1,:), cfg.beta_min, cfg.beta_max);
w    = bounded_sigmoid_dl(o(2,:), cfg.w_min,    cfg.w_max);

z = w .* mi_eesm(dlMIr, beta, eps2);

theta = build_theta_anchor(best.deltas, M, eps1);
P = ordinal_probs(z, theta, M, eps2);   % [B,M]
end

function lbl = method_key_to_label(key, METHODS)
lbl = string(key);
for i = 1:numel(METHODS)
    if strcmpi(METHODS(i).key, key)
        lbl = string(METHODS(i).label);
        return;
    end
end
end

function p = ordinal_probs_fixed(z, theta)
M = numel(theta) + 1;
p = zeros(1,M);
sig = @(x) 1 ./ (1 + exp(-x));
F = sig(theta(:)' - z);

if M == 1
    p = 1;
    return;
end

p(1) = max(F(1), 1e-12);
for m = 2:M-1
    p(m) = max(F(m)-F(m-1), 1e-12);
end
p(M) = max(1-F(M-1), 1e-12);
p = p ./ sum(p);
end

function S = append_result_struct(S, x)
if isempty(S)
    S = x;
    return;
end

fnS = fieldnames(S);
fnx = fieldnames(x);

for i = 1:numel(fnS)
    if ~isfield(x, fnS{i})
        x.(fnS{i}) = [];
    end
end

for i = 1:numel(fnx)
    if ~isfield(S, fnx{i})
        [S.(fnx{i})] = deal([]);
    end
end

x = orderfields(x, S);
S(end+1) = x;
end

%% ============================================================
%  EXTRA ANALYSIS / ABLATION / CALIBRATION / CONFUSION
% ============================================================
function tf = should_run_fusion_ablation(runCfg, cfg)
tf = ismember(string(runCfg.split_mode), string(cfg.ablation_split_list)) && ...
     any(abs(runCfg.train_subset_frac - cfg.ablation_frac_list) < 1e-12) && ...
     ismember(runCfg.rng_seed, cfg.ablation_seed_list);
end

function AblationResults = run_fusion_ablation_suite( ...
    AblationResults, runCfg, ...
    MI_trs, MI_tr, Xdeg_trs, y_tr, ...
    MI_vas, MI_va, Xdeg_vas, y_va, M, levels)

specs = runCfg.ablation_specs;

for i = 1:2:numel(specs)
    abName = specs{i};
    abMods = specs{i+1};

    % ---------- clone cfg and apply ablation overrides ----------
    abCfg = runCfg;
    fns = fieldnames(abMods);
    for k = 1:numel(fns)
        abCfg.(fns{k}) = abMods.(fns{k});
    end

    % ---------- fix RNG so different ablations are comparable ----------
    rng(double(runCfg.rng_seed), 'twister');

    % ---------- feature-group ablation on x_deg ----------
    Xtr_ab = Xdeg_trs;
    Xva_ab = Xdeg_vas;
    if isfield(abCfg, 'xdeg_zero_idx') && ~isempty(abCfg.xdeg_zero_idx)
        idx = abCfg.xdeg_zero_idx;
        idx = idx(idx >= 1 & idx <= size(Xtr_ab,2));
        if ~isempty(idx)
            Xtr_ab(:, idx) = 0;
            Xva_ab(:, idx) = 0;
        end
    end

    % ---------- build data branch ----------
    layersData = [
        featureInputLayer(abCfg.K,'Normalization','none','Name','mi_in')
        fullyConnectedLayer(abCfg.ours_data_hidden1,'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(abCfg.ours_data_hidden2,'Name','fc2')
        reluLayer('Name','relu2')
    ];
    netData = dlnetwork(layerGraph(layersData));

    % ---------- build physics branch ----------
    layersPhys = [
        featureInputLayer(size(Xtr_ab,2),'Normalization','none','Name','xdeg_in')
        fullyConnectedLayer(abCfg.ours_phys_hidden1,'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(abCfg.ours_phys_hidden2,'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(3,'Name','phys_out')
    ];
    netPhys = dlnetwork(layerGraph(layersPhys));

    % ---------- ordinal deltas ----------
    if M >= 3
        deltas_raw = dlarray(single(0.2 * ones(M-2,1)));
    else
        deltas_raw = dlarray(single([]));
    end

    % ---------- fusion params ----------
    fuseParams = init_ours_fusion_params(abCfg, M);

    % ---------- train ----------
    t_train = tic;
    [best, ~, Time4] = train_ours_pg_fusion( ...
        netData, netPhys, deltas_raw, fuseParams, ...
        MI_trs, MI_tr, Xtr_ab, y_tr, ...
        MI_vas, MI_va, Xva_ab, y_va, M, abCfg);
    train_s = toc(t_train);

    % ---------- evaluate ----------
    [acc, mae, accpm1, Met, Det] = eval_ours_pg_fusion( ...
        best, MI_vas, MI_va, Xva_ab, y_va, M, abCfg);

    infer_total = time_infer_repeated( ...
        @() eval_ours_pg_fusion_details(best, MI_vas, MI_va, Xva_ab, M, abCfg), ...
        abCfg.time_infer_reps);

    infer_us = 1e6 * infer_total / numel(y_va);
    GP = goodput_proxy_metrics(y_va, Det.yhat, levels);

    % ---------- record ----------
    rr = struct();
    rr.split = runCfg.split_mode;
    rr.frac  = runCfg.train_subset_frac;
    rr.seed  = runCfg.rng_seed;
    rr.ablation = string(abName);

    rr.acc = acc;
    rr.mae = mae;
    rr.accpm1 = accpm1;
    rr.lowMiss = Met.lowMiss;
    rr.highMiss = Met.highMiss;
    rr.ece = Met.ece;
    rr.goodput_norm = GP.goodput_norm;
    rr.outage = GP.outage;
    rr.train_s = train_s;
    rr.epoch_s_mean = Time4.epoch_s_mean;
    rr.infer_us = infer_us;

    if isempty(AblationResults)
        AblationResults = rr;
    else
        AblationResults(end+1) = rr;
    end

    fprintf('[ABLATION] %-14s | split=%s | frac=%.3f | acc=%.4f | mae=%.4f | highMiss=%.4f | gp=%.4f\n', ...
        char(rr.ablation), char(string(rr.split)), rr.frac, rr.acc, rr.mae, rr.highMiss, rr.goodput_norm);
end
end

function T = flatten_ablation_results(AblationResults)
if isempty(AblationResults), T = table(); return; end
n = numel(AblationResults);

split = strings(n,1);
frac = nan(n,1);
seed = nan(n,1);
ablation = strings(n,1);
acc = nan(n,1); mae = nan(n,1); accpm1 = nan(n,1);
lowMiss = nan(n,1); highMiss = nan(n,1); ece = nan(n,1);
goodput_norm = nan(n,1); outage = nan(n,1);
train_s = nan(n,1); epoch_s_mean = nan(n,1); infer_us = nan(n,1);

for i = 1:n
    split(i) = string(AblationResults(i).split);
    frac(i) = AblationResults(i).frac;
    seed(i) = AblationResults(i).seed;
    ablation(i) = string(AblationResults(i).ablation);
    acc(i) = AblationResults(i).acc;
    mae(i) = AblationResults(i).mae;
    accpm1(i) = AblationResults(i).accpm1;
    lowMiss(i) = AblationResults(i).lowMiss;
    highMiss(i) = AblationResults(i).highMiss;
    ece(i) = AblationResults(i).ece;
    goodput_norm(i) = AblationResults(i).goodput_norm;
    outage(i) = AblationResults(i).outage;
    train_s(i) = AblationResults(i).train_s;
    epoch_s_mean(i) = AblationResults(i).epoch_s_mean;
    infer_us(i) = AblationResults(i).infer_us;
end

T = table(split, frac, seed, ablation, acc, mae, accpm1, lowMiss, highMiss, ece, ...
    goodput_norm, outage, train_s, epoch_s_mean, infer_us);
end

function T = summarize_fusion_ablation(AblationResults)
if isempty(AblationResults), T = table(); return; end
Tall = flatten_ablation_results(AblationResults);

[G, splitU, fracU, abU] = findgroups(Tall.split, Tall.frac, Tall.ablation);
acc_mean = splitapply(@(x) mean(x,'omitnan'), Tall.acc, G);
acc_std  = splitapply(@(x) std(x,'omitnan'),  Tall.acc, G);
mae_mean = splitapply(@(x) mean(x,'omitnan'), Tall.mae, G);
high_mean= splitapply(@(x) mean(x,'omitnan'), Tall.highMiss, G);
ece_mean = splitapply(@(x) mean(x,'omitnan'), Tall.ece, G);
gp_mean  = splitapply(@(x) mean(x,'omitnan'), Tall.goodput_norm, G);
train_mean = splitapply(@(x) mean(x,'omitnan'), Tall.train_s, G);
infer_mean = splitapply(@(x) mean(x,'omitnan'), Tall.infer_us, G);
nRuns = splitapply(@numel, Tall.acc, G);

T = table(splitU, fracU, abU, nRuns, acc_mean, acc_std, mae_mean, high_mean, ...
    ece_mean, gp_mean, train_mean, infer_mean, ...
    'VariableNames', {'split','frac','ablation','nRuns','acc_mean','acc_std','mae_mean', ...
    'highMiss_mean','ece_mean','goodput_norm_mean','train_s_mean','infer_us_mean'});

T.delta_acc_vs_FULL = nan(height(T),1);
T.delta_highMiss_vs_FULL = nan(height(T),1);
T.delta_gp_vs_FULL = nan(height(T),1);

for i = 1:height(T)
    idxFull = (T.split == T.split(i)) & (abs(T.frac - T.frac(i)) < 1e-12) & (T.ablation == "FULL");
    if any(idxFull)
        T.delta_acc_vs_FULL(i) = T.acc_mean(i) - T.acc_mean(idxFull);
        T.delta_highMiss_vs_FULL(i) = T.highMiss_mean(i) - T.highMiss_mean(idxFull);
        T.delta_gp_vs_FULL(i) = T.goodput_norm_mean(i) - T.goodput_norm_mean(idxFull);
    end
end
end

function plot_fusion_ablation(AblationResults, cfg)
if isempty(AblationResults), return; end
T = summarize_fusion_ablation(AblationResults);
splits = unique(T.split, 'stable');

for s = 1:numel(splits)
    Ts = T(T.split == splits(s), :);
    if isempty(Ts), continue; end

    % 按 Accuracy 排序，三张图保持同一顺序
    [~, ord] = sort(Ts.acc_mean, 'descend');
    Ts = Ts(ord,:);

    labels = cellstr(Ts.ablation);
    cmap   = get_ablation_colors(labels);

    fig = figure('Color','w','Position',[100 100 1180 420]);

    % ---------------- Accuracy ----------------
    ax1 = subplot(1,3,1);
    b1 = bar(ax1, Ts.acc_mean, 'FaceColor','flat', ...
        'EdgeColor',[0.25 0.25 0.25], 'LineWidth',0.8);
    b1.CData = cmap;
    hold(ax1, 'on');
    errorbar(ax1, 1:height(Ts), Ts.acc_mean, Ts.acc_std, ...
        '.', 'Color',[0.15 0.15 0.15], 'LineWidth',1.1, 'CapSize',7);
    set(ax1, 'XTick',1:height(Ts), 'XTickLabel',labels);
    xtickangle(ax1, 35);
    ylabel(ax1, 'Accuracy');
    title(ax1, sprintf('Fusion ablation | %s', char(splits(s))));
    beautify_ax(ax1);

    % ---------------- Risk ----------------
    ax2 = subplot(1,3,2);
    b2 = bar(ax2, Ts.highMiss_mean, 'FaceColor','flat', ...
        'EdgeColor',[0.25 0.25 0.25], 'LineWidth',0.8);
    b2.CData = cmap;
    hold(ax2, 'on');
    if ismember('highMiss_std', Ts.Properties.VariableNames)
        errorbar(ax2, 1:height(Ts), Ts.highMiss_mean, Ts.highMiss_std, ...
            '.', 'Color',[0.15 0.15 0.15], 'LineWidth',1.1, 'CapSize',7);
    end
    set(ax2, 'XTick',1:height(Ts), 'XTickLabel',labels);
    xtickangle(ax2, 35);
    ylabel(ax2, 'HighMiss');
    title(ax2, 'Risk');
    beautify_ax(ax2);

    % ---------------- Efficiency ----------------
    ax3 = subplot(1,3,3);
    b3 = bar(ax3, Ts.goodput_norm_mean, 'FaceColor','flat', ...
        'EdgeColor',[0.25 0.25 0.25], 'LineWidth',0.8);
    b3.CData = cmap;
    hold(ax3, 'on');
    if ismember('goodput_norm_std', Ts.Properties.VariableNames)
        errorbar(ax3, 1:height(Ts), Ts.goodput_norm_mean, Ts.goodput_norm_std, ...
            '.', 'Color',[0.15 0.15 0.15], 'LineWidth',1.1, 'CapSize',7);
    end
    set(ax3, 'XTick',1:height(Ts), 'XTickLabel',labels);
    xtickangle(ax3, 35);
    ylabel(ax3, 'Goodput norm');
    title(ax3, 'Efficiency');
    beautify_ax(ax3);

    if cfg.save_analysis_fig
        fn = sprintf('Fusion_Ablation_%s', char(splits(s)));
        fn = regexprep(fn,'[^\w\d]+','_');
        exportgraphics(fig, fullfile(cfg.analysis_fig_dir,[fn '.png']), 'Resolution', 300);
    end
end
end

function T = build_split_label_distribution_table(Results, levels)
if isempty(Results), T = table(); return; end

idxUse = arrayfun(@(r) abs(r.frac - 1.0) < 1e-12, Results);
R = Results(idxUse);

rows = {};
for i = 1:numel(R)
    splitNow = string(R(i).split);
    seedNow  = R(i).seed;

    yt = double(R(i).y_train(:));
    yv = double(R(i).y_val(:));

    ht = histcounts(yt, -0.5:1:(numel(levels)-0.5));
    hv = histcounts(yv, -0.5:1:(numel(levels)-0.5));

    for m = 1:numel(levels)
        rows(end+1,:) = {splitNow, seedNow, 'train', m-1, levels(m), ht(m)}; %#ok<AGROW>
        rows(end+1,:) = {splitNow, seedNow, 'val',   m-1, levels(m), hv(m)}; %#ok<AGROW>
    end
end

T = cell2table(rows, 'VariableNames', {'split','seed','side','class','level_value','count'});
end

function plot_split_label_distribution(Results, levels, cfg)
if isempty(Results), return; end
idxUse = arrayfun(@(r) abs(r.frac - 1.0) < 1e-12 & r.seed == cfg.seeds(1), Results);
R = Results(idxUse);
if isempty(R), return; end

for i = 1:numel(R)
    yt = double(R(i).y_train(:));
    yv = double(R(i).y_val(:));
    ht = histcounts(yt, -0.5:1:(numel(levels)-0.5));
    hv = histcounts(yv, -0.5:1:(numel(levels)-0.5));

    fig = figure('Color','w','Position',[100 100 760 420]); hold on;
    b = bar([(0:numel(levels)-1)' [ht(:) hv(:)]], 'grouped'); %#ok<NASGU>
    xlabel('Class'); ylabel('Count');
    legend({'train','val'}, 'Location','best');
    title(sprintf('Label distribution | split=%s | frac=1 | seed=%d', R(i).split, R(i).seed));
    beautify_ax(gca);

    if cfg.save_analysis_fig
        fn = sprintf('Label_Distribution_%s', char(R(i).split));
        fn = regexprep(fn,'[^\w\d]+','_');
        exportgraphics(fig, fullfile(cfg.analysis_fig_dir,[fn '.png']), 'Resolution', 300);
    end
end
end

function plot_confusion_analysis(Results, levels, cfg, METHODS)
targetSplits = string(cfg.confusion_split_list);
for ss = 1:numel(targetSplits)
    hit = find_run(Results, targetSplits(ss), cfg.confusion_use_frac, cfg.confusion_use_seed);
    if isempty(hit), continue; end
    R = Results(hit);

    methKeys = {'FEATURE','E2E','OURSNOB','OURSFUSION'};
    for mm = 1:numel(methKeys)
        key = methKeys{mm};
        if ~isfield(R, ['Det_' key]), continue; end
        ytrue = double(R.(['Det_' key]).ytrue(:));
        yhat  = double(R.(['Det_' key]).yhat(:));

        C = confusionmat(ytrue, yhat, 'Order', 0:numel(levels)-1);
        if strcmpi(cfg.confusion_norm, 'row')
            Cn = C ./ max(sum(C,2), 1);
        else
            Cn = C;
        end

        fig = figure('Color','w','Position',[100 100 600 520]);
        imagesc(Cn);
        axis image; colorbar;
        xlabel('Predicted class'); ylabel('True class');
        title(sprintf('Confusion | %s | %s', char(targetSplits(ss)), method_key_to_label(key, METHODS)));
        set(gca,'XTick',1:numel(levels),'XTickLabel',0:numel(levels)-1);
        set(gca,'YTick',1:numel(levels),'YTickLabel',0:numel(levels)-1);
        beautify_ax(gca);

        if cfg.save_analysis_fig
            fn = sprintf('Confusion_%s_%s', char(targetSplits(ss)), key);
            fn = regexprep(fn,'[^\w\d]+','_');
            exportgraphics(fig, fullfile(cfg.analysis_fig_dir,[fn '.png']), 'Resolution', 300);
        end
    end
end
end

function T = build_classwise_metric_table(Results, levels, cfg, METHODS)
rows = {};
targetSplits = string(cfg.classwise_split_list);
methKeys = {'FEATURE','E2E','OURSNOB','OURSFUSION'};

for ss = 1:numel(targetSplits)
    hit = find_run(Results, targetSplits(ss), cfg.classwise_use_frac, cfg.classwise_use_seed);
    if isempty(hit), continue; end
    R = Results(hit);

    for mm = 1:numel(methKeys)
        key = methKeys{mm};
        if ~isfield(R, ['Det_' key]), continue; end

        ytrue = double(R.(['Det_' key]).ytrue(:));
        yhat  = double(R.(['Det_' key]).yhat(:));

        for c = 0:numel(levels)-1
            idx = (ytrue == c);
            if any(idx)
                recall_c = mean(yhat(idx) == c);
            else
                recall_c = NaN;
            end

            idxp = (yhat == c);
            if any(idxp)
                prec_c = mean(ytrue(idxp) == c);
            else
                prec_c = NaN;
            end

            rows(end+1,:) = {targetSplits(ss), method_key_to_label(key, METHODS), c, levels(c+1), ... %#ok<AGROW>
                sum(idx), recall_c, prec_c};
        end
    end
end

if isempty(rows)
    T = table();
else
    T = cell2table(rows, 'VariableNames', ...
        {'split','method','class','level_value','support','recall','precision'});
end
end

function plot_classwise_recall(Results, levels, cfg, METHODS)
targetSplits = string(cfg.classwise_split_list);
methKeys = {'FEATURE','E2E','OURSNOB','OURSFUSION'};

for ss = 1:numel(targetSplits)
    hit = find_run(Results, targetSplits(ss), cfg.classwise_use_frac, cfg.classwise_use_seed);
    if isempty(hit), continue; end
    R = Results(hit);

    fig = figure('Color','w','Position',[100 100 780 440]); hold on;
    xs = 0:numel(levels)-1;

    for mm = 1:numel(methKeys)
        key = methKeys{mm};
        if ~isfield(R, ['Det_' key]), continue; end

        ytrue = double(R.(['Det_' key]).ytrue(:));
        yhat  = double(R.(['Det_' key]).yhat(:));
        rec = nan(numel(xs),1);
        for c = xs
            idx = (ytrue == c);
            if any(idx), rec(c+1) = mean(yhat(idx) == c); end
        end

        sty = method_style(key);
        plot(xs, rec, '-','LineWidth',1.8,'Color',sty.color, ...
            'Marker',sty.marker,'MarkerFaceColor',sty.color,'MarkerSize',7);
    end

    xlabel('Class'); ylabel('Recall');
    title(sprintf('Class-wise recall | split=%s', char(targetSplits(ss))));
    legend({'DEG-MLP','MI-DNN','MI-ADA','MI-FUSE'}, 'Location','best');
    beautify_ax(gca);

    if cfg.save_analysis_fig
        fn = sprintf('Classwise_Recall_%s', char(targetSplits(ss)));
        fn = regexprep(fn,'[^\w\d]+','_');
        exportgraphics(fig, fullfile(cfg.analysis_fig_dir,[fn '.png']), 'Resolution', 300);
    end
end
end

function T = build_reliability_table(Results, cfg)
rows = {};
targetSplits = string(cfg.reliability_split_list);

for ss = 1:numel(targetSplits)
    hit = find_run(Results, targetSplits(ss), cfg.reliability_use_frac, cfg.reliability_use_seed);
    if isempty(hit), continue; end
    R = Results(hit);

    for mm = 1:numel(cfg.reliability_methods)
        key = cfg.reliability_methods{mm};
        if ~isfield(R, ['Det_' key]), continue; end
        P = double(R.(['Det_' key]).P);
        y = double(R.(['Det_' key]).ytrue(:));

        conf = max(P,[],2);
        [~, yh] = max(P,[],2); yh = yh - 1;
        acc = double(yh == y);

        edges = linspace(0,1,cfg.evalECE_bins+1);
        for b = 1:cfg.evalECE_bins
            in = conf >= edges(b) & conf < edges(b+1);
            if b == cfg.evalECE_bins
                in = conf >= edges(b) & conf <= edges(b+1);
            end
            if ~any(in), continue; end
            rows(end+1,:) = {targetSplits(ss), string(key), b, mean(conf(in)), mean(acc(in)), sum(in)}; %#ok<AGROW>
        end
    end
end

if isempty(rows)
    T = table();
else
    T = cell2table(rows, 'VariableNames', ...
        {'split','method','bin','mean_conf','mean_acc','count'});
end
end

function plot_reliability_analysis(Results, cfg, METHODS)
targetSplits = string(cfg.reliability_split_list);

for ss = 1:numel(targetSplits)
    hit = find_run(Results, targetSplits(ss), cfg.reliability_use_frac, cfg.reliability_use_seed);
    if isempty(hit), continue; end
    R = Results(hit);

    fig = figure('Color','w','Position',[100 100 900 700]);
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    for mm = 1:min(4, numel(cfg.reliability_methods))
        key = cfg.reliability_methods{mm};
        if ~isfield(R, ['Det_' key]), continue; end
        nexttile; hold on;

        P = double(R.(['Det_' key]).P);
        y = double(R.(['Det_' key]).ytrue(:));
        conf = max(P,[],2);
        [~, yh] = max(P,[],2); yh = yh - 1;
        acc = double(yh == y);

        edges = linspace(0,1,cfg.evalECE_bins+1);
        xc = nan(cfg.evalECE_bins,1);
        yc = nan(cfg.evalECE_bins,1);

        for b = 1:cfg.evalECE_bins
            in = conf >= edges(b) & conf < edges(b+1);
            if b == cfg.evalECE_bins
                in = conf >= edges(b) & conf <= edges(b+1);
            end
            if any(in)
                xc(b) = mean(conf(in));
                yc(b) = mean(acc(in));
            end
        end

        plot([0 1],[0 1],'k--','LineWidth',1);
        plot(xc, yc, '-o','LineWidth',1.8,'MarkerSize',6);
        xlabel('Confidence'); ylabel('Accuracy');
        title(sprintf('%s | %s', char(targetSplits(ss)), method_key_to_label(key, METHODS)));
        xlim([0 1]); ylim([0 1]);
        beautify_ax(gca);
    end

    if cfg.save_analysis_fig
        fn = sprintf('Reliability_%s', char(targetSplits(ss)));
        fn = regexprep(fn,'[^\w\d]+','_');
        exportgraphics(fig, fullfile(cfg.analysis_fig_dir,[fn '.png']), 'Resolution', 300);
    end
end
end

function plot_risk_coverage_analysis(Results, cfg, METHODS)
targetSplits = string(cfg.rc_split_list);

for ss = 1:numel(targetSplits)
    hit = find_run(Results, targetSplits(ss), cfg.rc_use_frac, cfg.rc_use_seed);
    if isempty(hit), continue; end
    R = Results(hit);

    fig = figure('Color','w','Position',[100 100 760 460]); hold on;

    for mm = 1:numel(cfg.rc_methods)
        key = cfg.rc_methods{mm};
        if ~isfield(R, ['Det_' key]), continue; end

        P = double(R.(['Det_' key]).P);
        y = double(R.(['Det_' key]).ytrue(:));
        yh = double(R.(['Det_' key]).yhat(:));
        conf = max(P,[],2);
        score = 1 - conf;
        risk = double(yh ~= y);

        [covGrid, riskCurve] = selective_risk_curve(score, risk, cfg.rc_npts);

        sty = method_style(key);
        plot(covGrid, riskCurve, 'LineWidth',1.8,'Color',sty.color, ...
            'Marker',sty.marker,'MarkerSize',6,'MarkerFaceColor',sty.color);
    end

    xlabel('Coverage'); ylabel('Selective risk');
    title(sprintf('Risk-Coverage | split=%s', char(targetSplits(ss))));
    legend(cellfun(@(k) char(method_key_to_label(k, METHODS)), cfg.rc_methods, 'UniformOutput', false), ...
        'Location','best');
    beautify_ax(gca);

    if cfg.save_analysis_fig
        fn = sprintf('RiskCoverage_%s', char(targetSplits(ss)));
        fn = regexprep(fn,'[^\w\d]+','_');
        exportgraphics(fig, fullfile(cfg.analysis_fig_dir,[fn '.png']), 'Resolution', 300);
    end
end
end

function [covGrid, riskCurve] = selective_risk_curve(score, risk, nPts)
score = double(score(:));
risk  = double(risk(:));
ok = isfinite(score) & isfinite(risk);
score = score(ok); risk = risk(ok);

N = numel(score);
[~, idx] = sort(score, 'ascend');
risk = risk(idx);

covGrid = linspace(max(1/N,1/nPts), 1, nPts)';
riskCurve = nan(nPts,1);
for i = 1:nPts
    k = max(1, round(covGrid(i) * N));
    riskCurve(i) = mean(risk(1:k));
end
end

function T = build_bootstrap_ci_table(Results, cfg)
if isempty(Results), T = table(); return; end

idxUse = arrayfun(@(r) abs(r.frac - cfg.bootstrap_use_frac) < 1e-12, Results);
R = Results(idxUse);
splits = string(cfg.bootstrap_split_list);

rows = {};
for ss = 1:numel(splits)
    idxS = strcmp(string({R.split}), splits(ss));
    Rs = R(idxS);
    if isempty(Rs), continue; end

    for mm = 1:numel(cfg.bootstrap_methods)
        key = cfg.bootstrap_methods{mm};

        acc = collect_field(Rs, [key '_acc']);
        hm  = collect_field(Rs, [key '_highMiss']);
        gp  = collect_field(Rs, [key '_goodput_norm']);

        [m1,l1,u1] = bootstrap_mean_ci(acc, cfg.bootstrap_B);
        [m2,l2,u2] = bootstrap_mean_ci(hm,  cfg.bootstrap_B);
        [m3,l3,u3] = bootstrap_mean_ci(gp,  cfg.bootstrap_B);

        rows(end+1,:) = {splits(ss), string(key), 'acc', m1, l1, u1}; %#ok<AGROW>
        rows(end+1,:) = {splits(ss), string(key), 'highMiss', m2, l2, u2}; %#ok<AGROW>
        rows(end+1,:) = {splits(ss), string(key), 'goodput_norm', m3, l3, u3}; %#ok<AGROW>
    end
end

if isempty(rows)
    T = table();
else
    T = cell2table(rows, 'VariableNames', {'split','method','metric','mean','ci_low','ci_high'});
end
end

function [mu, lo, hi] = bootstrap_mean_ci(x, B)
x = x(isfinite(x));
if isempty(x)
    mu = NaN; lo = NaN; hi = NaN; return;
end
mu = mean(x);
n = numel(x);
bs = nan(B,1);
for b = 1:B
    idx = randi(n, [n,1]);
    bs(b) = mean(x(idx));
end
lo = prctile(bs, 2.5);
hi = prctile(bs, 97.5);
end

function HighSNROut = run_highsnr_analysis_from_results(Results, cfg)
% ============================================================
% Build high-SNR conditional mechanism analysis from Results
%
% Required output fields:
%   HighSNROut.CorrTab
%   HighSNROut.SubsetSummary
%   HighSNROut.BinTab
%
% Save this file as:
%   run_highsnr_analysis_from_results.m
% ============================================================

    if nargin < 2, cfg = struct(); end
    if ~isfield(cfg, 'highsnr'), cfg.highsnr = struct(); end
    cfg.highsnr = fill_highsnr_defaults(cfg.highsnr);

    % --------------------------------------------------------
    % 1) Collect sample-level vectors from Results
    % --------------------------------------------------------
    [gap, varmi, ess, jbeta, mi_bar, beta, proxy, proxy_name] = ...
        collect_sample_level_fields(Results, cfg.highsnr);

    % Basic empty fallback
    if isempty(gap) || isempty(varmi) || isempty(ess) || isempty(jbeta)
        warning(['run_highsnr_analysis_from_results: insufficient sample-level fields found in Results. ' ...
                 'Returning empty tables.']);
        HighSNROut = struct();
        HighSNROut.CorrTab = table();
        HighSNROut.SubsetSummary = table();
        HighSNROut.BinTab = table();
        HighSNROut.ProxyName = "";
        HighSNROut.ProxyValue = [];
        return;
    end

    gap   = gap(:);
    varmi = varmi(:);
    ess   = ess(:);
    jbeta = jbeta(:);

    N = numel(gap);

    if isempty(mi_bar)
        mi_bar = nan(N,1);
    else
        mi_bar = mi_bar(:);
    end

    if isempty(beta)
        beta = nan(N,1);
    else
        beta = beta(:);
    end

    if isempty(proxy)
        proxy = mi_bar;
        proxy_name = "MIbar";
    else
        proxy = proxy(:);
    end

    % conservative score: larger means more conservative
    conserv = mi_bar - jbeta;

    valid = isfinite(gap) & isfinite(varmi) & isfinite(ess) & isfinite(jbeta) ...
          & isfinite(proxy);

    if any(isfinite(conserv))
        valid = valid & isfinite(conserv);
    end

    if nnz(valid) < 20
        warning('run_highsnr_analysis_from_results: too few valid samples. Returning empty tables.');
        HighSNROut = struct();
        HighSNROut.CorrTab = table();
        HighSNROut.SubsetSummary = table();
        HighSNROut.BinTab = table();
        HighSNROut.ProxyName = proxy_name;
        HighSNROut.ProxyValue = proxy;
        return;
    end

    q_hi = quantile(proxy(valid), cfg.highsnr.q_hi);
    q_lo = quantile(proxy(valid), cfg.highsnr.q_lo);

    idx_all = valid;
    idx_hi  = valid & (proxy >= q_hi);
    idx_lo  = valid & (proxy <= q_lo);

    % --------------------------------------------------------
    % 2) Correlation table
    % --------------------------------------------------------
    CorrTab = [
        subset_corr_table("ALL",  idx_all, gap, varmi, ess, conserv, beta)
        subset_corr_table("HIGH", idx_hi,  gap, varmi, ess, conserv, beta)
        subset_corr_table("LOW",  idx_lo,  gap, varmi, ess, conserv, beta)
    ];

    % --------------------------------------------------------
    % 3) Summary table
    % --------------------------------------------------------
    SubsetSummary = table( ...
        ["ALL"; "HIGH"; "LOW"], ...
        [nnz(idx_all); nnz(idx_hi); nnz(idx_lo)], ...
        [mean(proxy(idx_all),'omitnan'); mean(proxy(idx_hi),'omitnan'); mean(proxy(idx_lo),'omitnan')], ...
        [mean(gap(idx_all),'omitnan'); mean(gap(idx_hi),'omitnan'); mean(gap(idx_lo),'omitnan')], ...
        [mean(varmi(idx_all),'omitnan'); mean(varmi(idx_hi),'omitnan'); mean(varmi(idx_lo),'omitnan')], ...
        [mean(ess(idx_all),'omitnan'); mean(ess(idx_hi),'omitnan'); mean(ess(idx_lo),'omitnan')], ...
        [mean(conserv(idx_all),'omitnan'); mean(conserv(idx_hi),'omitnan'); mean(conserv(idx_lo),'omitnan')], ...
        [mean(beta(idx_all),'omitnan'); mean(beta(idx_hi),'omitnan'); mean(beta(idx_lo),'omitnan')], ...
        'VariableNames', {'Subset','N','ProxyMean','GapMean','VarMIMean','ESSMean','ConservMean','BetaMean'});

    % --------------------------------------------------------
    % 4) Gap-binning table
    % --------------------------------------------------------
    BinTab = [
        gap_binning_table("ALL",  idx_all, gap, varmi, ess, conserv, cfg.highsnr.nbins, cfg.highsnr.min_per_bin)
        gap_binning_table("HIGH", idx_hi,  gap, varmi, ess, conserv, cfg.highsnr.nbins, cfg.highsnr.min_per_bin)
        gap_binning_table("LOW",  idx_lo,  gap, varmi, ess, conserv, cfg.highsnr.nbins, cfg.highsnr.min_per_bin)
    ];

    % --------------------------------------------------------
    % 5) Output + optional plots
    % --------------------------------------------------------
    HighSNROut = struct();
    HighSNROut.CorrTab       = CorrTab;
    HighSNROut.SubsetSummary = SubsetSummary;
    HighSNROut.BinTab        = BinTab;
    HighSNROut.ProxyName     = proxy_name;
    HighSNROut.ProxyValue    = proxy;
    HighSNROut.q_hi          = q_hi;
    HighSNROut.q_lo          = q_lo;

    if cfg.highsnr.make_plots
        try
            outdir = fullfile(cfg.run_dir, 'highsnr_mechanism');
            if ~exist(outdir, 'dir'), mkdir(outdir); end
            make_highsnr_plots(outdir, gap, varmi, ess, conserv, idx_hi, idx_lo);
        catch ME
            warning('High-SNR plotting failed: %s', ME.message);
        end
    end
end

% ========================= Helpers =========================

function hs = fill_highsnr_defaults(hs)
    if ~isfield(hs,'q_hi'),        hs.q_hi = 0.70; end
    if ~isfield(hs,'q_lo'),        hs.q_lo = 0.30; end
    if ~isfield(hs,'nbins'),       hs.nbins = 6; end
    if ~isfield(hs,'min_per_bin'), hs.min_per_bin = 10; end
    if ~isfield(hs,'proxy_mode'),  hs.proxy_mode = 'auto'; end
    if ~isfield(hs,'make_plots'),  hs.make_plots = true; end
end

function [gap, varmi, ess, jbeta, mi_bar, beta, proxy, proxy_name] = ...
    collect_sample_level_fields(Results, hs_cfg)

    gap   = [];
    varmi = [];
    ess   = [];
    jbeta = [];
    mi_bar = [];
    beta  = [];
    proxy = [];
    proxy_name = "";

    % Flatten candidate structs from Results
    nodes = flatten_struct_nodes(Results);

    % Candidate names
    gap_names   = {'Gap_test','Gap','gap','gap_val','GapVal','gap_stat'};
    varmi_names = {'VarMI_test','VarMI','varmi','VarMI_val','varmi_val'};
    ess_names   = {'ESS_test','ESS','ess','ESS_val','ess_val'};
    jbeta_names = {'Jbeta_test','Jbeta','jbeta','J_beta','J_beta_test','phys_score_j'};
    mib_names   = {'MIbar_test','MIbar','mi_bar','MI_mean','MIbar_val','mi_mean'};
    beta_names  = {'beta_test','beta','beta_t','beta_val'};
    snr_names   = {'snr_db_test','snr_db','SNRdB','snrdb'};
    rho_names   = {'rho_test','rho','Rho'};
    mimat_names = {'MI_test','MI','mi_mat','mi_matrix','MI_mat'};

    % Search in order of preference
    gap   = pick_best_vector(nodes, gap_names);
    varmi = pick_best_vector(nodes, varmi_names);
    ess   = pick_best_vector(nodes, ess_names);
    jbeta = pick_best_vector(nodes, jbeta_names);
    mi_bar = pick_best_vector(nodes, mib_names);
    beta  = pick_best_vector(nodes, beta_names);

    % If MIbar missing, try derive from MI matrix
    if isempty(mi_bar)
        mi_mat = pick_best_matrix(nodes, mimat_names);
        if ~isempty(mi_mat)
            mi_bar = mean(mi_mat, 2, 'omitnan');
        end
    end

    % Proxy selection
    proxy_mode = lower(string(hs_cfg.proxy_mode));
    snr_db = pick_best_vector(nodes, snr_names);
    rho    = pick_best_vector(nodes, rho_names);

    switch proxy_mode
        case "snr_db"
            proxy = snr_db;
            proxy_name = "snr_db";
        case "rho"
            proxy = 10*log10(max(rho, eps));
            proxy_name = "10log10(rho)";
        case "mi_bar"
            proxy = mi_bar;
            proxy_name = "MIbar";
        otherwise
            if ~isempty(snr_db)
                proxy = snr_db;
                proxy_name = "snr_db";
            elseif ~isempty(rho)
                proxy = 10*log10(max(rho, eps));
                proxy_name = "10log10(rho)";
            else
                proxy = mi_bar;
                proxy_name = "MIbar";
            end
    end

    % Align lengths if possible
    lens = [numel(gap), numel(varmi), numel(ess), numel(jbeta), numel(mi_bar), numel(beta), numel(proxy)];
    lens = lens(lens > 0);
    if isempty(lens), return; end
    N = mode(lens);

    gap   = trim_or_empty(gap, N);
    varmi = trim_or_empty(varmi, N);
    ess   = trim_or_empty(ess, N);
    jbeta = trim_or_empty(jbeta, N);
    mi_bar = trim_or_empty(mi_bar, N);
    beta  = trim_or_empty(beta, N);
    proxy = trim_or_empty(proxy, N);
end

function nodes = flatten_struct_nodes(S)
    nodes = {};
    if isstruct(S)
        nodes{end+1} = S; %#ok<AGROW>
        fns = fieldnames(S);
        for i = 1:numel(fns)
            val = S.(fns{i});
            if isstruct(val)
                sub = flatten_struct_nodes(val);
                nodes = [nodes, sub]; %#ok<AGROW>
            elseif iscell(val)
                for j = 1:numel(val)
                    if isstruct(val{j})
                        sub = flatten_struct_nodes(val{j});
                        nodes = [nodes, sub]; %#ok<AGROW>
                    end
                end
            end
        end
    end
end

function x = pick_best_vector(nodes, candidate_names)
    x = [];
    bestN = -1;
    for i = 1:numel(nodes)
        S = nodes{i};
        for j = 1:numel(candidate_names)
            nm = candidate_names{j};
            if isfield(S, nm)
                v = S.(nm);
                if isnumeric(v) && isvector(v) && numel(v) > bestN
                    x = v(:);
                    bestN = numel(v);
                end
            end
        end
    end
end

function x = pick_best_matrix(nodes, candidate_names)
    x = [];
    bestN = -1;
    for i = 1:numel(nodes)
        S = nodes{i};
        for j = 1:numel(candidate_names)
            nm = candidate_names{j};
            if isfield(S, nm)
                v = S.(nm);
                if isnumeric(v) && ismatrix(v) && size(v,1) > 1 && size(v,2) > 1
                    if numel(v) > bestN
                        x = v;
                        bestN = numel(v);
                    end
                end
            end
        end
    end
end

function x = trim_or_empty(x, N)
    if isempty(x)
        x = nan(N,1);
        return;
    end
    x = x(:);
    if numel(x) >= N
        x = x(1:N);
    else
        y = nan(N,1);
        y(1:numel(x)) = x;
        x = y;
    end
end

function T = subset_corr_table(subset_name, idx, gap, varmi, ess, conserv, beta)
    metrics = {
        'corr_gap_varmi'
        'corr_gap_ess'
        'corr_gap_conserv'
        'corr_varmi_ess'
        'corr_varmi_conserv'
        'corr_ess_conserv'
        'corr_beta_conserv'
        };

    [r1,p1] = safe_corr(gap(idx),   varmi(idx));
    [r2,p2] = safe_corr(gap(idx),   ess(idx));
    [r3,p3] = safe_corr(gap(idx),   conserv(idx));
    [r4,p4] = safe_corr(varmi(idx), ess(idx));
    [r5,p5] = safe_corr(varmi(idx), conserv(idx));
    [r6,p6] = safe_corr(ess(idx),   conserv(idx));
    [r7,p7] = safe_corr(beta(idx),  conserv(idx));

    T = table( ...
        repmat(string(subset_name), numel(metrics), 1), ...
        string(metrics), ...
        [r1;r2;r3;r4;r5;r6;r7], ...
        [p1;p2;p3;p4;p5;p6;p7], ...
        'VariableNames', {'Subset','Metric','SpearmanR','PValue'});
end

function [r,p] = safe_corr(x,y)
    mask = isfinite(x) & isfinite(y);
    x = x(mask); y = y(mask);
    if numel(x) < 5 || numel(unique(x)) < 2 || numel(unique(y)) < 2
        r = NaN; p = NaN;
        return;
    end
    [r,p] = corr(x, y, 'Type', 'Spearman', 'Rows', 'complete');
end

function T = gap_binning_table(subset_name, idx, gap, varmi, ess, conserv, nbins, min_per_bin)
    x  = gap(idx);
    y1 = varmi(idx);
    y2 = ess(idx);
    y3 = conserv(idx);

    mask = isfinite(x) & isfinite(y1) & isfinite(y2) & isfinite(y3);
    x = x(mask); y1 = y1(mask); y2 = y2(mask); y3 = y3(mask);

    if numel(x) < max(20, nbins * min_per_bin)
        T = table();
        return;
    end

    edges = quantile(x, linspace(0,1,nbins+1));
    edges(1)   = edges(1) - 1e-12;
    edges(end) = edges(end) + 1e-12;
    bid = discretize(x, edges);

    rows = {};
    for b = 1:nbins
        ib = (bid == b);
        if nnz(ib) < min_per_bin, continue; end
        rows(end+1,:) = { ...
            string(subset_name), ...
            b, ...
            nnz(ib), ...
            mean(x(ib),  'omitnan'), ...
            mean(y1(ib), 'omitnan'), ...
            mean(y2(ib), 'omitnan'), ...
            mean(y3(ib), 'omitnan')}; %#ok<AGROW>
    end

    if isempty(rows)
        T = table();
        return;
    end

    T = cell2table(rows, 'VariableNames', ...
        {'Subset','Bin','N','GapMean','VarMIMean','ESSMean','ConservMean'});
end

function make_highsnr_plots(outdir, gap, varmi, ess, conserv, idx_hi, idx_lo)
    make_scatter(fullfile(outdir,'gap_vs_varmi.png'), ...
        gap, varmi, idx_hi, idx_lo, 'Gap', 'VarMI');

    make_scatter(fullfile(outdir,'gap_vs_ess.png'), ...
        gap, ess, idx_hi, idx_lo, 'Gap', 'ESS');

    make_scatter(fullfile(outdir,'gap_vs_conserv.png'), ...
        gap, conserv, idx_hi, idx_lo, 'Gap', 'Conservativeness = MIbar - Jbeta');
end

function make_scatter(savepath, x, y, idx_hi, idx_lo, xlab, ylab)
    fig = figure('Visible','off');
    hold on; grid on; box on;

    xl = x(idx_lo); yl = y(idx_lo);
    ml = isfinite(xl) & isfinite(yl);
    scatter(xl(ml), yl(ml), 18, 'filled', 'MarkerFaceAlpha', 0.25, ...
        'DisplayName', 'LOW');

    xh = x(idx_hi); yh = y(idx_hi);
    mh = isfinite(xh) & isfinite(yh);
    scatter(xh(mh), yh(mh), 18, 'filled', 'MarkerFaceAlpha', 0.25, ...
        'DisplayName', 'HIGH');

    if nnz(ml) >= 5
        pl = polyfit(xl(ml), yl(ml), 1);
        xx = linspace(min(xl(ml)), max(xl(ml)), 100);
        plot(xx, polyval(pl, xx), 'LineWidth', 1.5, 'HandleVisibility','off');
    end
    if nnz(mh) >= 5
        ph = polyfit(xh(mh), yh(mh), 1);
        xx = linspace(min(xh(mh)), max(xh(mh)), 100);
        plot(xx, polyval(ph, xx), '--', 'LineWidth', 1.5, 'HandleVisibility','off');
    end

    xlabel(xlab, 'Interpreter','none');
    ylabel(ylab, 'Interpreter','none');
    legend('Location','best');
    title(sprintf('%s vs %s', ylab, xlab), 'Interpreter','none');

    exportgraphics(fig, savepath, 'Resolution', 200);
    close(fig);
end

function cmap = get_ablation_colors(labels)
if isstring(labels), labels = cellstr(labels); end
n = numel(labels);
cmap = zeros(n,3);

for i = 1:n
    name = upper(strtrim(labels{i}));
    switch name
        case 'FULL'
            cmap(i,:) = [0.16 0.39 0.77];   % 深蓝
        case 'DATA_ONLY'
            cmap(i,:) = [0.15 0.62 0.56];   % 青绿
        case 'PHYS_ONLY'
            cmap(i,:) = [0.90 0.55 0.16];   % 橙色
        case 'NO_B'
            cmap(i,:) = [0.84 0.37 0.20];   % 红橙
        case 'NO_PRIOR_ALL'
            cmap(i,:) = [0.49 0.36 0.80];   % 紫色
        case 'NO_DEEPFADE'
            cmap(i,:) = [0.35 0.68 0.38];   % 绿色
        case 'NO_GATE'
            cmap(i,:) = [0.55 0.55 0.55];   % 灰
        case 'NO_RISK'
            cmap(i,:) = [0.72 0.72 0.72];   % 浅灰
        otherwise
            cmap(i,:) = [0.45 0.60 0.78];   % 默认蓝灰
    end
end
end

function MarginOut = run_margin_stability_analysis(Results, cfg)

targetSplits = string(cfg.margin_split_list);
epsList = double(cfg.margin_eps_list(:)');
nbins   = double(cfg.margin_nbins);
nMC     = double(cfg.margin_mc);

rows = {};
MarginOut = struct();
MarginOut.SummaryTab = table();

for ss = 1:numel(targetSplits)
    hit = find_run(Results, targetSplits(ss), cfg.margin_use_frac, cfg.margin_use_seed);
    if isempty(hit), continue; end

    R = Results(hit);

    reqFields = {'MI_val','OURSFUSION_beta_val','OURSFUSION_w_val','OURSFUSION_b_val','OURSFUSION_theta'};
    okFields = all(isfield(R, reqFields)) && isfield(R, 'Det_OURSFUSION');
    if ~okFields, continue; end

    MI    = double(R.MI_val);                        % [K,N]
    z0    = double(R.Det_OURSFUSION.z(:));          % [N,1]
    beta  = double(R.OURSFUSION_beta_val(:));       % [N,1]
    w     = double(R.OURSFUSION_w_val(:));          % [N,1]
    b     = double(R.OURSFUSION_b_val(:));          % [N,1]
    theta = double(R.OURSFUSION_theta(:));          % [M-1,1]

    N = numel(z0);
    if isempty(MI) || isempty(theta) || size(MI,2) ~= N, continue; end

    % physical decision induced by z and theta
    m0 = ordinal_decision_from_z(z0, theta);

    % margin to nearest threshold
    margin = compute_margin_to_theta(z0, theta);

    % bins for Figure 1
    if cfg.margin_use_quantile_bins
        edges = quantile(margin, linspace(0,1,nbins+1));
        edges(1) = edges(1) - 1e-12;
        edges(end) = edges(end) + 1e-12;
    else
        edges = linspace(min(margin), max(margin), nbins+1);
        edges(1) = edges(1) - 1e-12;
        edges(end) = edges(end) + 1e-12;
    end
    binCenter = 0.5*(edges(1:end-1) + edges(2:end));

    flipCurve = nan(nbins, numel(epsList));
    certifiedFrac = nan(numel(epsList),1);
    certifiedFlip = nan(numel(epsList),1);
    overallFlip   = nan(numel(epsList),1);

    for ee = 1:numel(epsList)
        epsNow = epsList(ee);

        % Monte Carlo estimate of flip probability under ||ΔMI||_inf <= epsNow
        flipCount = zeros(N,1);

        for mc = 1:nMC
            Delta = epsNow * (2*rand(size(MI)) - 1);   % each subcarrier perturbation in [-eps, eps]
            MIp = MI + Delta;

            z1 = w .* mi_eesm_samplewise(MIp, beta, cfg.eps2) + b;
            m1 = ordinal_decision_from_z(z1, theta);

            flipCount = flipCount + double(m1 ~= m0);
        end

        flipProb = flipCount / nMC;
        overallFlip(ee) = mean(flipProb, 'omitnan');

        % certified condition: |Δz| <= w_t * epsNow < margin_t
        certMask = (w .* epsNow) < margin;
        certifiedFrac(ee) = mean(certMask, 'omitnan');

        if any(certMask)
            certifiedFlip(ee) = mean(flipProb(certMask), 'omitnan');
        else
            certifiedFlip(ee) = NaN;
        end

        % bin-wise flip rate
        for bb = 1:nbins
            inb = margin >= edges(bb) & margin < edges(bb+1);
            if bb == nbins
                inb = margin >= edges(bb) & margin <= edges(bb+1);
            end
            if any(inb)
                flipCurve(bb,ee) = mean(flipProb(inb), 'omitnan');
            end
        end

        rows(end+1,:) = {char(targetSplits(ss)), epsNow, N, ...
                         mean(margin,'omitnan'), ...
                         certifiedFrac(ee), certifiedFlip(ee), overallFlip(ee)}; %#ok<AGROW>
    end

    % -------- Figure 1: margin-binned flip rate --------
    fig1 = figure('Color','w','Position',[100 100 700 520]); hold on;
    for ee = 1:numel(epsList)
        plot(binCenter, flipCurve(:,ee), '-o', 'LineWidth', 1.8, 'MarkerSize', 6, ...
             'DisplayName', sprintf('\\epsilon = %.3f', epsList(ee)));
    end
    xlabel('Margin bin center');
    ylabel('Decision flip rate');
    title(sprintf('Margin-binned flip rate | split=%s', char(targetSplits(ss))));
    legend('Location','northeast');
    grid on; beautify_ax(gca);

    if cfg.save_analysis_fig
        fn = sprintf('margin_binned_flip_%s', char(targetSplits(ss)));
        fn = regexprep(fn, '[^\w\d]+', '_');
        exportgraphics(fig1, fullfile(cfg.analysis_fig_dir, [fn '.png']), 'Resolution', 300);
    end

    % -------- Figure 2: certified coverage + certified flip --------
    fig2 = figure('Color','w','Position',[120 120 700 520]);
    yyaxis left;
    plot(epsList, certifiedFrac, '-o', 'LineWidth', 2.0, 'MarkerSize', 7);
    ylabel('Certified stable fraction');

    yyaxis right;
    plot(epsList, certifiedFlip, '-s', 'LineWidth', 2.0, 'MarkerSize', 7);
    ylabel('Actual flip rate on certified subset');

    xlabel('\epsilon');
    title(sprintf('Certified stability coverage | split=%s', char(targetSplits(ss))));
    grid on; beautify_ax(gca);

    if cfg.save_analysis_fig
        fn = sprintf('certified_coverage_%s', char(targetSplits(ss)));
        fn = regexprep(fn, '[^\w\d]+', '_');
        exportgraphics(fig2, fullfile(cfg.analysis_fig_dir, [fn '.png']), 'Resolution', 300);
    end
end

if ~isempty(rows)
    MarginOut.SummaryTab = cell2table(rows, ...
        'VariableNames', {'split','eps','N','margin_mean', ...
                          'certified_frac','certified_flip_rate','overall_flip_rate'});
else
    MarginOut.SummaryTab = table();
end
end

function z = mi_eesm_samplewise(MI, beta, eps2)
% MI: [K,N], beta: [N,1]
beta = double(beta(:))';  % [1,N]
tmp = exp(-MI .* beta);
z = -(1 ./ beta(:)) .* log(mean(tmp,1)' + eps2);
end

function m = ordinal_decision_from_z(z, theta)
% theta length = M-1
z = double(z(:));
theta = double(theta(:));
N = numel(z);
M = numel(theta) + 1;
m = zeros(N,1);

for i = 1:N
    zi = z(i);
    k = sum(zi >= theta);
    m(i) = min(max(k, 0), M-1);
end
end

function margin = compute_margin_to_theta(z, theta)
z = double(z(:));
theta = double(theta(:))';
D = abs(z - theta);
margin = min(D, [], 2);
end