% INVERSE_Variational_XTFC.m
% Variational-XTFC inverse problem (Gaussian RBF), scanning ONLY nu and auto-tuning eta.
% Boundary conditions are enforced EXACTLY by XTFC (no BC penalty block).
clc; clear; close all; rng(0);

%% -------------------- Problem + synthetic data --------------------
nu_true = 0.5;
xL = 0; xR = 1;
exact = @(x,nu) (exp(x./nu)-1) ./ (exp(1./nu)-1);

N_data    = 50;
p_data    = 3;                               % right-clustering exponent for data
x_data    = right_clustered_linspace(xL, xR, N_data, p_data);
noise_std = 1e-2;                            % known instrument noise
u_data    = exact(x_data, nu_true) + noise_std*randn(N_data,1);

% Boundary values and constrained (bridge) function g
BL = 0; BR = 1;                              % u(0)=BL, u(1)=BR
gfun = @(x) (1 - x).*BL + x.*BR;             % linear constrained part

%% -------------------- PDE collocation (no 'w', cluster near x=1) --------------------
Nc_total = 200;                               % total PDE collocation points
p_pde    = 3.5;                               % right-clustering exponent for PDE residual
X_pde    = right_clustered_linspace(xL, xR, Nc_total, p_pde);

%% -------------------- Gaussian RBF features (fixed) --------------------
Ns         = 100;                              % number of features (tune)
alpha_star = linspace(xL, xR, Ns)';           % centers in x
dx   = max(1e-12, alpha_star(2) - alpha_star(1));
sig_x = 5*dx * ones(size(alpha_star));        % widths (tune for sharp layers)

% Parameterization phi(z)=exp(-z^2) with z = m*x + b
m = 1./(sqrt(2)*sig_x);
b = -m.*alpha_star;                           % so z = m(x - center)

% XTFC constants: phi at boundaries x=0 and x=1
Ai = exp( - (b          ).^2 );               % Ns x 1 (phi at x=0)
Bi = exp( - (m + b      ).^2 );               % Ns x 1 (phi at x=1)

% XTFC basis rows: Psi(x) = phi(x) - (1-x)*phi(0) - x*phi(1)
psi_rows = @(x) exp( - (m'.*x + b').^2 ) ...
                - (1 - x)*Ai' - x*Bi';        % returns 1 x Ns for scalar x

% Residual rows: r_i(x) = d/dx Psi_i - nu * d2/dx2 Psi_i
% For Gaussian phi(z)=exp(-z^2): with z = m x + b
% d/dx phi = -2*m*z*phi,   d2/dx2 phi = m^2*(4*z^2 - 2)*phi
pde_rows_xtfc = @(x,nu) ...
    (Ai' - Bi') + ...
    exp( - (m'.*x + b').^2 ) .* ( ...
       -2*(m'.*(m'.*x + b')) ...
       - nu*(m'.^2).*(4*(m'.*x + b').^2 - 2) );

%% -------------------- Precisions (whitening) --------------------
beta_data = 1 / (noise_std^2 + 1e-12);        % data precision
beta_pde  = 1e2;                               % PDE residual precision (tune)

%% -------------------- Grids for nu (ONLY nu) --------------------
nu_grid = logspace(-3, 1, 500);               % 1e-3 .. 1

% Storage
loge_vec = -inf(numel(nu_grid),1);            % log-evidence over nu
eta_vec  = nan(numel(nu_grid),1);

% Auto-eta params
eta_bounds = [1e-12, 1e-2];
eta0       = 1e-7;

%% -------------------- Scan nu and auto-tune eta --------------------
tic;
for i = 1:numel(nu_grid)
    nu = nu_grid(i);
    [Phi, y] = build_whitened_xtfc(nu, ...
        x_data, u_data, X_pde, ...
        psi_rows, pde_rows_xtfc, ...
        beta_data, beta_pde, BL, BR, gfun);

    [eta_hat, ~, ~, loge] = auto_eta_for_Phi(Phi, y, eta0, eta_bounds);
    loge_vec(i) = loge;
    eta_vec(i)  = eta_hat;
end

% Pick best nu
[loge_best, i_best] = max(loge_vec);
nu_hat  = nu_grid(i_best);
eta_hat = eta_vec(i_best);

fprintf('Estimated nu_hat = %.6f (true %.6f)\n', nu_hat, nu_true);
fprintf('Selected eta_hat = %.3e\n', eta_hat);
toc;
%% -------------------- Posterior at (nu_hat, eta_hat) --------------------
[Phi_hat, y_hat] = build_whitened_xtfc(nu_hat, ...
    x_data, u_data, X_pde, ...
    psi_rows, pde_rows_xtfc, ...
    beta_data, beta_pde, BL, BR, gfun);

M  = size(Phi_hat,2);
A  = eta_hat*eye(M) + (Phi_hat.'*Phi_hat);
A  = 0.5*(A + A.');                           % symmetrize
[R, ok, ~] = chol_spd_with_jitter(A);
if ~ok, error('Cholesky failed even after jitter.'); end
rhs = (Phi_hat.'*y_hat);
mN  = R \ (R.' \ rhs);                        % posterior mean for coefficients

%% -------------------- Predictions and band --------------------
N_pred  = 500;
x_pred  = linspace(xL, xR, N_pred).';

H_pred  = build_rows_block(x_pred, psi_rows); % Psi rows
g_pred  = gfun(x_pred);
u_pred_mean = g_pred + H_pred * mN;

std_pred = zeros(N_pred,1);
for k = 1:N_pred
    h  = H_pred(k,:).';
    y1 = R.' \ h;            % solve R^T y1 = h
    z1 = R   \ y1;           % solve R z1  = y1
    std_pred(k) = sqrt(max(0, h.'*z1));  % sqrt(h^T A^{-1} h)
end

u_exact_true = exact(x_pred, nu_true);
u_exact_hat  = exact(x_pred, nu_hat);

%% -------------------- Plots --------------------
plot_evidence1d_neurips(nu_grid, loge_vec, nu_hat, 'Evidence_nu_XTFC');
plot_solution_neurips( ...
    x_pred, u_pred_mean, std_pred, ...
    x_data, u_data, ...
    u_exact_true, u_exact_hat, ...
    nu_true, nu_hat, ...
    'Inverse_Variational_XTFC');

%% -------------------- Local helper functions --------------------
function [Phi, y] = build_whitened_xtfc( ...
        nu, x_data, u_data, X_pde, ...
        psi_rows, pde_rows_xtfc, ...
        beta_data, beta_pde, BL, BR, gfun)

    % ----- Data block (XTFC): Psi*c ≈ u_data - g(x_data) -----
    H_data = build_rows_block(x_data, psi_rows);         % N_data x Ns
    y_data = u_data - gfun(x_data);                      % shift by constrained part

    % ----- PDE residual block (variational): R*c ≈ -f_g -----
    LHS_PDE = build_rows_block(X_pde, @(x) pde_rows_xtfc(x,nu));  % N_pde x Ns
    f_g     = (BR - BL);                                 % here = 1
    y_pde   = - f_g * ones(size(LHS_PDE,1),1);

    % ----- Whitening (row scaling) -----
    Phi = [ sqrt(beta_data)*H_data;
            sqrt(beta_pde )*LHS_PDE ];
    y   = [ sqrt(beta_data)*y_data;
            sqrt(beta_pde )*y_pde   ];
end

function x = right_clustered_linspace(xL,xR,N,p)
    % Power-law clustering near xR; p>1 increases clustering strength.
    t = linspace(0,1,N).';
    x = xL + (xR - xL) * (1 - (1 - t).^p);
end

function H = build_rows_block(xvec, rowfun)
    n = numel(xvec);
    r = rowfun(xvec(1));
    H = zeros(n, numel(r));
    H(1,:) = r;
    for k = 2:n
        H(k,:) = rowfun(xvec(k));
    end
end

function [eta_hat, mN, R, loge] = auto_eta_for_Phi(Phi, y, eta0, bounds)
    % Empirical Bayes (evidence maximization) with damping + small grid refine
    eta_min = bounds(1); eta_max = bounds(2);
    eta = min(max(eta0, eta_min), eta_max);

    [~,S,~] = svd(Phi, 'econ'); s2 = diag(S).^2;
    M = size(Phi,2); N = size(Phi,1); 

    maxit = 50; tol = 1e-6; damp = 0.3; ok_last = true;

    for it = 1:maxit
        A = eta*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
        [R, ok, ~] = chol_spd_with_jitter(A);
        if ~ok
            eta = min(10*max(eta, 1e-12), eta_max);
            ok_last = false; continue;
        end
        ok_last = true;

        rhs = Phi.'*y; mN = R \ (R.' \ rhs);
        gamma   = sum( s2 ./ (s2 + eta) );
        eta_new = gamma / max(mN.'*mN, 1e-30);
        eta_new = min(max(eta_new, eta_min), eta_max);
        eta_upd = (1-damp)*eta + damp*eta_new;

        if abs(eta_upd - eta) <= tol*(eta + 1e-12)
            eta = eta_upd; break;
        end
        eta = eta_upd;
    end

    if ~ok_last
        [eta, mN, R, ~] = refine_eta_grid(Phi, y, min(max(eta, 1e-9), 1e-3), 10, 9);
    end

    % Final evidence at eta
    A = eta*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
    [R, ok, ~] = chol_spd_with_jitter(A);
    if ~ok
        eta = min(max(eta*10, eta_min), eta_max);
        A = eta*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
        [R, ok, ~] = chol_spd_with_jitter(A);
        if ~ok, error('Failed to stabilize A for evidence.'); end
    end
    rhs = Phi.'*y; mN = R \ (R.' \ rhs);
    r   = y - Phi*mN;
    E   = 0.5*( r.'*r + eta*(mN.'*mN) );
    logdetA = 2*sum(log(diag(R)));
    loge = (M/2)*log(eta) - E - 0.5*logdetA - (N/2)*log(2*pi);

    eta_hat = eta;
end

function [eta_best, mN_best, R_best, loge_best] = refine_eta_grid(Phi, y, eta_center, span, npts)
    M = size(Phi,2);
    grid = logspace(log10(eta_center/span), log10(eta_center*span), npts);
    loge_best = -inf; eta_best = grid(1);
    mN_best = []; R_best = [];
    for e = grid
        A = e*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
        [R, ok, ~] = chol_spd_with_jitter(A);
        if ~ok, continue; end
        rhs = Phi.'*y; mN = R \ (R.' \ rhs);
        r = y - Phi*mN;
        E = 0.5*( r.'*r + e*(mN.'*mN) );
        logdetA = 2*sum(log(diag(R)));
        loge = (M/2)*log(e) - E - 0.5*logdetA - (N/2)*log(2*pi);
        if loge > loge_best
            loge_best = loge; eta_best = e; mN_best = mN; R_best = R;
        end
    end
end

function [R, ok, ridge] = chol_spd_with_jitter(A)
    [R,p] = chol(A);
    if p==0, ok=true; ridge=0; return; end
    ok=false;
    ridge = max(1e-12, eps(norm(A,1)));
    I = eye(size(A));
    for k = 1:8
        [R,p] = chol(A + ridge*I);
        if p==0, ok=true; return; end
        ridge = ridge*10;
    end
    R = [];
end

%% -------------------- Publication-quality plotting --------------------
function plot_evidence1d_neurips(nu_grid, loge_vec, nu_hat, save_prefix)
    S = neurips_style();
    fig = figure('Color','w'); hold on; box on; grid on;

    x = log10(nu_grid);
    plot(x, loge_vec, 'LineWidth', S.LineWidth);
    xhat = log10(nu_hat);
    plot(xhat, interp1(x,loge_vec,xhat,'linear','extrap'), 'ro', 'MarkerSize', S.MarkerSize, 'LineWidth', S.LineWidth);

    xlabel('$\log_{10}\,\nu$', 'FontSize', S.LabelFontSize);
    ylabel('log-evidence',      'FontSize', S.LabelFontSize);
    % title('log-evidence over $\nu$', 'FontSize', S.TitleFontSize);
    set(gca,'FontSize',S.AxisFontSize,'LineWidth',S.AxesLineWidth,'XMinorGrid','off','YMinorGrid','off');

    % save_figure(fig, [save_prefix '.png'], S.FigW*1.1, S.FigH*1.0, S.DPI);
    save_figure(fig, [save_prefix '.pdf'], S.FigW*1.1, S.FigH*1.0, S.DPI);
end

function plot_solution_neurips(x_pred, u_mean, std_model, ...
                               x_data, u_data, ...
                               u_exact_true, ~, ...
                               nu_true, nu_hat, save_prefix)
    S = neurips_style();
    fig = figure('Color','w'); hold on;

    % 95% model band
    xb = [x_pred; flipud(x_pred)];
    yb = [u_mean + 2*std_model; flipud(u_mean - 2*std_model)];
    hBand = patch('XData', xb, 'YData', yb, ...
                  'FaceColor', [0.85 0.88 1.00], 'EdgeColor', 'none');

    hMean = plot(x_pred, u_mean, 'r-', 'LineWidth', S.LineWidth+0.5);
    hTrue = plot(x_pred, u_exact_true, 'k--', 'LineWidth', S.LineWidth);
    %hHat  = plot(x_pred, u_exact_hat,  'b-.', 'LineWidth', S.LineWidth-0.2);
    hData = scatter(x_data, u_data, S.ScatterSize, 'filled','MarkerFaceColor', 'b');

    xlabel('$x$', 'FontSize', S.LabelFontSize);
    ylabel('$u(x)$', 'FontSize', S.LabelFontSize);
    title(sprintf('XTFC Prediction: $\\hat{\\nu}=%.3g$', nu_hat), ...
          'FontSize', S.TitleFontSize);
    grid on; box on;
    set(gca, 'FontSize', S.AxisFontSize, 'LineWidth', S.AxesLineWidth, ...
             'XMinorGrid','on','YMinorGrid','on');

    lgd = legend([hBand hMean hTrue hData], ...
        { '$95\%$ band (model)', ...
          sprintf('Predicted mean ($\\hat{\\nu}$ = %.3g)', nu_hat), ...
          sprintf('Exact (true $\\nu$ = %.3g)', nu_true), ...
          'Data' }, ...
        'FontSize', S.LegendFontSize, 'Location','best');
    set(lgd,'Interpreter','latex');

    % save_figure(fig, [save_prefix '.png'], S.FigW, S.FigH, S.DPI);
    save_figure(fig, [save_prefix '.pdf'], S.FigW, S.FigH, S.DPI);
end

function S = neurips_style()
    set(groot,'defaultTextInterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');

    S.FigW = 2*3.5;   % inches (single-column)
    S.FigH = 2*2.6;

    S.AxisFontSize   = 12;
    S.LabelFontSize  = 14;
    S.TitleFontSize  = 14;
    S.LegendFontSize = 12;
    S.LineWidth      = 1.8;
    S.AxesLineWidth  = 1.0;
    S.MarkerSize     = 7;
    S.ScatterSize    = 24;

    S.DPI = 300;
end

function save_figure(fig, filename, W, H, dpi)
    set(fig, 'Units','inches'); pos = get(fig,'Position'); pos(3)=W; pos(4)=H; set(fig,'Position',pos);
    set(fig, 'PaperUnits','inches','PaperPosition',[0 0 W H], 'PaperSize',[W H]);
    try
        exportgraphics(fig, filename, 'Resolution', dpi);
    catch
        [~,~,ext] = fileparts(filename);
        if strcmpi(ext,'.pdf')
            print(fig, filename, '-dpdf', sprintf('-r%d',dpi));
        else
            print(fig, filename, '-dpng', sprintf('-r%d',dpi));
        end
    end
end
