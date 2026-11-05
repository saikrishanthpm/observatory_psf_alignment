function fig = make_psf_alignment_figure(datafile, opts)
%MAKE_PSF_ALIGNMENT_FIGURE  PSF-based alignment figure: Nautilus -> DE.
%
%   make_psf_alignment_figure('tma_figdata.mat')
%   make_psf_alignment_figure('hubbletilus_figdata.mat', struct('xscale','linear'))
%
%   The .mat file is written by export_psf_figure_data.py (run it at the end
%   of hubbletilus.ipynb or tma_moredof.ipynb; the cells to paste are in that
%   file's docstring). MATLAB cannot run optiland, so every PSF, OPD and WFE
%   value is computed in Python and this script only draws.
%
%   Layout (reads left to right, top to bottom):
%     (a) PSF strip: initial -> after Nautilus -> DE snapshots A-D -> final,
%         one shared log intensity scale and colour bar, lambda/D scale bar,
%         RMS WFE (waves) and DE generation under every panel.
%     (b) RMS WFE vs DE generation (log y), snapshots A-D marked on the
%         curve, Marechal lambda/14 diffraction-limit line.
%     (c) Every alignment parameter vs generation (small multiples) with the
%         injected truth, so parameter convergence can be compared with WFE
%         convergence without a second y-axis.
%
%   How the notes on the original figure are implemented
%     1. Plot WFE, not one parameter      -> panel (b); parameters move to (c)
%     2. No crossing teal leader lines    -> letters A-D on the curve, faint
%                                            vertical lines at those generations
%     3. Left-to-right flow               -> single strip with stage headers
%     4. PSF display                      -> log10 stretch, shared limits,
%                                            one colour bar, lambda/D bar,
%                                            intensity relative to ideal peak
%                                            (so each panel's peak = Strehl)
%     5. Units and definitions            -> "waves RMS", Nautilus / DE named,
%                                            truth labelled, WFE for every panel
%     6. Consistent typography            -> one font, one size set, vector PDF
%     7. Early action compressed          -> log generation axis (opts.xscale)
%
%   Options (fields of OPTS, all optional)
%     out        output base name (default: datafile name + '_figure')
%     xscale     'log' (default) or 'linear' generation axis
%     width_in   figure width in inches (default 7.0, two-column page)
%     font       font name (default 'Helvetica')
%     fs         base font size in pt (default 8)
%     log_floor  lowest intensity shown, relative to ideal peak (default 1e-4)
%     show_opd   add a row of residual OPD maps (default false)
%     show_params  draw panel (c) (default true)
%     marechal   draw lambda/14 line (default true)
%     show_ideal add the aligned-system PSF as a reference panel
%                (default: only when there are fewer than 2 snapshots)
%     max_psf_in largest PSF panel size in inches (default 1.3)
%
%   Runs without a parameter log (e.g. tma_moredof2.ipynb, where DE ran with
%   disp=True and no callback) export has_trace = 0. The strip then shows
%   initial / Nautilus / final (+ aligned reference), panel (b) plots the DE
%   cost printed by disp=True, and panel (c) compares each parameter's error
%   after Nautilus and after DE on one scale (|error| / prior half-width).
%     visible    show the figure window (default true)
%     save       write PDF + PNG (default true)

if nargin < 1 || isempty(datafile)
    datafile = 'tma_figdata.mat';
end
if nargin < 2
    opts = struct();
end
[fdir, fname] = fileparts(datafile);
defaults = struct( ...
    'out', fullfile(fdir, [fname '_figure']), ...
    'xscale', 'log', ...
    'width_in', 7.0, ...
    'font', 'Helvetica', ...
    'fs', 8, ...
    'log_floor', 1e-4, ...
    'show_opd', false, ...
    'show_params', true, ...
    'marechal', true, ...
    'show_ideal', [], ...
    'max_psf_in', 1.3, ...
    'visible', true, ...
    'save', true);
opts = merge_opts(defaults, opts);

S = load(datafile);
req = {'gen', 'theta_true', 'theta_ns', 'theta_final', 'param_names', ...
       'param_units', 'psf_initial', 'psf_ns', 'psf_final', 'psf_ideal', ...
       'wfe_initial', 'wfe_ns', 'wfe_final'};
for k = 1:numel(req)
    if ~isfield(S, req{k})
        error('make_psf_alignment_figure:missing', ...
              '%s is missing "%s". Re-export with export_psf_figure_data.py.', ...
              datafile, req{k});
    end
end

% ------------------------------------------------------------------ data
gen    = S.gen(:);
truth  = S.theta_true(:).';
names  = cellstr_row(S.param_names);
units  = cellstr_row(S.param_units);
P      = numel(truth);
nGen   = numel(gen);
hasTr  = isfield(S, 'has_trace') && S.has_trace > 0 || ...
         (~isfield(S, 'has_trace') && isfield(S, 'wfe_trace') && ~isempty(S.wfe_trace));
if hasTr
    wfe   = S.wfe_trace(:);
    theta = S.theta_trace;
    snapG = S.snap_gen(:).';
else
    wfe   = [];  theta = [];  snapG = [];
end
cost   = [];
if isfield(S, 'objective'), cost = S.objective(:); end
K      = numel(snapG);
stage1 = get_str(S, 'stage1', 'Nautilus');
stage2 = get_str(S, 'stage2', 'Differential evolution');
noisy  = isfield(S, 'noise_sigma') && S.noise_sigma > 0;
if isempty(opts.show_ideal), opts.show_ideal = K < 2; end

psfs  = cat(3, S.psf_initial, S.psf_ns);
wfes  = [S.wfe_initial, S.wfe_ns];
ptitl = {'initial', 'median'};
pgen  = {'', ''};
if K > 0
    psfs  = cat(3, psfs, S.psf_snap);
    wfes  = [wfes, S.wfe_snap(:).'];
    ptitl = [ptitl, arrayfun(@(k) char(64 + k), 1:K, 'UniformOutput', false)];
    pgen  = [pgen, arrayfun(@(g) sprintf('gen %d', g), snapG, 'UniformOutput', false)];
end
psfs  = cat(3, psfs, S.psf_final);
wfes  = [wfes, S.wfe_final];
ptitl = [ptitl, {'final'}];
if nGen > 0, pgen = [pgen, {sprintf('gen %d', nGen)}]; else, pgen = [pgen, {''}]; end
iFinal = size(psfs, 3);
if opts.show_ideal
    psfs  = cat(3, psfs, S.psf_ideal);
    wfes  = [wfes, 0];
    ptitl = [ptitl, {'aligned'}];
    pgen  = [pgen, {''}];
end
nP    = size(psfs, 3);
if noisy, ptitl{1} = 'observed'; end

% ---------------------------------------------------------------- style
C.ink    = hex2rgb('#0b0b0b');
C.ink2   = hex2rgb('#52514e');
C.grid   = hex2rgb('#e4e3df');
C.hair   = hex2rgb('#cfcdc6');
C.series = hex2rgb('#2a78d6');   % one series per axes -> one hue
C.truth  = hex2rgb('#52514e');
fs  = opts.fs;
fnt = opts.font;

% ------------------------------------------------------------- geometry
% Everything is laid out in inches, then converted to normalised units, so
% the figure is identical on screen and in the exported PDF.
W      = opts.width_in;
L      = 0.62;                   % left margin (y labels of line plots)
Rm     = 0.72;                   % right margin (colour bar)
gapP   = 0.05;                   % gap between PSF panels
s      = min(opts.max_psf_in, (W - L - Rm - (nP - 1) * gapP) / nP);   % PSF panel size
hdrH   = 0.34;  subH = 0.15;  belowH = 0.32;
opdH   = opts.show_opd * (s + 0.30);
wfeH   = 1.30;
nCol   = min(P, 5);  nRow = ceil(P / nCol);
keyH   = 0.20;  titleH = 0.17;  miniH = 0.66;  miniGapY = 0.08;  miniGapX = 0.42;
if hasTr
    parH = opts.show_params * (keyH + nRow * (titleH + miniH) + (nRow - 1) * miniGapY);
else
    rowH = 0.17;                  % one row per parameter in the error plot
    parH = opts.show_params * (keyH + 0.10 + P * rowH + 0.40);
end
botH   = 0.48 - 0.36 * (~hasTr && opts.show_params);
if ~hasTr && isempty(cost), wfeH = 0; end
H      = 0.06 + hdrH + subH + s + belowH + opdH + 0.30 + wfeH + ...
         opts.show_params * 0.14 + (~hasTr) * opts.show_params * 0.40 + parH + botH;

vis = 'on';
if ~opts.visible, vis = 'off'; end
fig = figure('Units', 'inches', 'Position', [1 1 W H], 'Color', 'w', ...
             'PaperUnits', 'inches', 'PaperSize', [W H], ...
             'PaperPosition', [0 0 W H], 'PaperPositionMode', 'manual', ...
             'InvertHardcopy', 'off', 'Visible', vis);
set(fig, 'DefaultAxesFontName', fnt, 'DefaultTextFontName', fnt, ...
         'DefaultAxesFontSize', fs, 'DefaultTextFontSize', fs);
toN = @(x, y, w, h) [x / W, y / H, w / W, h / H];

% overlay axes in figure inches, for headers, brackets and panel letters
ov = axes('Position', [0 0 1 1], 'XLim', [0 W], 'YLim', [0 H], ...
          'Visible', 'off', 'HitTest', 'off');
hold(ov, 'on');

% --------------------------------------------------------- (a) PSF strip
y = H - 0.06 - hdrH - subH - s;              % bottom of PSF row
xP = L + (0:nP - 1) * (s + gapP);            % left edge of each panel
lo = log10(opts.log_floor);
cmapPSF = viridis(256);

axP = zeros(1, nP);
for i = 1:nP
    axP(i) = axes('Position', toN(xP(i), y, s, s));
    img = log10(max(psfs(:, :, i), opts.log_floor));
    imagesc(img, 'Parent', axP(i));
    set(axP(i), 'CLim', [lo 0]);
    colormap(axP(i), cmapPSF);
    axis(axP(i), 'image');
    set(axP(i), 'XTick', [], 'YTick', [], 'Box', 'off', 'XColor', 'none', 'YColor', 'none');

    % panel title (above) and values (below), in figure inches
    text(xP(i) + s / 2, y + s + 0.05, ptitl{i}, 'Parent', ov, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'baseline', ...
         'FontSize', fs, 'FontWeight', ifelse(i >= 3 && i <= 2 + K, 'bold', 'normal'), 'Color', C.ink);
    % one size smaller: "1.2x10^-4 waves" has to fit under a ~0.8 in panel
    text(xP(i) + s / 2, y - 0.04, sprintf('%s waves', fmt3(wfes(i))), 'Parent', ov, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'FontSize', fs - 1, 'Color', C.ink);
    if ~isempty(pgen{i})
        text(xP(i) + s / 2, y - 0.18, pgen{i}, 'Parent', ov, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'FontSize', fs - 1, 'Color', C.ink2);
    end
end
% "RMS WFE" key to the left of the values row
text(L - 0.06, y - 0.04, 'WFE', 'Parent', ov, 'HorizontalAlignment', 'right', ...
     'VerticalAlignment', 'top', 'FontSize', fs - 1, 'Color', C.ink2);
text(L - 0.06, y - 0.18, 'DE', 'Parent', ov, 'HorizontalAlignment', 'right', ...
     'VerticalAlignment', 'top', 'FontSize', fs - 1, 'Color', C.ink2);

% stage headers with brackets: [start] [stage 1] [stage 2 .......]
yb = y + s + subH + 0.06;                    % bracket height
groups = {1, 'Start';  2, stage1;  3:iFinal, stage2};
if opts.show_ideal, groups(end + 1, :) = {nP, 'Reference'}; end
for g = 1:size(groups, 1)
    idx = groups{g, 1};
    x0 = xP(idx(1));  x1 = xP(idx(end)) + s;
    line([x0 x0 x1 x1], [yb - 0.04, yb, yb, yb - 0.04], 'Parent', ov, ...
         'Color', C.ink2, 'LineWidth', 0.6);
    text((x0 + x1) / 2, yb + 0.05, groups{g, 2}, 'Parent', ov, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'baseline', ...
         'FontSize', fs, 'FontWeight', 'bold', 'Color', C.ink);
end
panel_letter(ov, 0.06, yb + 0.03 + 0.15, 'a', fs, C.ink);

% lambda/D scale bar in the first panel
if isfield(S, 'lamD_pix') && isfinite(S.lamD_pix) && S.lamD_pix > 0
    npx  = size(psfs, 2);
    span = npx / S.lamD_pix;                 % panel width in lambda/D
    cand = [1 2 5 10 20 50];
    Lb   = cand(find(cand <= 0.3 * span, 1, 'last'));
    if isempty(Lb), Lb = cand(1); end
    xb = 0.08 * npx;  yb2 = 0.90 * size(psfs, 1);
    hold(axP(1), 'on');
    line(xb + [0, Lb * S.lamD_pix], [yb2 yb2], 'Parent', axP(1), 'Color', 'w', 'LineWidth', 1.5);
    text(xb + Lb * S.lamD_pix / 2, yb2 - 0.04 * npx, sprintf('%g\\lambda/D', Lb), ...
         'Parent', axP(1), 'Color', 'w', 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', fs - 1);
else
    warning('make_psf_alignment_figure:noScale', 'lamD_pix missing: no scale bar drawn.');
end

% shared colour bar, drawn as its own axes so it lands exactly here
axC = axes('Position', toN(xP(end) + s + 0.10, y, 0.08, s));
ramp = linspace(lo, 0, 256).';
imagesc([0 1], [lo 0], ramp, 'Parent', axC);
colormap(axC, cmapPSF);
tk = lo:0;
tl = arrayfun(@(e) sprintf('10^{%d}', e), tk, 'UniformOutput', false);
set(axC, 'CLim', [lo 0], 'YDir', 'normal', 'XTick', [], 'YTick', tk, 'YTickLabel', tl, ...
    'YAxisLocation', 'right', 'TickDir', 'out', 'TickLength', [0.04 0.04], ...
    'FontSize', fs - 1, 'Box', 'off', 'XColor', 'none', 'YColor', C.ink2, 'LineWidth', 0.6);
try, set(axC, 'TickLabelInterpreter', 'tex'); catch, end
ylabel(axC, 'I / I_{peak, ideal}', 'FontSize', fs - 1, 'Color', C.ink);

% ------------------------------------------------ optional OPD residual row
if opts.show_opd && isfield(S, 'opd_initial')
    opds = cat(3, S.opd_initial, S.opd_ns, S.opd_snap, S.opd_final);
    yo = y - belowH - s;
    cmapD = diverging(256);
    for i = 1:nP
        ax = axes('Position', toN(xP(i), yo, s, s));
        m = opds(:, :, i);
        lim = max(abs(m(isfinite(m))));
        if isempty(lim) || lim == 0 || ~isfinite(lim), lim = 1; end
        ok = isfinite(m);
        m(~ok) = 0;                           % neutral if alpha is unsupported
        h = imagesc(m, 'Parent', ax);
        set(h, 'AlphaData', double(ok));      % outside the pupil -> white
        set(ax, 'CLim', [-lim lim], 'Color', 'w');
        colormap(ax, cmapD);
        axis(ax, 'image');
        set(ax, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none');
        text(xP(i) + s / 2, yo - 0.04, sprintf('\\pm%s waves', fmt3(lim)), 'Parent', ov, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'FontSize', fs - 1, 'Color', C.ink2);
    end
    text(L - 0.06, yo + s / 2, {'residual', 'OPD'}, 'Parent', ov, 'HorizontalAlignment', 'right', ...
         'VerticalAlignment', 'middle', 'FontSize', fs - 1, 'Color', C.ink2);
    y = yo;
end

% --------------------------------------------------- (b) WFE vs generation
plotR = W - Rm;                              % right edge shared by the line plots
if hasTr
    yW = y - belowH - 0.30 - wfeH;
    axW = axes('Position', toN(L, yW, plotR - L, wfeH));
    hold(axW, 'on');
    style_axes(axW, C, fs);
    set(axW, 'YScale', 'log', 'XScale', opts.xscale);
    xl = [1, max(gen(end), 2)];
    yl = [min(wfe(wfe > 0)) / 2, max(wfe) * 2];
    if opts.marechal
        yl = [min(yl(1), 1 / 14 / 2), max(yl(2), 1 / 14 * 2)];
    end
    set(axW, 'XLim', xl, 'YLim', yl);
    set_gen_ticks(axW, xl, opts.xscale);

    for g = snapG
        line([g g], yl, 'Parent', axW, 'Color', C.hair, 'LineWidth', 0.5);
    end
    if opts.marechal
        line(xl, [1 1] / 14, 'Parent', axW, 'Color', C.ink2, 'LineWidth', 0.6);
        if exist('OCTAVE_VERSION', 'builtin'), e_acute = 'e'; else, e_acute = char(233); end
        % label under the line at the right end, where a converged curve sits far below
        text(xl(2), 1 / 14 / 1.2, ['Mar' e_acute 'chal \lambda/14 '], 'Parent', axW, ...
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
             'FontSize', fs - 1, 'Color', C.ink2, 'BackgroundColor', 'w', 'Margin', 1);
    end
    stairs(axW, gen, wfe, 'Color', C.series, 'LineWidth', 1.2);
    % letters sit on the top edge; the hairline below each one leads to its dot.
    % Letters closer than ~2% of the axis width are nudged apart.
    if strcmp(opts.xscale, 'log')
        fwd = @(x) (log10(x) - log10(xl(1))) / (log10(xl(2)) - log10(xl(1)));
        invf = @(u) 10 .^ (log10(xl(1)) + u * (log10(xl(2)) - log10(xl(1))));
    else
        fwd = @(x) (x - xl(1)) / diff(xl);
        invf = @(u) xl(1) + u * diff(xl);
    end
    u = fwd(snapG);
    for k = 2:K
        u(k) = max(u(k), u(k - 1) + 0.022);
    end
    labX = invf(u);
    for k = 1:K
        g = snapG(k);
        plot(axW, g, wfe(g), 'o', 'MarkerSize', 5, 'MarkerFaceColor', C.ink, ...
             'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
        text(labX(k), yl(2), char(64 + k), 'Parent', axW, 'FontWeight', 'bold', 'Clipping', 'off', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', fs, 'Color', C.ink);
    end
    plot(axW, gen(end), wfe(end), 'o', 'MarkerSize', 5, 'MarkerFaceColor', C.ink, ...
         'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
    % "final" goes in the right margin, clear of the curve
    text(xl(2), wfe(end), '  final', 'Parent', axW, 'Clipping', 'off', ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', fs - 1, 'Color', C.ink);
    ylabel(axW, 'RMS WFE (waves)', 'FontSize', fs, 'Color', C.ink);
    if opts.show_params
        set(axW, 'XTickLabel', []);
    else
        xlabel(axW, 'DE generation', 'FontSize', fs, 'Color', C.ink);
    end
    panel_letter(ov, 0.06, yW + wfeH + 0.14, 'b', fs, C.ink);

else
    % ---- (b') no parameter log: plot the DE cost that disp=True printed
    yW = y - belowH - 0.30 - wfeH;
    if ~isempty(cost)
        axW = axes('Position', toN(L, yW, plotR - L, wfeH));
        hold(axW, 'on');
        style_axes(axW, C, fs);
        xl = [1, max(nGen, 2)];
        yl = [min(cost(cost > 0)) / 2, max(cost) * 2];
        set(axW, 'YScale', 'log', 'XScale', opts.xscale, 'XLim', xl, 'YLim', yl);
        set_gen_ticks(axW, xl, opts.xscale);
        stairs(axW, gen, cost, 'Color', C.series, 'LineWidth', 1.2);
        plot(axW, gen(end), cost(end), 'o', 'MarkerSize', 5, 'MarkerFaceColor', C.ink, ...
             'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
        text(xl(2), cost(end), '  final', 'Parent', axW, 'Clipping', 'off', ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', fs - 1, 'Color', C.ink);
        ylabel(axW, 'DE cost', 'FontSize', fs, 'Color', C.ink);
        xlabel(axW, 'DE generation', 'FontSize', fs, 'Color', C.ink);
        % what the cost is, stated once above the plot
        text(plotR, yW + wfeH + 0.04, get_str(S, 'cost_label', 'DE cost'), 'Parent', ov, ...
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
             'FontSize', fs - 1, 'Color', C.ink2);
        panel_letter(ov, 0.06, yW + wfeH + 0.14, 'b', fs, C.ink);
    end
end

% ------------------------------------------- (c) parameters vs generation
if hasTr
    if opts.show_params
        yTop = yW - 0.14;                         % top of panel (c) block
        % key: which line is which, on the same line as the panel letter
        kx = L;  ky = yTop - 0.08;
        line(kx + [0 0.22], [ky ky], 'Parent', ov, 'Color', C.series, 'LineWidth', 1.2);
        text(kx + 0.27, ky, 'DE best estimate', 'Parent', ov, 'VerticalAlignment', 'middle', ...
             'FontSize', fs - 1, 'Color', C.ink);
        kx2 = kx + 1.35;
        line(kx2 + [0 0.22], [ky ky], 'Parent', ov, 'Color', C.truth, 'LineWidth', 0.8);
        text(kx2 + 0.27, ky, 'injected truth', 'Parent', ov, 'VerticalAlignment', 'middle', ...
             'FontSize', fs - 1, 'Color', C.ink);
        panel_letter(ov, 0.06, yTop, 'c', fs, C.ink);

        miniW = (plotR - L - (nCol - 1) * miniGapX) / nCol;
        for p = 1:P
            r = ceil(p / nCol);  c = p - (r - 1) * nCol;
            xm = L + (c - 1) * (miniW + miniGapX);
            ym = yTop - keyH - r * (titleH + miniH) - (r - 1) * miniGapY;
            ax = axes('Position', toN(xm, ym, miniW, miniH));
            hold(ax, 'on');
            style_axes(ax, C, fs - 1);
            set(ax, 'XScale', opts.xscale, 'XLim', xl);
            set_gen_ticks(ax, xl, opts.xscale);
            v = theta(:, p);
            lim = [min([v; truth(p)]), max([v; truth(p)])];
            pad = 0.08 * max(diff(lim), eps(max(abs(lim))) * 10);
            set(ax, 'YLim', lim + [-pad pad], 'YTick', nice_ticks(lim + [-pad pad], 4));
            for g = snapG
                line([g g], lim + [-pad pad], 'Parent', ax, 'Color', C.hair, 'LineWidth', 0.5);
            end
            line(xl, truth(p) * [1 1], 'Parent', ax, 'Color', C.truth, 'LineWidth', 0.8);
            stairs(ax, gen, v, 'Color', C.series, 'LineWidth', 1.0);
            % left-aligned title placed in inches (title() offsets differ by version)
            text(xm, ym + miniH + 0.03, param_label(names{p}, units{p}), 'Parent', ov, ...
                 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
                 'FontSize', fs - 1, 'Color', C.ink);
            if r < nRow && p + nCol <= P
                set(ax, 'XTickLabel', []);
            else
                xlabel(ax, 'DE generation', 'FontSize', fs - 1, 'Color', C.ink);
            end
        end
    end

elseif opts.show_params
    % ---- (c') no parameter log: error after Nautilus vs after DE, one scale
    yTop = yW - 0.14 - 0.40;                  % leave room for (b)'s x label
    if isfield(S, 'prior_halfwidth') && ~isempty(S.prior_halfwidth)
        scl = S.prior_halfwidth(:).';  xlab = '|estimate - truth| / prior half-width';
    else
        scl = max(abs(truth), eps);    xlab = '|estimate - truth| / |truth|';
    end
    eNS = abs(S.theta_ns(:).' - truth);  eDE = abs(S.theta_final(:).' - truth);
    rNS = max(eNS ./ scl, realmin);      rDE = max(eDE ./ scl, realmin);

    % key
    ky = yTop - 0.08;
    plot(ov, L + 0.06, ky, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
         'MarkerEdgeColor', C.ink2, 'LineWidth', 0.9);
    text(L + 0.16, ky, 'after Nautilus (median)', 'Parent', ov, 'VerticalAlignment', 'middle', ...
         'FontSize', fs - 1, 'Color', C.ink);
    plot(ov, L + 1.70, ky, 'o', 'MarkerSize', 5, 'MarkerFaceColor', C.series, ...
         'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
    text(L + 1.80, ky, 'after DE (final)', 'Parent', ov, 'VerticalAlignment', 'middle', ...
         'FontSize', fs - 1, 'Color', C.ink);
    panel_letter(ov, 0.06, yTop, 'c', fs, C.ink);

    colW = 0.55;                              % two value columns on the right
    axR  = plotR - 2 * colW - 0.10;
    xm   = L + 0.30;                          % room for "pdx (um)" labels
    ym   = yTop - keyH - 0.10 - P * rowH;
    ax = axes('Position', toN(xm, ym, axR - xm, P * rowH));
    hold(ax, 'on');
    style_axes(ax, C, fs - 1);
    lo10 = floor(log10(min([rNS rDE]))) ;  hi10 = ceil(log10(max([rNS rDE])));
    if hi10 <= lo10, hi10 = lo10 + 1; end
    set(ax, 'XScale', 'log', 'XLim', 10 .^ [lo10 hi10], 'YDir', 'reverse', ...
        'YLim', [0.5 P + 0.5], 'YTick', 1:P, 'YGrid', 'off', ...
        'YTickLabel', cellfun(@(n, u) param_label(n, u), names, units, 'UniformOutput', false));
    set(ax, 'XTick', 10 .^ (lo10:hi10));
    for p = 1:P
        line([rNS(p) rDE(p)], [p p], 'Parent', ax, 'Color', C.hair, 'LineWidth', 1.5);
        plot(ax, rNS(p), p, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'MarkerEdgeColor', C.ink2, 'LineWidth', 0.9);
        plot(ax, rDE(p), p, 'o', 'MarkerSize', 5, 'MarkerFaceColor', C.series, ...
             'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
        yrow = ym + (P - p + 0.5) * rowH;     % row centre in inches (reversed axis)
        text(axR + 0.10 + colW, yrow, sprintf('%+.2g', S.theta_ns(p) - truth(p)), 'Parent', ov, ...
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', fs - 1, 'Color', C.ink2);
        text(axR + 0.10 + 2 * colW, yrow, sprintf('%+.2g', S.theta_final(p) - truth(p)), 'Parent', ov, ...
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', fs - 1, 'Color', C.ink);
    end
    text(axR + 0.10 + colW, ym + P * rowH + 0.03, 'Nautilus', 'Parent', ov, 'HorizontalAlignment', 'right', ...
         'VerticalAlignment', 'bottom', 'FontSize', fs - 1, 'Color', C.ink2);
    text(axR + 0.10 + 2 * colW, ym + P * rowH + 0.03, 'DE', 'Parent', ov, 'HorizontalAlignment', 'right', ...
         'VerticalAlignment', 'bottom', 'FontSize', fs - 1, 'FontWeight', 'bold', 'Color', C.ink);
    text(axR + 0.10 + 2 * colW, ym - 0.06, 'estimate - truth (units at left)', 'Parent', ov, ...
         'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', fs - 2, 'Color', C.ink2);
    xlabel(ax, xlab, 'FontSize', fs - 1, 'Color', C.ink);
end

% ------------------------------------------------------------------ save
if opts.save
    if exist('exportgraphics') > 0 %#ok<EXIST>
        exportgraphics(fig, [opts.out '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
        exportgraphics(fig, [opts.out '.png'], 'Resolution', 600, 'BackgroundColor', 'white');
    else
        print(fig, [opts.out '.pdf'], '-dpdf', '-painters');
        print(fig, [opts.out '.png'], '-dpng', '-r300');
    end
    fprintf('Saved %s.pdf and %s.png\n', opts.out, opts.out);
end

print_caption(S, snapG, nGen, noisy, stage1, stage2, opts);
end

% ======================================================================
% helpers
% ======================================================================
function o = merge_opts(d, o)
f = fieldnames(d);
for k = 1:numel(f)
    if ~isfield(o, f{k}) || isempty(o.(f{k}))
        o.(f{k}) = d.(f{k});
    end
end
end

function c = cellstr_row(x)
if ischar(x)
    c = cellstr(x).';
else
    c = cellfun(@(s) strtrim(char(s)), x(:).', 'UniformOutput', false);
end
end

function s = get_str(S, f, dflt)
if isfield(S, f) && ~isempty(S.(f))
    s = strtrim(char(S.(f)));
else
    s = dflt;
end
end

function out = ifelse(cond, a, b)
if cond, out = a; else, out = b; end
end

function s = fmt3(v)
% three significant figures, no exponent for the values seen here
if v == 0
    s = '0';
elseif v >= 0.001
    s = sprintf('%.3g', v);
else
    e = floor(log10(v));
    s = sprintf('%.2g\\times10^{%d}', v / 10 ^ e, e);
end
end

function lbl = param_label(name, unit)
u = strtrim(unit);
d = double(u);
% 'um', 'micron', or a micro sign / Greek mu (MATLAB: 181 / 956; Octave UTF-8 bytes: 194 / 206)
if any(strcmp(u, {'um', 'micron', 'microns'})) || ...
        (numel(d) >= 2 && any(d(1) == [181 956 194 206]) && u(end) == 'm')
    u = '\mum';
end
if isempty(u)
    lbl = name;
else
    lbl = sprintf('%s (%s)', name, u);
end
end

function style_axes(ax, C, fs)
set(ax, 'Box', 'off', 'TickDir', 'out', 'TickLength', [0.012 0.012], ...
    'XColor', C.ink2, 'YColor', C.ink2, 'LineWidth', 0.6, 'FontSize', fs, ...
    'XGrid', 'on', 'YGrid', 'on', 'XMinorGrid', 'off', 'YMinorGrid', 'off', ...
    'Layer', 'bottom', 'Color', 'w');
% hairline, solid, recessive grid (property names differ across versions)
try, set(ax, 'GridColor', C.grid, 'GridAlpha', 1, 'GridLineStyle', '-'); catch, end
try, set(ax, 'XMinorTick', 'off', 'YMinorTick', 'off'); catch, end
end

function t = nice_ticks(lim, n)
% the finest 1-2-5 step that puts at most N (and at least 2) ticks inside LIM
span = diff(lim);
if ~(span > 0), t = lim(1); return; end
mag = 10 ^ floor(log10(span));
steps = [0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10] * mag;
t = lim(1);
for st = steps
    tt = (ceil(lim(1) / st):floor(lim(2) / st)) * st;
    if numel(tt) <= n && numel(tt) >= 2
        t = tt;  break;
    end
end
t(abs(t) < span * 1e-9) = 0;
end

function set_gen_ticks(ax, xl, xscale)
if strcmp(xscale, 'log')
    t = 10 .^ (0:ceil(log10(xl(2))));
    t = t(t <= xl(2));
    set(ax, 'XTick', t, 'XTickLabel', arrayfun(@(v) sprintf('%d', v), t, 'UniformOutput', false));
end
end

function panel_letter(ov, x, y, ch, fs, col)
text(x, y, sprintf('(%s)', ch), 'Parent', ov, 'FontWeight', 'bold', ...
     'FontSize', fs + 1, 'Color', col, 'VerticalAlignment', 'top');
end

function rgb = hex2rgb(h)
h = strrep(h, '#', '');
rgb = [hex2dec(h(1:2)), hex2dec(h(3:4)), hex2dec(h(5:6))] / 255;
end

function m = viridis(n)
% matplotlib viridis (CC0), 17 anchors, linearly interpolated
a = [0.2670 0.0049 0.3294; 0.2823 0.0950 0.4173; 0.2788 0.1755 0.4834;
     0.2590 0.2515 0.5247; 0.2297 0.3224 0.5457; 0.1994 0.3876 0.5546;
     0.1727 0.4488 0.5579; 0.1490 0.5081 0.5573; 0.1276 0.5669 0.5506;
     0.1206 0.6258 0.5335; 0.1579 0.6838 0.5017; 0.2461 0.7389 0.4520;
     0.3692 0.7889 0.3829; 0.5160 0.8312 0.2943; 0.6785 0.8637 0.1895;
     0.8456 0.8873 0.0997; 0.9932 0.9062 0.1439];
m = interp1(linspace(0, 1, size(a, 1)), a, linspace(0, 1, n));
end

function m = diverging(n)
% blue -> neutral gray -> red, equal arms
a = [hex2rgb('#184f95'); hex2rgb('#2a78d6'); hex2rgb('#9ec5f4'); hex2rgb('#f0efec');
     hex2rgb('#f3b0ad'); hex2rgb('#e34948'); hex2rgb('#a32b2b')];
m = interp1(linspace(0, 1, size(a, 1)), a, linspace(0, 1, n));
end

function print_caption(S, snapG, nGen, noisy, stage1, stage2, opts)
w = S.wavelength_um;
K = numel(snapG);
if noisy
    first = sprintf('the observed misaligned PSF (Gaussian noise, sigma = %g%% of the ideal peak)', S.noise_sigma);
else
    first = 'the misaligned PSF';
end
st2 = stage2;  st2(1) = lower(st2(1));
if K > 0
    g = sprintf('%d, ', snapG);  g = g(1:end - 2);
    mid = sprintf('during %s (DE) at generations %s (A-%s), and after %d generations (final)', ...
                  st2, g, char(64 + K), nGen);
else
    mid = sprintf('and after %d generations of %s (DE, final)', nGen, st2);
end
if opts.show_ideal, mid = [mid ', next to the aligned system (reference)']; end
cap = sprintf(['(a) On-axis PSFs at lambda = %.2f um: %s, after %s (posterior median), %s. ' ...
    'Intensity is relative to the aligned system''s peak, so each panel''s peak equals its ' ...
    'Strehl ratio (log stretch, floor %g). Values are RMS wavefront error in waves, the RMS of ' ...
    'the OPD difference from the nominal design.'], w, first, stage1, mid, opts.log_floor);
if K > 0
    cap = [cap ' (b) RMS WFE of the best DE member at each generation; letters mark the PSFs in (a).'];
    if opts.marechal
        cap = [cap ' Gray line: Marechal criterion, lambda/14.'];
    end
    if opts.show_params
        cap = [cap ' (c) Best-member value of each alignment parameter at each generation (blue) ' ...
               'and the injected truth (gray).'];
    end
else
    cap = [cap sprintf([' (b) %s of the best DE member at each generation. Parameter vectors ' ...
           'were not logged during this run, so WFE is known only at the start and end.'], ...
           get_str(S, 'cost_label', 'DE cost'))];
    if opts.show_params
        cap = [cap ' (c) Error of each parameter after Nautilus (open circles) and after DE ' ...
               '(filled), divided by the prior half-width; the columns give estimate minus truth.'];
    end
end
fprintf('\nSuggested caption:\n%s\n\n', cap);
end
