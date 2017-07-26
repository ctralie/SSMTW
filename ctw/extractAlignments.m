% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 01-20-2012
%   modify  -  Feng Zhou (zhfe99@gmail.com), 05-05-2013
%   modify  -  Chris Tralie (chris.tralie@gmail.com), 06-16-2017

function [] = extractAlignments()
    addpath(genpath('ctw'));
    addpath(genpath('.'));
    load('Xs.mat');

    addPath;
    prSet(1);
    
    Y1 = cmdscale(pdist2(X1, X1));
    Y2 = cmdscale(pdist2(X2, X2));
    if ~(size(Y1, 2) == size(Y2, 2))
        dim = max(size(Y1, 2), size(Y2, 2));
        temp = zeros(size(Y1, 1), dim);
        temp(:, 1:size(Y1, 2)) = Y1;
        Y1 = temp;
        temp = zeros(size(Y2, 1), dim);
        temp(:, 1:size(Y2, 2)) = Y2;
        Y2 = temp;
    end
    
    %% Setup time series
    Xs = cell(1, 2);
    Xs{1} = X1';
    Xs{2} = X2';
    
    Ys = cell(1, 2);
    Ys{1} = Y1';
    Ys{2} = Y2';

    %% src parameter
    l = 300; % #frame of the latent sequence (Z)
    aliT = [];

    %% algorithm parameters
    parDtw = [];
    parImw = st('lA', 1, 'lB', 1); % IMW: regularization weight
    parCca = st('d', .95); % CCA: reduce dimension to keep at least 0.95 energy
    parCtw = [];
    parGN = st('nItMa', 2, 'inp', 'linear'); % Gauss-Newton: 2 iterations to update the weight in GTW, 
    parGtw = st('nItMa', 20);

    %% monotonic basis
    ns = cellDim(Xs, 2);
    bas = baTems(l, ns, 'stp', [], 'pol', [5, 0.5], 'tan', [5 1 1], 'log', [5], 'exp', 5);

    %% utw (initialization, uniform time warping)
    aliUtw = utw(Xs, bas, aliT);

    %% dtw
    aliDtw = dtw(Ys, aliT, parDtw);
    PDTW = aliDtw.P;

    %% ddtw
    aliDdtw = ddtw(Ys, aliT, parDtw);
    PDDTW = aliDdtw.P;

    %% imw
    aliImw = pimw(Ys, aliUtw, aliT, parImw, parDtw);
    PIMW = aliImw.P;

    %% ctw
    aliCtw = ctw(Xs, aliUtw, aliT, parCtw, parCca, parDtw);
    PCTW = aliCtw.P;

    %% gtw
    aliGtw = gtw(Xs, bas, aliUtw, aliT, parGtw, parCca, parGN);
    PGTW = aliGtw.P;

    save('matlabResults.mat', 'PDTW', 'PDDTW', 'PIMW', 'PCTW', 'PGTW');
end
