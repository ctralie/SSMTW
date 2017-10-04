% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 01-20-2012
%   modify  -  Feng Zhou (zhfe99@gmail.com), 05-05-2013
%   modify  -  Chris Tralie (chris.tralie@gmail.com), 06-16-2017

function [] = extractAlignments()
    addPath();
    load('Xs.mat');
    prSet(1);
    
    if size(X1, 2) ~= size(X2, 2)
        %Zeropad if sizes aren't the same
        N = max(size(X1, 2), size(X2, 2));
        if size(X1, 2) < N
            temp = zeros(size(X1, 1), N);
            temp(:, 1:size(X1, 2)) = X1;
            X1 = temp;
        end
        if size(X2, 2) < N
            temp = zeros(size(X2, 1), N);
            temp(:, 1:size(X2, 2)) = X2;
            X2 = temp;
        end
    end
    
    %% Setup time series
    X0s = cell(1, 2);
    X0s{1} = double(X1');
    X0s{2} = double(X2');
    if doPCA
        Xs = pcas(X0s, st('d', 0.99));
        Xs2 = pcas(X0s, st('d', 5));
        fprintf(1, 'Original Dimension = %i\n', size(X0s{1}, 1));
        fprintf(1, 'Reduced Dimension = %i\n', size(Xs{1}, 1));
        fprintf(1, 'Reduced Dimension 2 = %i\n', size(Xs2{1}, 1));
    else
        Xs = X0s;
        Xs2 = X0s;
    end
    X1Mean = bsxfun(@minus, Xs{1}', mean(Xs{1}', 1));
    IMWReg = 30*mean(sqrt(sum(X1Mean.^2, 2)));
    fprintf(1, 'IMWReg = %g\n', IMWReg);

    %% src parameter
    l = 300; % #frame of the latent sequence (Z)
    aliT = [];

    %% algorithm parameters
    parDtw = [];
    parImw = st('lA', IMWReg, 'lB', IMWReg); % IMW: regularization weight
    parCca = st('d', 3, 'lams', 0.1); % CCA: reduce dimension to keep at least 0.95 energy
    parCtw = st('nItMa', 100);
    parGN = st('nItMa', 2, 'inp', 'linear'); % Gauss-Newton: 2 iterations to update the weight in GTW, 
    parGtw = st('nItMa', 20);

    %% monotonic basis
    ns = cellDim(Xs, 2);
    bas = baTems(l, ns, 'stp', [], 'pol', [5, 0.5], 'tan', [5 1 1], 'log', [5], 'exp', 5);

    %% utw (initialization, uniform time warping)
    aliUtw = utw(Xs, bas, aliT);

    %% dtw
    aliDtw = dtw(Xs, aliT, parDtw);
    PDTW = aliDtw.P;
    
    %% ddtw
    aliDdtw = ddtw(Xs, aliT, parDtw);
    PDDTW = aliDdtw.P;

    %% imw
    aliImw = pimw(Xs, aliUtw, aliT, parImw, parDtw);
    PIMW = aliImw.P;

    %% ctw
    aliCtw = ctw(Xs2, aliDtw, aliT, parCtw, parCca, parDtw);
    PCTW = aliCtw.P;

    %% gtw
    PGTW = aliUtw.P;
    try
        aliGtw = gtw(Xs2, bas, aliUtw, aliT, parGtw, parCca, parGN);
        PGTW = aliGtw.P;
    catch ME
        disp('Error running GTW');
    end

    save('matlabResults.mat', 'PDTW', 'PDDTW', 'PIMW', 'PCTW', 'PGTW');
end
