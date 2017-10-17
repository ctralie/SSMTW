function [X, XQ, boneNames] = getMOCAPTrajectories(asfName, amcName)
    addpath(genpath('MOCAP/HDM05-Parser'));
    [skel, mot] = readMocap(asfName, amcName);
    %Extract motion trajectories
    T = mot.jointTrajectories;
    X = zeros(size(T{1}, 1), length(T), size(T{1}, 2));
    for ii = 1:length(T)
        X(:, ii, :) = T{ii};
    end
    %Extract quaternions
    Q = mot.rotationQuat;
    XQ = zeros(4, size(X, 2), size(X, 3));
    XQ(1, :, :) = 1; %Identity quaternion in case it's not defined
    for ii = 1:length(T)
        if numel(Q{ii}) > 0
            XQ(:, ii, :) = Q{ii};
        end
    end
    boneNames = skel.boneNames;
end
