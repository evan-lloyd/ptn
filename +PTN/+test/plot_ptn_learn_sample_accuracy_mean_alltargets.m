function [ fig ] = plot_ptn_learn_sample_accuracy_mean_alltargets( resultsPath, qName, yRange )
% plot_ptn_learn Summary of this function goes here
%   Detailed explanation goes here
    [testDesc, results] = PTN.test.parse_PTN_test_results(resultsPath);
    numSamplePoints = numel(testDesc.nSamples);
    numTarget = numel(testDesc.target);
    numRes = numel(testDesc.res);
    
    pointKL = [results.KL];
    pointSampleKL = [results.sampleKL];
    
    targetTypes = arrayfun(@(j) find(ismember(PTN.test.randomPTNTargetTypes(), testDesc.target{j})), 1:numel(testDesc.target));
    
    plotDesc = cell((numRes+1)*numTarget, 1);
    
    fig = figure(1);
    set(fig, 'Visible', 'off');
    clf;
    points = zeros(numSamplePoints, (numRes+1)*numTarget);
    
    for i=1:numTarget
        
        for j=1:numRes
            pointCol = sub2ind([numTarget numRes], i, j);
            points(:,pointCol) = arrayfun(@(x) mean(...
                pointKL([results.target] == targetTypes(i) ...
                & [results.res] == testDesc.res(j) ...
                & [results.nSamples] == x)), ...
                testDesc.nSamples);
            plotDesc{pointCol} = sprintf('ptnLearn, res=%d, target=%s', testDesc.res(j), testDesc.target{i});
        end
        points(:,numRes*numTarget+i) = arrayfun(@(x) mean(...
            pointSampleKL([results.target] == targetTypes(i) ...
            & [results.res] == max(testDesc.res) ...
            & [results.nSamples] == x)), ...
            testDesc.nSamples);
        plotDesc{numRes*numTarget+i} = sprintf('Quantized sample distribution, res=%d, target=%s', max(testDesc.res), testDesc.target{i});
    end
    
    loglog(testDesc.nSamples, points, '-o');
    
    ax = get(fig, 'CurrentAxes');
    
    legend(ax, plotDesc, 'Location', 'northeast');
    xlabel(ax, 'Number of samples');
    ylabel(ax, 'Mean KL to target');
    title(ax, sprintf('%s mean KL vs sample count\n(%d trials per condition)', qName, testDesc.nTrials));
    
    if nargin >= 3
        set(ax, 'YLim', yRange);
    end
    
    % Display mode
    if nargout == 0
        set(fig, 'Visible', 'on');
    end
end

