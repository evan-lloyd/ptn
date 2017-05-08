function [ figures ] = plot_ptn_learn_sample_accuracy_mean( resultsPath, qName, yRange )
% plot_ptn_learn Summary of this function goes here
%   Detailed explanation goes here
    [testDesc, results] = PTN.test.parse_PTN_test_results(resultsPath);
    numSamplePoints = numel(testDesc.nSamples);
    numTarget = numel(testDesc.target);
    numRes = numel(testDesc.res);
    
    pointKL = [results.KL];
    pointSampleKL = [results.sampleKL];
    
    targetTypes = find(ismember(PTN.test.randomPTNTargetTypes(), testDesc.target));
    
    plotDesc = arrayfun(@(x) sprintf('ptnLearn, res=%d', x), testDesc.res, 'UniformOutput', false);
    plotDesc{end+1} = 'Sample distribution';
    
    figures = cell(numTarget, 1);
    
    for i=1:numTarget
        figures{i} = figure(i);
        clf;
        set(figures{i}, 'Visible', 'off');
        points = zeros(numSamplePoints, numRes+1);
        for j=1:numRes
            points(:,j) = arrayfun(@(x) mean(...
                pointKL([results.target] == targetTypes(i) ...
                & [results.res] == testDesc.res(j) ...
                & [results.nSamples] == x)), ...
                testDesc.nSamples);
        end
        points(:,j+1) = arrayfun(@(x) mean(...
            pointSampleKL([results.target] == targetTypes(i) ...
            & [results.res] == max(testDesc.res) ...
            & [results.nSamples] == x)), ...
            testDesc.nSamples);
        
        loglog(testDesc.nSamples, points, '-o');
        ax = get(figures{i}, 'CurrentAxes');
        if nargin >= 3
            set(ax, 'YLim', yRange);
        end
        
        legend(ax, plotDesc{:}, 'Location', 'northeast');
        xlabel(ax, 'Number of samples');
        ylabel(ax, 'Mean KL to target');
        title(ax, sprintf('%s mean KL vs sample count, targets generated via %s\n(%d trials per condition)', qName, testDesc.target{i}, testDesc.nTrials));
    end
    
    % Display mode
    if nargout == 0
        for i=1:numel(figures)
            set(figures{i}, 'Visible', 'on');
        end
    end
end

