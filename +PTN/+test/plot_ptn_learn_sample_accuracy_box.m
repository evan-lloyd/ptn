function [figures] = plot_ptn_learn_sample_accuracy_box( resultsPath, qName, yRange )
% plot_ptn_learn Summary of this function goes here
%   Detailed explanation goes here
    [testDesc, results] = PTN.test.parse_PTN_test_results(resultsPath);
    numSamplePoints = numel(testDesc.nSamples);
    numTarget = numel(testDesc.target);
    numRes = numel(testDesc.res);
    
    pointKL = [results.KL];
    
    targetTypes = find(ismember(PTN.test.randomPTNTargetTypes(), testDesc.target));
    conditionsCrossShape = [numTarget, numRes];

    figures = cell(numTarget*numRes, 1);
    
    for i=1:numTarget
        for j=1:numRes
            points = arrayfun(@(x) ...
                pointKL([results.target] == targetTypes(i) ...
                & [results.res] == testDesc.res(j) ...
                & [results.nSamples] == x), ...
                testDesc.nSamples, 'UniformOutput', false);
            
            figIdx = sub2ind(conditionsCrossShape, i, j);
            figures{figIdx} = figure(figIdx);
            clf;
            set(figures{figIdx}, 'Visible', 'off');
            
            hold on;
            for k=1:numSamplePoints
                PTN.test.bplot(points{k}, testDesc.nSamples(k), 'outliers', 'logwidth', 'barwidth', 0.2, 'linewidth', 1);
            end
            hold off;
            
            ax = get(figures{figIdx}, 'CurrentAxes');
            set(ax, 'XScale', 'log');
            set(ax, 'YScale', 'log'); 
           
            if nargin >= 3
                set(ax, 'YLim', yRange);
            end
            
            xlabel(ax, 'Number of samples');
            ylabel(ax, 'KL to target');
            title(ax, sprintf('%s KL vs sample count, targets generated via %s\n(%d trials per condition)', qName, testDesc.target{i}, testDesc.nTrials));
        end
    end
    
    % Display mode
    if nargout == 0
        for i=1:numel(figures)
            set(figures{i}, 'Visible', 'on');
        end
    end
    
end

