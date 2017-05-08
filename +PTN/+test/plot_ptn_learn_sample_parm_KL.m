function [ figures ] = plot_ptn_learn_sample_parm_KL( resultsPath, qName, yRange )
% plot_ptn_learn Summary of this function goes here
%   Detailed explanation goes here
    [testDesc, results] = PTN.test.parse_PTN_test_results(resultsPath);
    numSamplePoints = numel(testDesc.nSamples);
    numTarget = numel(testDesc.target);
    numRes = numel(testDesc.res);
    
    numQPS = size(results(1).parameterKL, 2);
    
    targetTypes = find(ismember(PTN.test.randomPTNTargetTypes(), testDesc.target));
    
    plotDesc = arrayfun(@(x) sprintf('ptnLearn, res=%d', x), testDesc.res, 'UniformOutput', false);
    
    figures = cell(numTarget*numQPS, 1);
    
    for q=1:numQPS
        for i=1:numTarget
            figIdx = sub2ind([numQPS, numTarget], q, i);
            figures{figIdx} = figure(figIdx);
            clf;
            set(figures{figIdx}, 'Visible', 'off');
            points = zeros(numSamplePoints, numRes);
            for j=1:numRes
                for x=1:numSamplePoints
                    thisKL = results([results.target] == targetTypes(i) ...
                        & [results.res] == testDesc.res(j) ...
                        & [results.nSamples] == testDesc.nSamples(x));

                    points(x,j) = mean(arrayfun(@(y) y.parameterKL(q), thisKL));
                end
            end
            
            semilogx(testDesc.nSamples, points, '-o');
            ax = get(figures{figIdx}, 'CurrentAxes');
            if nargin >= 3
                set(ax, 'YLim', yRange);
            end
            
            legend(ax, plotDesc{:}, 'Location', 'northeast');
            xlabel(ax, 'Number of samples');
            ylabel(ax, 'Mean KL to target');
            title(ax, sprintf('%s mean KL for QPS #%d vs sample count, targets generated via %s\n(%d trials per condition)', qName, q, testDesc.target{i}, testDesc.nTrials));
        end
    end
    
    % Display mode
    if nargout == 0
        for i=1:numel(figures)
            set(figures{i}, 'Visible', 'on');
        end
    end
end

