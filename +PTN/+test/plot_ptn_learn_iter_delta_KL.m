function [ figures ] = plot_ptn_learn_iter_delta_KL( resultsPath, qName, yRange )
% plot_ptn_learn Summary of this function goes here
%   Detailed explanation goes here
    [testDesc, results] = PTN.test.parse_PTN_test_results(resultsPath);
    numSamplePoints = numel(testDesc.nSamples);
    numTarget = numel(testDesc.target);
    numRes = numel(testDesc.res);
    numQPS = size(results(1).iterParameterKL, 2);
    
    targetTypes = find(ismember(PTN.test.randomPTNTargetTypes(), testDesc.target));
    
    figures = cell(1,numTarget*numRes*numQPS);
    
    for q=1:numQPS
        for i=1:numTarget
            for j=1:numRes
                figIdx = sub2ind([numQPS numTarget numRes], q, i, j);
                figures{figIdx} = figure(figIdx);
                set(figures{figIdx}, 'Visible', 'off');
                clf;
                
                plotDesc = cell(numSamplePoints, 1);
        
                points = zeros(testDesc.maxIter, numSamplePoints);

                for k=1:numSamplePoints
                    thisKL = cell(testDesc.nTrials, 1);
                    [thisKL{:}] = results([results.target] == targetTypes(i) ...
                        & [results.nSamples] == testDesc.nSamples(k) ...
                        & [results.res] == testDesc.res(j)).deltaKL;
                    % Extract KL values for the current QPS
                    for t=1:testDesc.nTrials
                        thisKL{t} = thisKL{t}(:,q)';
                        thisKL{t} = [thisKL{t} repmat(thisKL{t}(end), 1,...
                            testDesc.maxIter-numel(thisKL{t}))];
                    end

                    points(:,k) = mean(cell2mat(thisKL), 1);
                    plotDesc{k} = sprintf('learnPTN, %0.0e samples', testDesc.nSamples(k));
                end
                
                plot(1:testDesc.maxIter, points, 'o');
                ax = get(figures{figIdx}, 'CurrentAxes');
                set(ax, 'YScale', 'log');
                
                if nargin >= 3
                    set(ax, 'YLim', yRange);
                end
                
                legend(ax, plotDesc{:}, 'Location', 'northeast');
                xlabel(ax, 'Number of alternating update iterations');
                ylabel(ax, 'Mean KL(QPS_{i-1}, QPS_{i})');
                title(ax, sprintf('%s mean parameter delta-KL for QPS #%d vs number of iterations,\ntargets generated via %s, res=%d (%d trials per condition)', qName, q, testDesc.target{i}, testDesc.res(j), testDesc.nTrials));
            end
        end
    end
    
    % Display mode
    if nargout == 0
        for i=1:numel(figures)
            set(figures{i}, 'Visible', 'on');
        end
    end
end

