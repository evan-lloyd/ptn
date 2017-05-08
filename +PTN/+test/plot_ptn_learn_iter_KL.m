function [ figures ] = plot_ptn_learn_iter_KL( resultsPath, qName, yRange )
% plot_ptn_learn Summary of this function goes here
%   Detailed explanation goes here
    [testDesc, results] = PTN.test.parse_PTN_test_results(resultsPath);
    numSamplePoints = numel(testDesc.nSamples);
    numTarget = numel(testDesc.target);
    numRes = numel(testDesc.res);
    
    targetTypes = find(ismember(PTN.test.randomPTNTargetTypes(), testDesc.target));
    
    figures = cell(1,numTarget*numRes);
    
    for i=1:numTarget
        for j=1:numRes
            figIdx = sub2ind([numTarget numRes], i, j);
            figures{figIdx} = figure(figIdx);
            set(figures{figIdx}, 'Visible', 'off');
            clf;
            
            plotDesc = cell(numSamplePoints, 1);
    
            points = zeros(testDesc.maxIter, numSamplePoints);

            for k=1:numSamplePoints
                thisKL = cell(testDesc.nTrials, 1);
                [thisKL{:}] = results([results.target] == targetTypes(i) ...
                    & [results.nSamples] == testDesc.nSamples(k) ...
                    & [results.res] == testDesc.res(j)).iterKL;
                % If converged early, replicate final KL value
                for t=1:testDesc.nTrials
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
            ylabel(ax, 'Mean KL to target');
            title(ax, sprintf('%s mean KL vs number of iterations, targets generated via %s, res=%d\n(%d trials per condition)', qName, testDesc.target{i}, testDesc.res(j), testDesc.nTrials));
        end
    end
    
    % Display mode
    if nargout == 0
        for i=1:numel(figures)
            set(figures{i}, 'Visible', 'on');
        end
    end
end

