function [ ] = test_ptn_learn(testDescriptionPath, outputPath, resume, parallel)
    % -------------------------------------------------------------------------
    % Purpose: Test ptn_learn's accuracy as a function of sample size. For
    % every combination of the given testing conditions, and repeating for the
    % given number of trials, runs ptn_learn with a random target distribution
    % generated for that trial with the method specified in the trial
    % condition, and outputs statistics on the algorithm's performance for
    % each trial.
    % 
    % Inputs:
    % testDescriptionPath -- path to a JSON file describing the test to run,
    % with the following fields:
    %   QPTN     -- JSON description of the QPTN to learn
    %   nTrials  -- number of trials to run for each condition
    %   maxIter  -- maximum number of alternating iterations
    %   nSamples -- list of sample numbers to use
    %   res      -- list of QPS resolutions to use (positive integers)
    %   target   -- subset of ["randIndex", "randSimplex", "randJPT"]
    % outputPath -- path to a file to write results, which will contain on the
    % first line the test description above, and on each subsequent line a JSON
    % object with the following fields:
    %   condition -- [nSamples, target, res]
    %   KL        -- KL divergence to target distribution for this trial
    %   sampleKL  -- KL divergence from quantized sample distribution to target
    %                for trial
    % resume -- Boolean, append any missing test conditions to the end of the
    % output file if it already exists and has the same test description
    % parallel -- Boolean, use the Parallel Computing Toolbox to speed up PTN
    % learning?
    %
    %
    % Outputs:
    % Writes (or appends) to the file in outputPath the results in the format
    % specified above.
    % -------------------------------------------------------------------------

    if nargin < 4
        parallel = false;
    end
    
    testDesc = loadjson(testDescriptionPath);
    
    % Try not to overwrite existing files
    if exist(outputPath, 'file') && ~resume
        error('Output file exists, use resume=true to continue.');
    end
    
    % If appending, verify test conditions match the input, and initialize
    % to append the missing trials.
    if resume
        if ~exist(outputPath, 'file')
            error('Output file doesn''t exist, can''t append output.');
        end
        outFile = fopen(outputPath, 'rb');
        headerLine = fgetl(outFile);
        
        try
            oldDesc = loadjson(headerLine);
        catch
            error('Error parsing existing output file.');
        end
        
        if ~isequal(oldDesc, testDesc)
            error('Input test description and output file to resume don''t match, aborting.');
        end
        
        % Get number of lines in file to initialize appending
        fseek(outFile, 0, 'eof');
        outFileSize = ftell(outFile);
        frewind(outFile);
        outData = fread(outFile, outFileSize, 'uint8');
        
        % Count newlines, plus one for final line
        numLines = sum(outData == 10) + 1;
        
        % A file with only a header has 1 line, and the current trial is
        % the first.
        curTrial = numLines;
        
        fclose(outFile);
        
        fprintf('Resuming at test #%d\n', curTrial);
    else
        curTrial = 1;
        outFile = fopen(outputPath, 'w');
        fprintf(outFile, '%s', savejson('', testDesc, 'ArrayToStruct', 1, 'Compact', 1));
        fclose(outFile);
    end
    
    % Initialize test description
    trialDesc = PTN.test.parse_PTN_test_description(testDesc);
    targetTypes = PTN.test.randomPTNTargetTypes();
    
    if isfield(testDesc, 'testAlternatingIterationKL')
        testAlternatingIterationKL = testDesc.testAlternatingIterationKL;
    else
        testAlternatingIterationKL = false;
    end
        
    
    % Run tests
    lastRes = 0;
    while curTrial <= size(trialDesc, 1)
        fprintf('Begin trial %d of %d (%d, %d, %d, %d)...\n', curTrial, size(trialDesc, 1), trialDesc(curTrial,1), trialDesc(curTrial,2), trialDesc(curTrial,3), trialDesc(curTrial, 4));
        res = trialDesc(curTrial, 1);
        targetGen = targetTypes{trialDesc(curTrial, 2)};
        sampleCount = trialDesc(curTrial, 3);
        maxIter = trialDesc(curTrial, 4);
        
        % Recompute QPTN when resolution changes
        if res ~= lastRes
            fprintf('Generate QPTN with res=%d\n', res);
            Q = PTN.QPTN_from_description(testDesc.QPTN, res);
            lastRes = res;
            
            if parallel
                parallelBlockSize = prod(Q.outcomeShape);
            else
                parallelBlockSize = 1;
            end
        end
        
        % Generate the target for this trial
        [target, params] = PTN.test.generate_ptn_dist(Q, targetGen);
        
        % Draw samples from target distribution
        samples = PTN.draw_from_JPT(target, sampleCount);
        
        % Quantize sample distribution to the given resolution for a more
        % fair comparison with our method
        sampleDist = reshape((samples + 1) / (sampleCount + numel(target)), size(target));
        sampleRes = res^numel(samples);
        sampleDist = max(round(sampleDist * sampleRes), 1) / sampleRes;
        sampleDist = sampleDist / sum(sampleDist(:));
        
        if testAlternatingIterationKL
            [predictiveDist, parameterEstimates, deltaKL, thetaHist, iterKL, iterParmKL] = PTN.learnPTN_iterKL(Q, samples, parallelBlockSize, maxIter, target, params);
        else
            [predictiveDist, parameterEstimates, deltaKL, thetaHist] = PTN.learnPTN(Q, samples, parallelBlockSize, maxIter);
        end

        % Output stats
        outStats = struct();
        
        outStats.KL = PTN.KL(predictiveDist(:)', target(:)');
        outStats.sampleKL = PTN.KL(sampleDist(:)', target(:)');
        outStats.targetDistribution = target;
        outStats.targetParameters = params;
        outStats.predictiveDist = predictiveDist;
        outStats.parameterEstimates = parameterEstimates;
        outStats.parameterKL = zeros(1, Q.nQPS);
        outStats.samples = samples;
        outStats.deltaKL = deltaKL;
        outStats.thetaHist = thetaHist;
        
        if testAlternatingIterationKL
            outStats.iterKL = iterKL;
            outStats.iterParameterKL = iterParmKL;
        end
        
        
        if ~isempty(params)
            curQPS = 1;
            for i=1:Q.n
                for j=1:numel(Q.lambda{i})
                    outStats.parameterKL(curQPS) = PTN.KL(PTN.extractQPSEstimate(Q, i, j, parameterEstimates{i}), PTN.extractQPSEstimate(Q, i, j, params{i}));
                    curQPS = curQPS + 1;
                end
            end
        end
        
        disp([sprintf('KL: %f', outStats.KL), sprintf('\tsampleKL: %f', outStats.sampleKL)]);
        disp('QPS KL:')
        disp(outStats.parameterKL);
        
        outFile = fopen(outputPath, 'a');
        fprintf(outFile, '\n%s', savejson('', outStats, 'Compact', 1));
        fclose(outFile);
        
        curTrial = curTrial + 1;
    end

end