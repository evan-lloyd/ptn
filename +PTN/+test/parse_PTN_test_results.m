function [ testDesc, lines ] = parse_PTN_test_results( resultsPath )
% Parse the results of a PTN test. Creates a structure for each trial containing
% the statistics used in the thesis plots.
    resultsFile = fopen(resultsPath, 'r');
    testDesc = loadjson(fgetl(resultsFile));
    
    if ~iscell(testDesc.target)
        testDesc.target = {testDesc.target};
    end
    
    trialDesc = PTN.test.parse_PTN_test_description(testDesc);
    
    if isfield(testDesc, 'testAlternatingIterationKL')
        testAlternatingIterationKL = testDesc.testAlternatingIterationKL;
    else
        testAlternatingIterationKL = false;
    end
    
    lines = struct();
    lines(size(trialDesc,1)).res = [];
    for i=1:numel(lines)
        line = loadjson(fgetl(resultsFile));
        lines(i).KL = line.KL;
        lines(i).sampleKL = line.sampleKL;
        lines(i).res = trialDesc(i,1);
        lines(i).target = trialDesc(i,2);
        lines(i).nSamples = trialDesc(i,3);
        lines(i).maxIter = trialDesc(i,4);
        lines(i).deltaKL = line.deltaKL;
        lines(i).parameterKL = line.parameterKL;
        if testAlternatingIterationKL
            lines(i).iterKL = line.iterKL;
            lines(i).iterParameterKL = line.iterParameterKL;
        end
    end
    
    fclose(resultsFile);
end

