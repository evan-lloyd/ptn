function [ trialDesc ] = parse_PTN_test_description( testDesc )
% Parse a PTN test description structure into a list of test conditions.
    % Linearize cross of test conditions, so resuming is simply a matter of
    % advancing to the next index in the collection.
    % Each row is a tuple (resolution, target, nSamples, maxIter)
    types = find(ismember(PTN.test.randomPTNTargetTypes(), testDesc.target));
    desc = cell(1, 4);
    [desc{:}] = ndgrid(testDesc.maxIter,...
                       testDesc.nSamples,...
                       types,...
                       testDesc.res ...
                       );
    trialDesc = cell2mat(cellfun(@(x) x(:), fliplr(desc), 'UniformOutput', false));
    trialDesc = trialDesc(repmat(1:size(trialDesc,1), testDesc.nTrials, 1),:);
end