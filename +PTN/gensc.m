function [lambda, logLambda] = gensc(shape, stoch, res)
% Generate a simplicial construction of the given shape and stochasticity.
% The result is a cell array such that each block in the stack is a qps
% with a simplicial quantization, and the overall tensor has the desired
% stochasticity.

    if nargin < 3
        res = 3;
    end
    
    unStoch = setdiff(1:numel(shape), stoch);
    blockShape = shape(unStoch);
    qpsShape = shape(stoch);
    
    if numel(blockShape) == 0
        if nargout > 1
            [lambda, logLambda] = PTN.genqps(qpsShape, res);
            lambda = {lambda};
            logLambda = {logLambda};
        else
            lambda = PTN.genqps(qpsShape, res);
            lambda = {lambda};
        end
        return;
    elseif numel(blockShape) == 1
        blocks = cell(blockShape, 1);
        logBlocks = cell(blockShape, 1);
    else
        blocks = cell(blockShape);
        logBlocks = cell(blockShape);
    end
    
    for i=1:numel(blocks)
        if nargout > 1
            [blocks{i}, logBlocks{i}] = PTN.genqps(qpsShape, res);
        else
            blocks{i} = PTN.genqps(qpsShape, res);
        end
    end
 
    lambda = blocks;
    
    if nargout > 1
        logLambda = logBlocks;
    end
end