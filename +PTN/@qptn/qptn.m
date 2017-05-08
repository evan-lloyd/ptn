classdef qptn
    % qptn Class to encapsulate a Quantized Probabilistic Tensor Network
    %   Detailed explanation goes here
    
    properties
        n
        arity
        size
        edge
        lambda
        logLambda
        nodeOutcomes
        outcomeNodes
        outcomeWays
        outcomeShape
        contractionWays
        contractionNodes
        nodeContractions
        res
        stoch
        unStoch
        nQPS
        fixedParameters
    end
    
    methods
        function q=qptn(a, s, e, stoch, fixedParameters, res)
            if isempty(fixedParameters)
                fixedParameters = repmat({{}}, numel(a), 1);
            end
            q.fixedParameters = fixedParameters;
            q.n = numel(a);
            q.edge = symEdges(e);
            q.arity = a;
            q.size = s;
            q.lambda = cell(q.n, 1);
            q.logLambda = cell(q.n, 1);
            q.res = res;
            q.stoch = stoch;
            q.unStoch = cell(q.n, 1);
            for i=1:q.n
                if isempty(q.fixedParameters{i})
                    q.unStoch{i} = setdiff(1:q.arity(i), q.stoch{i});
                end
            end
            
            % Build simplicial constructions for each (non-fixed) node
            for i=1:q.n
                if isempty(q.fixedParameters{i})
                    [q.lambda{i}, q.logLambda{i}] = PTN.gensc(q.size{i}, q.stoch{i}, q.res);
                end
            end
            
            q.nQPS = sum(arrayfun(@(x) numel(q.lambda{x}), 1:q.n));
            
            q.outcomeNodes = [];
            q.contractionNodes = [];
            q.outcomeWays = [];
            q.contractionWays = [];
            q.nodeOutcomes = cell(1, q.n);
            q.nodeContractions = cell(1, q.n);
            q.outcomeShape = [];
            
            curOutcome = 0;
            curContraction = 0;
            for i=1:q.n
                outWays = 1:q.arity(i);
                outWays(q.edge(q.edge(:,1) == i, 2)) = [];
                inWays = setdiff(1:q.arity(i), outWays);
                
                q.outcomeNodes = [q.outcomeNodes repmat(i, 1, numel(outWays))];
                q.contractionNodes = [q.contractionNodes repmat(i, 1, numel(inWays))];
                
                q.outcomeWays = [q.outcomeWays outWays];
                q.contractionWays = [q.contractionWays inWays];
                
                q.nodeOutcomes{i} = curOutcome+(1:numel(outWays));
                q.nodeContractions{i} = curContraction+(1:numel(inWays));
                
                curOutcome = curOutcome + numel(outWays);
                curContraction = curContraction + numel(inWays);
                q.outcomeShape = [q.outcomeShape q.size{i}(outWays)];
            end
            
            % enforce constraints
            %   - TN constraints
            %   - PTN constraints
            %   - output ways after all internal ways
        end
    end
    
end

function [e] = symEdges(e)
    newEdges = zeros(size(e, 1), 4);
    for i=1:size(e,1);
        edge = e(i,:);
        newEdges(i,:) = [edge(3) edge(4) edge(1) edge(2)];
    end
    e = [e; newEdges];
    [~,idx] = unique(e, 'rows');
    e = e(idx, :);
end

