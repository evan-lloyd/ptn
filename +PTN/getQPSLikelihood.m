function [ update ] = getQPSLikelihood( Q, logTheta, node, offset, obsIdx, blockIdx )
    % Likelihood for this QPS depends on products of its own lambda
    % with adjacent nodes in the graph, modulo a constant term
    % for non-adjacent nodes dependent only on their parameter
    % estimates.
    pdtShape = Q.lambda{node}{offset}{1}.n';
    nodeIdx = repmat({':'}, 1, Q.arity(node));
    obsWays = Q.outcomeWays(Q.nodeOutcomes{node});
    contractionWays = Q.contractionWays(Q.nodeContractions{node});

    stoch = Q.stoch{node};
    unStoch = Q.unStoch{node};
    
    nodeIdx(obsWays) = obsIdx(Q.nodeOutcomes{node});

    % Get terms to logsumexp together for update on this
    % observation
    constantTerm = -Inf;
    qpsTerms = cell(0);
    for q=1:prod(Q.size{node}(contractionWays))
        [nodeIdx{contractionWays}] = ind2sub(Q.size{node}(contractionWays), q);

        % Compute subnetwork excluding this node, and flattening
        % internal edges referencing this qps
        subEval = PTN.do_log_subtn_eval(Q, logTheta, obsIdx, node, nodeIdx);

        % 0 probability; won't contribute to likelihood so
        % skip it
        if subEval == -Inf
            continue;
        end

        % Is this QPS included in the contraction?
        if numel(unStoch) == 0 || isequal(nodeIdx(unStoch), blockIdx)
            qpsTerms{end+1} = Q.logLambda{node}{offset}{nodeIdx{stoch}} + subEval;
        else
            % Update constant term with the subnetwork
            % evaluation
            constantTerm = PTN.logsumexp_amen([constantTerm, logTheta{node}(nodeIdx{:}) + subEval]);
        end
    end
    if constantTerm > -Inf
        qpsTerms{end+1} = tt_ones(pdtShape) * constantTerm;
    end

    % Get Bayesian update tensor train
    if ~isempty(qpsTerms)
        update = amen_cross(qpsTerms, @PTN.logsumexp_amen, 1e-14, 'nswp', 49, 'vec', true, 'verb', 0);
    else
        update = {};
    end
end

