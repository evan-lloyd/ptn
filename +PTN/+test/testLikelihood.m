function [ ] = testLikelihood( Q, logTheta, qps)
    
    qpsNode = zeros(1, Q.nQPS);
    qpsOffset = zeros(1, Q.nQPS);
    curQPS = 1;
    for i=1:Q.n
        if isempty(Q.fixedParameters{i})
            for j=1:numel(Q.lambda{i})
                qpsNode(curQPS) = i;
                qpsOffset(curQPS) = j;

                curQPS = curQPS + 1;
            end
        end
    end
    
    node = qpsNode(qps);
    offset = qpsOffset(qps);
    unStoch = Q.unStoch{node};

    % 1-dimensional? (or 0, in which case ignore this later)
    if numel(unStoch) <= 1
        blockIdx = {offset};
    else
        blockIdx = cell(1, numel(unStoch));
        [blockIdx{:}] = ind2sub(Q.size{node}(unStoch), offset);
    end
    
    % sanity check likelihood
    nTrials = 100;
    diff = zeros(1, nTrials);
    for chk=1:nTrials
        obs = round(rand()*119)+1;
        obsIdx = cell(1,numel(Q.outcomeShape));
        [obsIdx{:}] = ind2sub(Q.outcomeShape, obs);
        likelihood = PTN.getQPSLikelihood(Q, logTheta, node, offset, obsIdx, blockIdx);
        
        % Check a random parameter from the QPS
         if numel(unStoch) > 0
            idx = num2cell(round(rand(1, prod(Q.size{node}(Q.stoch{node}))*Q.res)) + 1);

            if numel(unStoch) > 1
                target = arrayfun(@(x) Q.logLambda{node}{offset}{x}(idx{:}), ...
                                     1:prod(Q.size{node}(Q.stoch{node})));
                if numel(Q.stoch{node}) > 1
                    target = reshape(target, Q.size{node}(Q.stoch{node}));
                else
                    target = reshape(target, Q.size{node}(Q.stoch{node}), 1);
                end
            else
                target = ...
                    arrayfun(@(x) Q.logLambda{node}{offset}{x}(idx{:}), ...
                                     1:prod(Q.size{node}(Q.stoch{node})));
            end
        else % Fully stochastic, just get a random JPT
            idx = num2cell(round(rand(1, prod(Q.size{node})*Q.res)) + 1);
            if Q.arity(node) > 1
                target = reshape(arrayfun(@(x) Q.logLambda{node}{1}{x}(idx{:}), ...
                                         1:prod(Q.size{node})), Q.size{node});
            else
                target = arrayfun(@(x) Q.logLambda{node}{1}{x}(idx{:}), ...
                                         1:prod(Q.size{node}))';
            end
         end
        thetaIdx = cell(1, Q.arity(node));

        if numel(unStoch) > 0
            [thetaIdx{unStoch}] = ind2sub(Q.size{node}(unStoch), offset);
        end

        thetaIdx(Q.stoch{node}) = repmat({':'}, 1, numel(Q.stoch{node}));
        logTheta{node}(thetaIdx{:}) = target;
        trueLikelihood = PTN.do_log_eval(Q, logTheta);
        diff(chk) = abs(trueLikelihood(obs) - likelihood(idx{:}));
        if diff(chk) > 1e-12
            blockIdx
        end
    end
    median(diff)
    mean(diff)
    max(diff)
end

