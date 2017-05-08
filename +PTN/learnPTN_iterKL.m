function [ predictiveDist, theta, deltaKL, thetaHist, iterKL, iterParmKL, pdt ] = learnPTN_iterKL( Q, data, nParallelUpdates, maxIter, targetDist, targetTheta)
    % learn_ptn: Learn a set of Parameter Distribution Tensors over the
    % parameters outlined in the QPTN Q, given a tensor of observation counts.
    %   Return values:
    %   predictiveDist---Predictive distribution (tensor of same shape as data)
    %   pdt---Tensor representing state of belief over quantizations of PTN
    %   parameters (one for each QPS in Q)

    % Max threads is one per observation
    if nParallelUpdates > numel(data)
        nParallelUpdates = numel(data);
    end
    
    if nParallelUpdates < 1
        nParallelUpdates = 1;
    end
    
    % Stopping criteria
    if nargin < 4
        maxIter = 5;
    end
    
    % If literally no change after an iteration, it's safe to quit early,
    % since each later iteration won't change either
    stopKL = 0;
    
    thetaHist = cell(0);
    
    % TT-rank of PDT tensors, to keep memory requirements reasonable
    maxPDTRank = 200;

    % Priors over each individual QPS (possibly multiple per node)
    prior = cell(0);

    % Construct uniform priors for each PDT
    for i=1:Q.n
        if isempty(Q.fixedParameters{i})
            for j=1:numel(Q.lambda{i})
                prior{end+1} = tt_ones(Q.lambda{i}{j}{1}.n') * -log(Q.lambda{i}{j}{1}.n(1)) * Q.lambda{i}{j}{1}.d;
            end
        end
    end

    theta = cell(Q.n, 1);
    logTheta = cell(Q.n, 1);
    qpsNode = zeros(1, numel(prior));
    qpsOffset = zeros(1, numel(prior));

    % Initialize parameter estimates (expected value under priors)
    curQPS = 1;
    for i=1:Q.n
        if ~isempty(Q.fixedParameters{i})
            theta{i} = Q.fixedParameters{i};
            logTheta{i} = log(Q.fixedParameters{i});
            continue;
        end
        
        if Q.arity(i) > 1
            theta{i} = zeros(Q.size{i});
        else
            theta{i} = zeros(Q.size{i}, 1);
        end

        % For each internal index
        for j=1:numel(Q.lambda{i})
            theta{i} = PTN.getParameterEstimate(Q, i, j, prior{curQPS}, theta{i});
            qpsNode(curQPS) = i;
            qpsOffset(curQPS) = j;

            curQPS = curQPS + 1;
        end
        logTheta{i} = log(theta{i});
    end

    thetaHist{1} = theta;
    
    prevTheta = theta;
    pdt = prior;
    
    % Alternating optimization: for each QPS: fix the other QPS's to their
    % estimated value, then get posterior estimate for given observation
    % counts.
    deltaKL = [];
    iterKL = [];
    iterParmKL = {};
    for i=1:maxIter
        fprintf('Alternating optimization iteration %d of %d\n', i, maxIter);

        for j=1:Q.nQPS
            fprintf('Computing posterior for QPS %d...\n', j);
            pdt{j} = prior{j};

            node = qpsNode(j);
            offset = qpsOffset(j);
            unStoch = Q.unStoch{node};

            % 1-dimensional? (or 0, in which case ignore this later)
            if numel(unStoch) <= 1
                blockIdx = {offset};
            else
                blockIdx = cell(1, numel(unStoch));
                [blockIdx{:}] = ind2sub(Q.size{node}(unStoch), offset);
            end

            % Get QPS posterior
            for block=0:ceil(numel(data) / nParallelUpdates)-1
                if (block+1)*nParallelUpdates > numel(data)
                    thisBlock = mod(numel(data), nParallelUpdates);
                else
                    thisBlock = nParallelUpdates;
                end
                updates = cell(1, thisBlock);
                
                % Compute Bayesian updates, running the updates for
                % different observations in parallel
                parfor k=1:thisBlock
                    obsIdx = cell(1,ndims(data));
                    [obsIdx{:}] = ind2sub(size(data), k+block*nParallelUpdates);
                    updates{k} = PTN.getQPSLikelihood(Q, logTheta, node, offset, obsIdx, blockIdx);
                end

                % Apply updates to PDT
                for k=1:thisBlock
                    if ~isempty(updates{k})
                        pdt{j} = pdt{j} + data(k+block*nParallelUpdates) * updates{k};
                        % Prevent exploding tt-rank by rounding
                        pdt{j} = round(pdt{j}, 1e-14, maxPDTRank);
                    end
                end
            end
            theta{node} = PTN.getParameterEstimate(Q, node, offset, pdt{j}, theta{node});
            logTheta{node} = log(theta{node});
        end


        deltaKL(end+1,:) = zeros(1, Q.nQPS);
        for j=1:Q.nQPS
            prevQPS = PTN.extractQPSEstimate(Q, qpsNode(j), qpsOffset(j), prevTheta{qpsNode(j)});
            curQPS  = PTN.extractQPSEstimate(Q, qpsNode(j), qpsOffset(j), theta{qpsNode(j)});
            deltaKL(end, j) = max(PTN.KL(prevQPS, curQPS), 0);
        end
        fprintf('log10 of KL for updated QPS estimates:%s', evalc('log10(deltaKL(end,:))'));

        predictiveDist = exp(PTN.do_log_eval(Q, logTheta));
        iterKL(i) = PTN.KL(predictiveDist, targetDist);
        fprintf('KL to target dist for this iteration: %f\n', iterKL(i));
        
        iterParmKL{i} = zeros(1, Q.nQPS);
        
        curQPS = 1;
        for j=1:Q.n
            for k=1:numel(Q.lambda{j})
                iterParmKL{i}(curQPS) = PTN.KL(PTN.extractQPSEstimate(Q, j, k, theta{j}), PTN.extractQPSEstimate(Q, j, k, targetTheta{j}));
                curQPS = curQPS + 1;
            end
        end
        
        disp('Parameter KL for this iteration:')
        disp(iterParmKL{i});
        
        thetaHist{end + 1} = theta;
        prevTheta = theta;
        
        % Break when maximum change in estimated parameters is small
        if max(deltaKL(end,:)) <= stopKL
            break;
        end
    end

    predictiveDist = exp(PTN.do_log_eval(Q, logTheta));
end

