function out = worker(jobStruct,env,id)
%function out = worker(jobStruct,env,id)
% Worker function that performs cross-validation fit for 
% multi task logistic regression penalized with l1, nuclear norm and
% ridge (l2 squared)
%
%   Input:
%      jobStruct: a structure containing description of the job
%                 jobStruct.params contains lambda, mu, gamma weights for
%                 the l1,nuclearnorm,ridge penalties
%      env:       Environment used in fitting the task
%                 env.x         predictor matrix
%                 env.fnames    predictor names cell array
%                 env.grps      mapping samples to labels
%
ct = 0;
folds = 3;
rand('seed',1);
fullIndices = crossvalind('Kfold',env.grps,folds);
for i=1:4
    grpi = find(env.grps==i);
    ni = length(grpi);
    for j=i+1:4
        grpj = find(env.grps==j);
        nj = length(grpj);
        base(i,j) = min(ni,nj)/(ni + nj);
        base(j,i) = base(i,j);
        ct = ct+1;
        Xs{ct} = [env.x(:,grpi) env.x(:,grpj)]';
        ys{ct} = [0*ones(length(grpi),1); 1*ones(length(grpj),1)];
        indices{ct} = fullIndices([grpi,grpj]);
        usable = find(~isnan(sum(Xs{ct},2)));
        Xs{ct} = Xs{ct}(usable,:);
        ys{ct} = ys{ct}(usable);
        ns(ct) = length(usable);
        indices{ct} = indices{ct}(usable);
        taskName{ct} = [env.grpNames{i} ' vs ' env.grpNames{j}];
    end
end

lambdas = zeros(size(env.x,1),ct);
mu = 0;

if length(jobStruct.params) == ct+2
    mu = jobStruct.params(ct+1);
    gamma = jobStruct.params(ct+2);
else
    mu = jobStruct.params(2);
    gamma = jobStruct.params(3);
end

for ci=1:ct
    if length(jobStruct.params) == ct + 2
        lambdas(:,ci) = jobStruct.params(ci);
    else
        if length(jobStruct.params) == 3
            lambdas(:,ci) = jobStruct.params(1);
        else
            error('jobStruct has wrong number of parameters');
        end
    end
end

for k=1:folds
    for ci=1:ct
        Xtrain{ci} = Xs{ci}(indices{ci}~= k,:);
        ytrain{ci} = ys{ci}(indices{ci} ~= k,:);
        Xtest{ci} = Xs{ci}(indices{ci} == k,:);
        ytest{ci} = ys{ci}(indices{ci} == k,:);
        kns(ci) = length(find(indices{ci} == k));
    end
    
    
    rho = 1;
    
    ret = solveLogRegMTL1NN(ytrain,Xtrain,ytest,Xtest, ...
        lambdas, ...
        mu,gamma,rho,[],id<0);
    if abs(ret.pd)>1e-4
        rho = rho/2;
        fprintf('primal-dual: %d rescaling rho, new rho is: %d\n',abs(ret.pd),rho)
    else
        fprintf('primal-dual: %d done',abs(ret.pd))
    end
    
    
    if abs(ret.pd)>1e-4
        save(['dbg' id]);
        error(['no convergence for:' num2str(jobStruct.params,'%2.4g ')])
    end
    err(k,:) = ret.err;
    kerr(k,:) = ret.err./kns;
end
out.params = jobStruct.params;
out = solveLogRegMTL1NN(ys,Xs,[],[],lambdas,mu,gamma,rho,[],id<0);
out.err = sum(err)./ns;
out.kerr = kerr;
out.taskName = taskName;