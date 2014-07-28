function st = solveLogRegMTL1NN(ys,Xs,valys,valXs,lambda,mu,gamma,rho,old,DEBUG)
% st = solveLogRegMTL1NN(ys,Xs,valys,valXs,lambda,mu,gamma,rho,old,DEBUG)
% Solves Multitask Logistic Regression with regression weight matreix 
% penalized by entrywise L1, nuclear norm and squared Frobenius norm:
%
% minimize_W  sum_i 1/n_i*(- < y_i, X_i*w_i > + sum(log(1 + exp(X_i w_i)))  
%             + lambda||W(:)||_1 + mu||W||_* + gamma/2||W||_F^2
%
% Input: 
%   ys          cell array of binary response vectors, one per task. Used
%               for training
%   Xs          cell array of predictor matrices, one per task. Used for
%               training
%   valys       cell array of binary response vectors, one per task. Used
%               for validation
%   valXs       cell array of predictor matrices, one per task. Used for
%               validation
%   lambda      penalty weight for entrywise L1 on W
%   mu          penalty weight for nuclear norm on matrix W
%   gamma       penalty weight on sum of squares of entries of W
%   rho         augmented lagrangian factor
%   old         old state used as warm start for the optimization
%   DEBUG       set to 1 to enable trace and plotting
% Copyright 2012, Vladimir Jojic

t = length(Xs);
assert(length(ys) == t);
p = size(Xs{1},2);
n = zeros(t,1);
nu = 1.1; tauInc = 1.1; tauDec = 1.1;

if ~exist('DEBUG','var')
    DEBUG = 0;
end
USEQSVD = 0;

if length(lambda)==1
    lambda = lambda*ones(p,t);
end

for ti=1:t
    assert(size(Xs{ti},2) == p);
    n(ti) = size(Xs{ti},1);
    [Xs{ti},centers{ti},scales{ti}] = standardize(Xs{ti});
    assert(size(ys{ti},1) == n(ti));
    if ~isempty(valys)
        assert(size(valys{ti},1) == size(valXs{ti},1));
        assert(size(valXs{ti},2) == p);
    end
end

if ~exist('old','var') || isempty(old)
    U0 = cell(t,1);
    for ti=1:t
        U0{ti} = 0.5*ones(n(ti),1);
    end
    U1 = 0*ones(p,t);
    U2 = 0*ones(p,t);
    U3 = 0*ones(p,t);
    W = zeros(p,t);
    alphas = zeros(1,t);
else
    U0 = old.U0;
    U1 = old.U1;
    U2 = old.U2;
    U3 = old.U3;
    W = old.W;
    alphas = old.alphas;
end

mask1 = zeros(p,t);
mask2 = zeros(p,t);
mask3 = zeros(p,t);

mask1(find(lambda(:))==0) = 1;
mask2(:) = (mu==0);
mask3(:) = (gamma==0);

U1(find(mask1(:)==0)) = 0;
U2(find(mask2(:)==0)) = 0;
U3(find(mask3(:)==0)) = 0;

if all(lambda(:)==0) && all(gamma(:)==0) && all(mu(:)==0)
    fprintf('Skipping no penalty problem!');
    primal = Inf;
    dual = -Inf;
    it = 0;
else
    tic
    for it=1:5000
        prevU1 = U1; prevU2 = U2; prevU3 = U3;
        
        %%%% U0 update
        for ti=1:t
            while 1
                X = Xs{ti}; w = W(:,ti); y = ys{ti}; alpha = alphas(ti);
                u0 = U0{ti}; u1 = U1(:,ti); u2 = U2(:,ti); u3 = U3(:,ti);
                X1 = [ones(length(y),1) X];
                
                for ni=1:n(ti)
                    U0{ti}(ni) = coordLogRegSc(y(ni),...
                        1/n(ti),...
                        X1(ni,:)',...
                        X1([1:ni-1 ni+1:end],:)'*U0{ti}([1:ni-1 ni+1:end]) + [0;u1] + [0;u2] + [0;u3] - 1/rho*[alpha;w],...
                        rho);
                end
                d(ti) = norm(u0 - U0{ti},1);
                if d(ti)<1e-2
                    break;
                end
            end
        end
        
        %%%% U1 update
        for ti=1:t
            X = Xs{ti}; w = W(:,ti);
            u0 = U0{ti}; u2 = U2(:,ti); u3 = U3(:,ti);
            U1(:,ti) = -(X'*u0+u2+u3-1/rho*w);
        end
        U1 = min(abs(U1),lambda).*sign(U1);
        
        %%%% U2 update
        for ti=1:t
            X = Xs{ti}; w = W(:,ti);
            u0 = U0{ti}; u1 = U1(:,ti); u3 = U3(:,ti);
            U2(:,ti) = -(X'*u0+u1+u3-1/rho*w);
        end
        if ~USEQSVD
            [U,S,V] = svd(U2);
        else
            [U,S,V] = qsvd(U2,1e-10);
        end
        assert(all(S(:))>=0);
        nonz = find(diag(S)>0);
        U2 = U(:,nonz)*min(S(nonz,nonz),mu)*V(:,nonz)';
        
        %%%% U3 update
        if gamma ~= 0
            II = [eye(p,p);sqrt(gamma*rho)*eye(p,p);];
            for ti=1:t
                X = Xs{ti}; w = W(:,ti);
                u0 = U0{ti}; u1 = U1(:,ti); u2 = U2(:,ti);
                B = [zeros(p,1);-sqrt(gamma*rho)*(X'*u0 + u1 + u2 - 1/rho*w)];
                U3(:,ti) =  II\B;
            end
        end
        
        %%%% updates of W and alpha
        for ti=1:t
            X = Xs{ti};
            u0 = U0{ti}; u1 = U1(:,ti);  u2 = U2(:,ti); u3 = U3(:,ti);
            alphas(ti) = alphas(ti) - rho*(ones(n(ti),1)'*u0);
            W(:,ti) = W(:,ti) - rho*(X'*u0 + u1 + u2 + u3);
        end
        
        %%%% primal and dual residual computation
        for ti=1:t
            X = Xs{ti};
            u0 = U0{ti}; u1 = U1(:,ti);  u2 = U2(:,ti); u3 = U3(:,ti);
            s0{ti} = rho*X*(U1(:,ti) - prevU1(:,ti) + U2(:,ti) - prevU2(:,ti) + U3(:,ti) - prevU3(:,ti));
            r(:,ti) = [ones(n(ti),1)'*u0;X'*u0 + u1 + u2 + u3];
        end
        
        % if penalty is zero dual variable is clamped at zero and there is no
        % residual, the mask prevents overcounting of residuals.
        s1 = rho*mask1.*(U3 - prevU3 + U2 - prevU2);
        s2 = rho*mask2.*(U3 - prevU3);
        
        prRes = norm(r,'fro');
        duRes = norm(s1,'fro') + norm(s2,'fro');
        for ti=1:t
            duRes = duRes + norm(s0{ti},'fro');
        end
        
        
        
        dual = 0;
        for ti=1:t
            u0 = U0{ti}; y = ys{ti};
            dual = dual + 1/n(ti)*((n(ti)*u0+y)'*safelog(n(ti)*u0+y) +  (1-n(ti)*u0-y)'*safelog(1-n(ti)*u0-y));
        end
        
        if gamma ~= 0
            dual = dual + 1/(2*gamma)*sum(sum(U3.^2));
        end
        
        primal = 0;
        for ti=1:t
            X = Xs{ti}; w = W(:,ti); y = ys{ti}; alpha = alphas(ti);
            primal = primal + 1/n(ti)*(-sum(y'*(alpha + X*w)) + sum(log(1 + exp(alpha + X*w))));
            primal = primal + sum(abs(lambda(:,ti).*w));
            primal = primal + gamma/2*sum(w.^2);
        end
        if ~USEQSVD
            [~,S,~] = svd(W);
        else
            [~,S,~] = qsvd(W,1e-10);
        end
        primal = primal + mu*sum(abs(diag(S)));
        
        if DEBUG
            prs(it) = primal;
            dus(it) = -dual;
        end
        nnw(it) = sum(abs(diag(S)));
        
        if ~isfinite(primal) || ~isfinite(dual)
            error('primal or dual is not finite')
        end
        if mod(it,10)==0
            toc
            tic
            fprintf(' %d ',it);
            fprintf('\t primal:%d dual:%d prRes:%d duRes:%d rho:%d\tnz:%d\n',primal,dual,prRes,duRes,rho,sum(sum(abs(W)>1e-4)));
            if DEBUG
                subplot(3,1,1);
                plot(log2(abs(prs-dus)),'r');
                subplot(3,1,2);
                plot(prs,'r');
                hold on
                plot(dus,'b');
                hold off
                subplot(3,1,3);
                plot(nnw,'r');
                drawnow
            end
        end
        if abs(primal + dual) < 1e-6 && it>100
            break;
        end
        
        if prRes > nu*duRes
            rho = rho*tauInc;
        else
            if duRes > nu*prRes
                rho = rho/tauDec;
            end
        end
    end
end
fprintf('\n');
if ~isempty(valys)
    for ti=1:t
        valy = valys{ti}; valX = valXs{ti};
        valX = (valX - repmat(centers{ti},[size(valX,1) 1]))./repmat(scales{ti},[size(valX,1) 1]);
        w = W(:,ti);
        alpha = alphas(ti);
        pred = (alpha + valX*w) - log(1 + exp(alpha + valX*w)) > log(0.5);
        st.err(ti) = sum((pred>0).*(valy==0) + (pred<=0).*(valy==1));
    end
end

st.lambda = lambda;
st.mu = mu;
st.gamma = gamma;
st.U0 = U0;
st.U1 = U1;
st.U2 = U2;
st.U3 = U3;
st.W = W;
st.alphas = alphas;
st.iters = it;
st.pd = primal + dual;

function r=safelog(a)
r = log(a).*(a > 1e-50) + (-400).*(a<1e-50);

function r = logsum(a,b)
m = max(a,b);
a = a - m;
b = b - m;
r = log(exp(a) + exp(b))+m;

function [X,m,s] = standardize(X)
% standardizes each column of input matrix to be 0-mean and 1-variance
for i=1:size(X,2)
    m(i) = nanmean(X(:,i));
    s(i) = nanstd(X(:,i));
    X(:,i) = (X(:,i) - m(i))/s(i);
end


function u = coordLogRegSc(y,sc,b,c,rho)
% solve scaled max entropy problem
%
% min sc*((1/sc*u+y)log(1/sc*u+y) + (1-1/sc*u-y)log(1-1/sc*u-y)) + rho/2*||b*u + c||^2
% u,y,sc are scalars
% b,c are vectors
%
% e.g.   sc=1/n b = X(i,:) c = sum_{r=0,N) P_r^T u_r - 1/rho*w

if y==0
    u = sc/2;
    c1 = 0; c2 = sc;
else
    u = -sc/2;
    c1 = -sc; c2 = 0;
end
isc = 1/sc;
g = 1;gg=1;
it = 0;
s = 1;
while (abs(g/gg)>1e-10 && it<100)
    prevu = u;
    g = log(isc*u+y) - log((1-y) - isc*u) + rho*(c + b*u)'*b;
    gg = isc/((isc*u+y)*((1-y)-isc*u)) + rho*(b'*b);    
    if ~isfinite(g) || ~isfinite(gg) || ~isreal(g)
        error('derivative computation failed');
    end
    u = u - s*g/gg;
    if u>=c2
        u = 0.5*prevu + 0.5*c2;
    else
        if u<=c1
            u = 0.5*prevu + 0.5*c1;
        end
    end
    if ~isreal(u) || u>1 || u<-1
        error('out of bounds');
    end    
    it = it+1;
end

if it==100, fprintf(' line search did not terminate\n'); end;
assert(u<=1 && u>=-1)

