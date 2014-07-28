
load envAgeCMV2-publish

job.params = [0.07,0.02,0.05]; 

out = worker(job,env,0)

for i=1:length(out.err)
    fprintf('%s : %4.4g%%\n',out.taskName{i},100*(1-out.err(i)));
end