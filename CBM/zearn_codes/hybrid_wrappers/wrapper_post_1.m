function [loglik, prob, choice] = wrapper_post_1(parameters, subj)
% Outcome variable: NNDSVD_student1
subj.outcome = subj.NNDSVD_student1;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
[loglik, prob, choice] = logit_posterior(parameters, subj);
end