function [loglik, prob, choice] = wrapper_post_10(parameters, subj)
% Outcome variable: NNDSVD_student4
subj.outcome = subj.NNDSVD_student4;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
[loglik, prob, choice] = hybrid_ql_logit_posterior(parameters, subj);
end