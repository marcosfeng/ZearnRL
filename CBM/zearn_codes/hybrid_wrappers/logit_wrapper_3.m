function [loglik] = logit_wrapper_3(parameters, subj)
%LOGIT_WRAPPER_3
%   Outcome variable: NNDSVD_student2
%   Action variable: NNDSVD_teacher2
subj.outcome = subj.NNDSVD_student2;
subj.action = subj.NNDSVD_teacher2;
loglik = logit_model(parameters, subj);
end