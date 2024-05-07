function [loglik] = hybrid_wrapper_5(parameters, subj)
%HYBRID_WRAPPER_5
% Outcome variable: NNDSVD_student3
subj.outcome = subj.NNDSVD_student3;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
loglik = hybrid_ql_logit(parameters, subj);
end