function [loglik] = hybrid_wrapper_7(parameters, subj)
%HYBRID_WRAPPER_1
% Outcome variable: NNDSVD_student4
subj.outcome = subj.NNDSVD_student4;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
loglik = hybrid_ql_logit(parameters, subj);
end