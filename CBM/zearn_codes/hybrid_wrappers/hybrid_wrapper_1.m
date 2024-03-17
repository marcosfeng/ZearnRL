function [loglik] = hybrid_wrapper_1(parameters, subj)
%HYBRID_WRAPPER_1 Summary of this function goes here
%   Outcome variable: NNDSVD_student1
%   Action variable: NNDSVD_teacher2
subj.outcome = subj.NNDSVD_student1;
subj.action = subj.NNDSVD_teacher2;
loglik = hybrid_ql_logit(parameters, subj);
end