function [loglik] = hybrid_wrapper_43(parameters, subj)
%HYBRID_WRAPPER_43
%   Outcome variable: NNDSVD_student4
%   Action variable: NNDSVD_teacher2
%   State variables: NNDSVD_student1
subj.outcome = subj.NNDSVD_student4;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student1];
loglik = hybrid_ac_logit(parameters, subj);
end