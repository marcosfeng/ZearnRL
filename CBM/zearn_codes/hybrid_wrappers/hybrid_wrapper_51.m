function [loglik] = hybrid_wrapper_51(parameters, subj)
%HYBRID_WRAPPER_51 Summary of this function goes here
%   Outcome variable: NNDSVD_student4
%   Action variable: NNDSVD_teacher2
%   State variables: NNDSVD_student2 NNDSVD_student3 
subj.outcome = subj.NNDSVD_student4;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student2 , subj.NNDSVD_student3];
loglik = hybrid_ac_logit(parameters, subj);
end