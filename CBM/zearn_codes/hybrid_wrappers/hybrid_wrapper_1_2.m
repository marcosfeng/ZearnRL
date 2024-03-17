function [loglik] = hybrid_wrapper_1_2(parameters, subj)
%HYBRID_WRAPPER_1_2 Summary of this function goes here
%   Outcome variable: NNDSVD_student1
%   Action variable: NNDSVD_teacher2
%   State variables: NNDSVD_student3 
subj.outcome = subj.NNDSVD_student1;  
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student3];
loglik = hybrid_ql_ac(parameters, subj);
end