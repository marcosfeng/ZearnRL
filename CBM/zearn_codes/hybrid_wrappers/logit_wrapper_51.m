function [loglik] = logit_wrapper_51(parameters, subj)
%LOGIT_WRAPPER_51 
%   Outcome variable: NNDSVD_student4
%   Action variable: NNDSVD_teacher2
%   State variables: NNDSVD_student2 NNDSVD_student3 
subj.outcome = subj.NNDSVD_student4;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student2 , subj.NNDSVD_student3];
loglik = logit_states_model(parameters, subj);
end