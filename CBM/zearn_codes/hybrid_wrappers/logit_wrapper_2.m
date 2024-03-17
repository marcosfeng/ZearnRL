function [loglik] = logit_wrapper_2(parameters, subj)
%LOGIT_WRAPPER_2 
%   Outcome variable: NNDSVD_student1
%   Action variable: NNDSVD_teacher2
%   State variables: NNDSVD_student3 
subj.outcome = subj.NNDSVD_student1;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student3];
loglik = logit_states_model(parameters, subj);
end