function [loglik] = logit_wrapper_3s(parameters, subj)
%   Outcome variable: NNDSVD_student4
%   Action variable: NNDSVD_teacher2
%   State variables: NNDSVD_student1 
subj.outcome = subj.NNDSVD_student1;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student4];
loglik = logit_states_model(parameters, subj);
end