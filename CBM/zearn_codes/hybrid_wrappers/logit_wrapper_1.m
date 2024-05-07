function [loglik] = logit_wrapper_1(parameters, subj)
%LOGIT_WRAPPER_1
% Outcome variable: NNDSVD_student1
subj.outcome = subj.NNDSVD_student1;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
loglik = logit_model(parameters, subj);
end