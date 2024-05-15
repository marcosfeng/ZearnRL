function [loglik] = hybrid_wrapper_1(parameters, subj)
% Outcome variable: NNDSVD_student2
subj.outcome = subj.NNDSVD_student2;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
loglik = hybrid_ql_logit(parameters, subj);
end