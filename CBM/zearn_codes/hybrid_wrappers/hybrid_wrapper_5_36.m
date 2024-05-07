function [loglik] = hybrid_wrapper_5_36(parameters, subj)
% Outcome variable: NNDSVD_student3
subj.outcome = subj.NNDSVD_student3;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
% State variables: NNDSVD_student1 NNDSVD_student4 
subj.state = [subj.NNDSVD_student1 , subj.NNDSVD_student4 ];
loglik = hybrid_ql_ac(parameters, subj);
end