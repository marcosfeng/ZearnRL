function [loglik] = logit_wrapper_36s(parameters, subj)
subj.outcome = subj.NNDSVD_student3;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student1 , subj.NNDSVD_student4 ];
loglik = logit_states_model(parameters, subj);
end