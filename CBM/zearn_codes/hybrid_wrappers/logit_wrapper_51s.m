function [loglik] = logit_wrapper_51s(parameters, subj)
subj.outcome = subj.NNDSVD_student4;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student2 , subj.NNDSVD_student3 ];
loglik = logit_states_model(parameters, subj);
end