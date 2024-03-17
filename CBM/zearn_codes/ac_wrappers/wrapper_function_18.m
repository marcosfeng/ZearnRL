function [loglik] = wrapper_function_18(parameters, subj)
    % Outcome variable: NNDSVD_student2
    subj.outcome = subj.NNDSVD_student2;
    % Action variable: NNDSVD_teacher3
    subj.action = subj.NNDSVD_teacher3;
    % State variables: NNDSVD_student1 
    subj.state = [subj.NNDSVD_student1 ];
    loglik = actor_critic_model(parameters, subj);
end
