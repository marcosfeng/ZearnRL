function [loglik] = wrapper_function_46(parameters, subj)
    % Outcome variable: NNDSVD_student4
    subj.outcome = subj.NNDSVD_student4;
    % Action variable: NNDSVD_teacher3
    subj.action = subj.NNDSVD_teacher3;
    % State variables: NNDSVD_student1 
    subj.state = [subj.NNDSVD_student1 ];
    loglik = actor_critic_model(parameters, subj);
end
