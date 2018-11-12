% SSN
%% SET PARAMS (FOR FIGURE 3)
N_e = 4000; % Number of excitatory units
N_i = 1000; % Number of inhibitory units
T_e = 20; % in ms, Membrane time constant for excitatory units
T_i = 10; % in ms, Membrane time constant for inhibitory units
V_rest = -70; % in mV, Resting membrane potential
V_0 = -70; % in mV, Rectification threshold potential
k = 0.3; % in mV^-n, Nonlinearity gain
n = 2; % Nonlinearity exponent (Powerlaw for nonlinearity input/output function)
W_ee = 1.25; % in mV*s, E->E connection weight(or sum therof)
W_ie = 1.2; % in mV*s, E->I connection weight (or sum therof)
W_ei = 0.65; % in mV*s, E->I connection weight (or sum therof)
W_ii = 0.5; % in mV*s, I->I connection weight (or sum therof)
T_noise = 50; % in ms; Noise correlation time constant
sigma_0e = 1; % in mV; Noise standard deviation (excitatory neurons)
sigma_0i = 0.5; % in mV; Noise standard deviation (inhibitory neurons)
p_e = 0.1; % Outgoing connection probabilty (excitatory neurons)
p_i = 0.4; % Outgoing connection probability (inhibitory neurons)
T_syn = 2; % in ms; synaptic time constants
delta = 0.5;% in ms; axonal time delay
iters= 100; % Number of timesteps to iterate over
%% INPUT NOISE FUNCTION - 20% COVARIANCE AND USE WEIGHTED REVERSION TO MEAN AT EACH ITERATION
% Input Noise private, but input h common
%%%%%%%%%%%%%%%%%%%%%%%%%% IMPOSE .2 CORR COEFF FOR INPUT NOISE!
%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP TO GET NOISE FOR EACH CELL
p_cov = .2;
% Figure 2: Noise terms choosen to be uncorrelated
% Figure 3: Input noise covariance uniform
        % Covar_noise = sigma_noise^2[d_ij(1-p)+p]
            % sigma_noise -> sigma_0e or sigma_0i as specified
            % pairwise correlation coeff p = 0.2
% iters= 100; % Number of timesteps to iterate over 
% n_i(t) is the noise term; Input Noise 
    % Generated using orn/uhl process in above section
    % Instead use perturbations that revert to mean with a reversion
    % constant similar to that used for orstein-uhlenbeck process
% T_noise = 50; % in ms; Noise correlation time constant
    % External noise had a constant of 50ms
    % so with 30ms timesteps- reversion coefficient= 3/5;
    revert = 30/50; % Since membrane time constant 50ms and timestep 30ms, noise should revert over window ~3/5 of previous timestep perturbation (linear assumption)
% sigma_0e = 1; % in mV; Noise standard deviation (excitatory neurons)
% sigma_0i = 0.5; % in mV; Noise standard deviation (inhibitory neurons)
% for first step use mean input
n_e(: , i) = mvnrnd(0, p_cov, N_e)*(sigma_0e/.5); % For initial step, equal to single perturbation
n_i(:, i) = mvnrnd(0, p_cov, N_i)*(sigma_0i/.5); 

for i = 2:iters
    n_e(:, i) = (1 - revert) .* n_e(:, i-1) + mvnrnd(0, p_cov, N_e)*(sigma_0e/.5); %Draw perturbation from guassian with mean 0 and stdev ,1
    n_i(:, i) = (1 - revert) .* n_i(:, i-1) + mvnrnd(0, p_cov, N_i)*(sigma_0i/.5); % %Draw perturbation from guassian with mean 0 and stdev 1
end

% impose .2 corr coeff
figure, imagesc(cov(n_e))
title('Covariance excitatory noise')
figure, imagesc(cov(n_i))
title('Covariance inhibitory noise')

%% EQUATION 2- UPDATE MEMBRANE POTENTIAL FOR EXCITATORY AND INHIBITORY UNITS @ EACH TIMESTEP
%%%%%%%%%%%%%%%%%%%%%%%% NEED TO DETERMINE INTIALIZATION VALUES FOR N_e AND N_i unts

% iters= 100; % Number of timesteps to iterate over
 
h = ones(1, iters)*5; 
% h(t) is the stimulus input that is common to both excitatory and
% inhibitory units
    % Input constant, range for "Figure 2 : 0-20 and Figure 3: 0-15
    % Potentially time-varying but deterministic component (the mean)
% n_i(t) is the noise term; Input Noise
    % Generated using orn/uhl process in above section
    % Instead use perturbations that revert to mean with a reversion
% W_ee = 1.25; % in mV*s, E->E connection weight(or sum therof)
% W_ie = 1.2; % in mV*s, E->I connection weight (or sum therof)
% W_ei = 0.65; % in mV*s, E->I connection weight (or sum therof)
% W_ii = 0.5; % in mV*s, I->I connection weight (or sum therof)
    % W is the (positive or zero) strength of the synaptic connection from neuron j to neuron i
    % Constants rather than matrices in this model
% V_i Denotes membrane voltage V_m of neuron i
% T_e = 20; % in ms, Membrane time constant for excitatory units
% T_i = 10; % in ms, Membrane time constant for inhibitory units
% V_rest = -70; % in mV, Resting membrane potential
% V_0 = -70; % in mV, Rectification threshold potential
V_e = zeros(N_e, iters); % Trace of excitatory units at each timestep of iteration
V_i = zeros(N_i, iters); % Trace of inhibitory units at each timestep of iteration
V_e(:,1) = h(1) + n_e(:,1); % Update first timestep with initial values (Excitatory Units)
V_i(:,1) = h(1) + n_i(:,1);  % Update first timestep with initial values (Inhibitory Units)
% Find firing rate at first step using supralinear nonlinearity powerlaw
% threshold
    % I/O FUNCTION- MOMEMTARY FIRING RATE
    % r(t) is the momemtary firing rate at timestep t
r_e(:, 1) = k*max(floor(V_e(:, 1)-V_0), 0).^n;
r_i(:, 1) = k*max(floor(V_i(:, 1)-V_0), 0).^n;


for i = 2:iters % LOOP THRO?UGH TO UPDATE TIME CONSTANT FOR EACH 30ms TIMESTEP INTERATION
    for ii = 1:N_e % LOOP THROUGH EXCITATORY UNITS
        % MEMBRANE UPDATE
        V_e(ii, i) = (-V_e(ii, i-1) + V_rest + h(i) + n_e(i) + ...
            (W_ee*r_e(:,i-1) + W_ei*r_i(:,i-1)))/ T_i;
        % NONLINEAR FIRING RATE FUNTION
        r_e(ii, i) =  k*max(floor(V_e(:, 1)-V_0), 0).^n;
    end
    for iii = 1:N_i % LOOP THROUGH INHIBITORY UNITS
        % MEMBRANE UPDATE
        V_i(iii, i) = (-V_i(iii, i-1) + V_rest + h(i) + n_i(i) + ...
            (W_ii*r_e(:,i-1) + W_ie*r_i(:,i-1)))/ T_i;
        % NONLINEAR FIRING RATE FUNTION
        r_i(iii, i) = k*max(floor(V_i(:, 1)-V_0), 0).^n;
    end
    
end
