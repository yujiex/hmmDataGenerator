import numpy as np

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# helper functions
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# Generate an array of length num_states, and sum to 1
def initialize_start_prob(num_states):
    v = np.random.uniform(size=num_states)
    normalized = v/sum(v)
    normalized[-1] = 1 - sum(normalized[:-1])
    return(normalized)

# Sample categorical RV
# prob: an np.array holding the PMF of a categorical RV
def get_state(prob):
    return list(np.random.multinomial(n=1, pvals=prob)).index(1)

# Initialize a probability matrix each row has probabilities sum to one
# row name is a "from" state, col name is a "to" state
# result[i,j] is the probability of transition from state i to j
def rand_init_prob_matrix(nrow, ncol):
    return np.array([initialize_start_prob(ncol) for r in range(nrow)])

# Get the next hidden states from the previous state and a long-ago state
# prev_state: the hidden state at the (t-1) step
# long_ago_state: the hidden state at the (t-q) step
# weight: for adjusting the strength of dependence on state t-1 vs t-q (q could
# be 100 for example).
# transition_matrix
def get_next_hidden_state(prev_state, long_ago_state, weight, transition_matrix):
    assert prev_state is not None
    if long_ago_state is None:
        assert prev_state < transition_matrix.shape[0]
        state = get_state(transition_matrix[prev_state,:])
    else:
        assert long_ago_state < transition_matrix.shape[0]
        assert prev_state < transition_matrix.shape[0]
        prob = weight * transition_matrix[long_ago_state,:] + \
               (1 - weight) * (transition_matrix[prev_state,:])
        state = get_state(prob)
    return state

# Generate a sequence of length "output_length"
# num_hidden_states: number of hidden states
# num_obs_states: number of observable states
# lag_size: the number of step ago for the long-ago state
# weight: for adjusting the strength of dependence on state t-1 vs t-q (q could
# output_length: the length of the generated sequence
# starting_prob: a length num_hidden_states array to sample the starting hidden state from
# transition_matrix: the transition matrix
# emission_matrix: the emission matrix
def hmm_generator_short_long(num_hidden_states, num_obs_states,
                             lag_size, weight, output_length,
                             starting_prob,
                             transition_matrix, emission_matrix):
    print("hmm generating a data sequence of length {:d}".format(output_length))
    seq_hidden = np.empty(output_length + 1, dtype = int)
    seq_obs = np.empty(output_length, dtype = int)

    seq_hidden[0] = get_state(starting_prob)
    # print("hidden state {}".format(seq_hidden[0]))
    for t in range(1, output_length):
        # for the beginning lag_size states, generate with only the previous
        # state
        if (t - lag_size < 0):
            seq_hidden[t] = get_next_hidden_state(seq_hidden[t - 1], None,
                                                  weight, transition_matrix)
        else:
            seq_hidden[t] = get_next_hidden_state(seq_hidden[t - 1],
                                                  seq_hidden[t - lag_size],
                                                  weight, transition_matrix)
        seq_obs[t] = get_state(emission_matrix[seq_hidden[t], :])
        # print("{} hidden state: {:d}, output state: {:d}".format(t,
        #                                                          seq_hidden[t],
        #                                                          seq_obs[t]))
    return seq_obs


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# a toy example
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
hid = 3
obs = 5

# initialize the probability tables
start_prob = initialize_start_prob(num_states=hid)
print(start_prob)
trans = rand_init_prob_matrix(hid, hid)
print(trans)
emission = rand_init_prob_matrix(hid, obs)
print(emission)

lag_size = 100
# initialize the length of the sequence
T = np.random.randint(lag_size + 1, 10 * lag_size)

seq_data = hmm_generator_short_long(num_hidden_states=hid, num_obs_states=obs,
                                    lag_size=lag_size, weight=0.8, output_length=T,
                                    starting_prob=start_prob, transition_matrix=trans,
                                    emission_matrix=emission)
