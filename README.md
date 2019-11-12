Comments on the code:

    1. Transition matrix: should depend on the previous two states.
        not linear combination of the transition matrix applied to the states separately.
        Q: could there be two matrices? yes but then it would take them independently.

    2. There are two functions one for hidden markov model generator (there are hidden states)
    And one for markov chain generator (there are no hidden states). Markov chain is more similar to
    n-grams, however not so much here since the dependence is on a long ago state and the previous state.