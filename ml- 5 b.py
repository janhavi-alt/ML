#pip install numpy hmmlearn scipy cython
import numpy as np
from hmmlearn import hmm

# Example data (sequence of observed data)
# Let's assume these are the observed states (e.g., temperature readings)
observations = np.array([[0], [1], [2], [1], [0], [2], [1], [0], [1], [2]])

# Initialize the HMM model
# The number of hidden states is assumed to be 2 in this case (e.g., hot/cold weather)
model = hmm.CategoricalHMM(n_components=2, random_state=42, init_params="")

# Manually set the start probabilities, transition probabilities, and emission probabilities
model.startprob_ = np.array([0.5, 0.5])  # start probabilities must sum to 1
model.transmat_ = np.array([[0.7, 0.3],  # transition matrix: row sums to 1
                            [0.4, 0.6]]) # transition matrix: row sums to 1
model.emissionprob_ = np.array([[0.3, 0.4, 0.3],  # emission probabilities: column sums to 1
                                [0.5, 0.3, 0.2]])

# Fit the model to the observation sequence
model.fit(observations)

# Make predictions about the hidden states corresponding to the observations
hidden_states = model.predict(observations)

# Print the hidden states
print("Hidden States:")
print(hidden_states)

# To predict the likelihood of a given observation sequence
logprob = model.score(observations)
print(f"Log probability of the observation sequence: {logprob}")

# To generate new observations based on the model
generated_samples, _ = model.sample(10)  # generate 10 samples
print("Generated samples:")
print(generated_samples)
