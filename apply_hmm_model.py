import numpy as np
import pandas as pd
import datetime

from sklearn.metrics import confusion_matrix

from pomegranate import (HiddenMarkovModel,
                         State,
                         DiscreteDistribution,
                         )

###################
# build hmm model
###################

# The confusiong matrix from model validation data is used  for the HMM Emissions
model_validation_data = pd.read_csv('vgg16_v2_45epochs_validation_results.csv')

model_cm = confusion_matrix(y_true = model_validation_data.field_status_crop, 
                            y_pred = model_validation_data.field_status_prediction,
                            normalize = 'true')

def get_emissions(true_class, emissions_cm):
    """ 
    observed_var should be an int corresponding to a class
    
    emissions_cm is a an array confusiong matrix from the classification model
    normalized to each observed var. ie. given a true value X what is the probability
    of it being classififed as class X
    
    from sklearn.metrics.confusion_matric(y_true, y_pred, normalize='true')
    """

    
    # true labels in the cm are rows, so rowwise sums should == 1 here
    # small rounding errors allows
    #assert np.isclose(emissions_cm.sum(1), 1).all(), 'rowise sums != 1'
    assert emissions_cm.shape[0] == emissions_cm.shape[1]

    errors = emissions_cm[true_class].copy()
    # We all make mistakes. Each observed state should have at least a small 
    # probability for every hidden state
    errors[errors<=0.01] = 0.01
    errors = errors / errors.sum()
    
    n_classes = emissions_cm.shape[0]

    emissions = {}
    for predicted_class in range(n_classes):
        emissions[predicted_class] = errors[predicted_class]
    
    return emissions
    
classes = np.unique(model_validation_data.field_status_crop.values)
classes.sort()
states = {}
for true_class in classes:
    e = get_emissions(true_class, emissions_cm=model_cm)
    states[true_class] = State(DiscreteDistribution(e), name = str(true_class) + '_obs') # add _obs to clarify the state space variables

# Hand made transition probabilites
transition_probabilites = pd.read_csv('state_transition_weights.csv')


# drop unknown, snow, and water classes from hidden state.
# observed states can still be "unknown"
#_ = states.pop(6)
#_ = states.pop(7)
_ = states.pop(8)
transition_probabilites = transition_probabilites[(transition_probabilites.from_class_id <= 7 ) & (transition_probabilites.to_class_id <= 7)]

###############
model = HiddenMarkovModel()

# Initialize each hidden state.
# All states have an equal chance of being the starting state.
for s in states.values():
    model.add_state(s)
    model.add_transition(model.start, s, 1)

# Pairwise transitions between states
for row_i, row in transition_probabilites.iterrows():
    from_state = states[row.from_class_id]
    to_state   = states[row.to_class_id]
    model.add_transition(from_state, to_state, row.weight)

model.bake(verbose=True)
#####################
# Load the file prepared by prep_predictions_for_hmm.R
image_predictions = pd.read_csv('results/image_predictions_for_hmm.csv')

def apply_hmm_model(grouped_site_sequence, model = model):
    obs = grouped_site_sequence.sort_values('date').class_id.to_list()
    grouped_site_sequence['hmm_class_id'] = model.predict(obs, algorithm='viterbi')[1:]
    return grouped_site_sequence

image_predictions.groupby(['phenocam_name', 'site_sequence_id']).apply(apply_hmm_model).to_csv('results/image_predictions_hmm_final.csv', index=False)








