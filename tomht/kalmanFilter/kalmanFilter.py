#!/usr/bin/env python3
import numpy as np
import numpy.linalg as linalg

def filterPredict(transition_matrix, transition_covariance, current_state_mean,current_state_covariance):
    predicted_state_mean = np.dot(transition_matrix, current_state_mean)
    predicted_state_covariance = (np.dot(transition_matrix,np.dot(current_state_covariance,transition_matrix.T))+ transition_covariance)
    return (predicted_state_mean, predicted_state_covariance)

def filterCorrect(observation_matrix, observation_covariance, predicted_state_mean,predicted_state_covariance, observation):
    predicted_observation_mean = (
        np.dot(observation_matrix,
         predicted_state_mean)
        )
    residual_covariance = (
        np.dot(observation_matrix,
         np.dot(predicted_state_covariance,
          observation_matrix.T))
        + observation_covariance
        )

    kalman_gain = (
        np.dot(predicted_state_covariance,
         np.dot(observation_matrix.T,
          linalg.pinv(residual_covariance)))
        )

    measurement_residual =  observation - predicted_observation_mean

    filtered_state_mean = (
        predicted_state_mean
        + np.dot(kalman_gain, measurement_residual)
        )
    filtered_state_covariance = (
        predicted_state_covariance
        - np.dot(kalman_gain,
           np.dot(observation_matrix,
            predicted_state_covariance))
        )
    return (measurement_residual,
            residual_covariance,
            kalman_gain, 
            filtered_state_mean,
            filtered_state_covariance
            )