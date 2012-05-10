#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       gmphdfilter.py
#       
#       Copyright 2012 Sharad Nagappa <snagappa@gmail.com>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#       

import numpy as np
from phdfilter import PHD
import copy


class GMPHD(PHD):
    def __init__(self, markov_predict_fn_handle, markov_predict_fn_parameters, 
                 state_update_fn_handle, state_update_fn_parameters,
                 obs_fn_handle, obs_fn_parameters, 
                 clutter_fn_handle, clutter_fn_parameters,
                 birth_fn_handle, birth_fn_parameters,
                 ps_fn_handle, pd_fn_handle, likelihood_fn_handle,
                 estimate_fn_handle, 
                 phd_parameters={"max_terms":100,
                                 "elim_threshold":1e-4,
                                 "merge_threshold":4}):
        super(GMPHD, self).__init__( 
            markov_predict_fn_handle, markov_predict_fn_parameters, 
            state_update_fn_handle, state_update_fn_parameters,
            obs_fn_handle, obs_fn_parameters, 
            clutter_fn_handle, clutter_fn_parameters,
            birth_fn_handle, birth_fn_parameters,
            ps_fn_handle, pd_fn_handle, likelihood_fn_handle,
            estimate_fn_handle, phd_parameters)
        
    
    def phdUpdate(self, observation_set):
        num_x = len(self.states)
        num_observations = len(observation_set)
        if num_observations:
            z_dim = len(observation_set[0])
        else:
            z_dim = 0
        
        detection_probability = self.pd_fn_handle(self.states)
        clutter_pdf = [self.clutter_fn_handle(_observation_) for _observation_ in observation_set]
        
        # Account for missed detection
        self._states_ = copy.deepcopy(self.states)
        self._weights_ = [self.weights*(1-detection_probability)]
        
        # Scale the weights by detection probability -- same for all detection terms
        self.weights.__imul__(detection_probability)
        
        # Split x and P out from the combined state vector
        x = [self.states[i][0] for i in range(num_x)]
        P = [self.states[i][1] for i in range(num_x)]
        
        # Part of the Kalman update is common to all observation-updates
        inv_sqrt_S, det_S, pred_z, kalman_gain = kalman_update(x, P, 
                                                    self.obs_fn_parameters.H, 
                                                    self.obs_fn_parameters.R)
        
        # We need to update the states and find the updated weights
        for (_observation_, obs_count) in zip(observation_set, range(num_observations)):
            new_x = copy.deepcopy(x)
            # Apply the Kalman update to get the new state
            residuals = kalman_update_x(new_x, pred_z, _observation_, kalman_gain)
            
            # Calculate the weight of the Gaussians for this observation
            # Calculate term in the exponent
            x_pdf = [np.exp(-0.5*(np.dot(inv_sqrt_S[i], residuals[i]).A[0]**2).sum())/np.sqrt(det_S[i]*(2*np.pi)^z_dim) for i in range(num_x)]
            new_weight = self.weights*x_pdf
            # Normalise the weights
            new_weight.__idiv__(clutter_pdf(obs_count) + new_weight.sum())
            
            # Create new state with new_x and P to add to _states_
            self._states_ += [[copy.copy(new_x[i]), copy.copy(P[i])] for i in range(num_x)]
            self._weights += [new_weight.copy()]
            
        self._weights = np.concatenate(self._weights)
        
    
    def phdPrune(self):
        if (self.phd_parameters.elim_threshold <= 0):
            return
        retain_indices = np.where(self._weights_ >= self.phd_parameters.elilm_threshold)
        pruned_states = [self._states_[ri] for ri in retain_indices]
        pruned_weights = self._weights_[retain_indices]
        
        self._states_ = pruned_states
        self._weights_ = pruned_weights
        
    
    def phdMerge(self):
        if (self.phd_parameters.merge_threshold <= 0):
            return
        
    

"""
    % Merge Gaussian components
    NewComponentIndex = 0;
    NumRemainingComponents = length(RetainIndices);
    NumStates = size(PHD.Mn, 1);

    while NumRemainingComponents
        NewComponentIndex = NewComponentIndex + 1;
        [~, MaxWtIndex] = max(PHD.Wt);
        
        % Calculate Mahalanobis distance
        MD = zeros(1, NumRemainingComponents);
        for MDComponentIndex = 1:NumRemainingComponents
            Residual = PHD.Mn(:,MDComponentIndex)-PHD.Mn(:,MaxWtIndex);
            MD(MDComponentIndex) = ... 
                Residual'*inv(PHD.Cv(:,:,MaxWtIndex))*Residual; %#ok<MINV>
        end
        
        % Obtain list of components to be merged that have MD < MergeThresh
        MergedComponentsList = find(MD < Threshold.Merge);
        NumMergedComponents = length(MergedComponentsList);
        
        % Calculate merged weight
        MergedWeight = sum(PHD.Wt(MergedComponentsList));
        
        % Calculate merged mean
        MergedMean = sum( ...
            repmat(PHD.Wt(MergedComponentsList)', NumStates, 1).* ...
            PHD.Mn(:, MergedComponentsList), 2)/MergedWeight;
        
        % Calculate merged covariance
        MergedCov = zeros(NumStates);
        MergedComponentsResiduals = ...
            repmat(MergedMean, 1, NumMergedComponents);
        MergedComponentsResiduals = ...
            MergedComponentsResiduals - PHD.Mn(:, MergedComponentsList);
        for MergedComponentsCount = 1:NumMergedComponents
            MergedCov = MergedCov + ...
                PHD.Wt(MergedComponentsList(MergedComponentsCount))* ...
                ... Cv(:,:,MaxWtIndex); % in gaus_merge - wrong?
                (PHD.Cv(:,:,MergedComponentsCount) + ...
                MergedComponentsResiduals(:,MergedComponentsCount)* ...
                MergedComponentsResiduals(:,MergedComponentsCount)');
        end
        MergedCov = MergedCov/MergedWeight;
        
        % Save merged components
        mPHD = CatPHD(mPHD, ... 
            InitPHDStruct([], MergedWeight, MergedMean, MergedCov));
        
        
        % Remove merged components from original state
        AllIndices = 1:NumRemainingComponents;
        RetainIndices = unique([0 ... 
            (~sum(repmat(AllIndices,length(MergedComponentsList),1)- ...
            repmat( ... 
            MergedComponentsList(:),1,NumRemainingComponents)==0, 1)).* ...
            AllIndices]);
        RetainIndices = RetainIndices(2:end);
        PHD = SelectPHDComponents(PHD, RetainIndices);
        
        NumRemainingComponents = length(RetainIndices);
    end
else
    mPHD = PHD;
end
"""
    
    
    
    
    def phdEstimate(self):
        pass
    

def kalman_predict(x, P, F, Q):
    num_x = len(x)
    # Predict state
    x_pred = [np.dot(F, x[i]).A[0] for i in range(num_x)]
    
    # Predict covariance
    F = np.matrix(F)
    P_pred = [F*P[i]*F.T + Q for i in range(num_x)]
    
    x = x_pred
    P = P_pred


def kalman_update(x, P, H, R, z=None):
    num_x = len(x)
    if len(H) == 1:
        h_idx = [0]*num_x
    else:
        h_idx = range(num_x)
    if len(R) == 1:
        r_idx = [0]*num_x
    else:
        r_idx = range(num_x)
        
    # Evaluate inverse and determinant using Cholesky decomposition
    sqrt_S = [np.linalg.cholesky(H[h_idx[i]*P[i]*H[h_idx[i]].T] + R[r_idx[i]]) for i in range(num_x)]
    inv_sqrt_S = [sqrt_S[i].getI() for i in range(num_x)]
    
    det_S = [np.diag(sqrt_S[i]).prod()**2 for i in range(num_x)]
    inv_S = [inv_sqrt_S[i].T*inv_sqrt_S[i] for i in range(num_x)]
    
    # Kalman gain
    kalman_gain = [P[i]*H[h_idx[i]].T*inv_S[i] for i in range(num_x)]
    
    # Predicted observations
    pred_z = [H[h_idx[i]]*x[i] for i in range(num_x)]
    
    # Update to new state if observations were received
    if type(z) != None:
        residuals = [z - pred_z[i] for i in range(num_x)]
        [x[i].__iadd__(np.dot(kalman_gain[i], residuals[i]).A[0])]
        
    # Update covariance
    [P[i].__isub__(kalman_gain[i]*H[h_idx[i]]*P[i])]
    return inv_sqrt_S, det_S, pred_z, kalman_gain
    

def kalman_update_x(x, zhat, z, kalman_gain):
    num_x = len(x)
    residuals = [z - zhat[i] for i in range(num_x)]
    [x[i].__iadd__(np.dot(kalman_gain[i], residuals[i]).A[0])]
    return residuals
    
