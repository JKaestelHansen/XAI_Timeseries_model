# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from captum.attr import Saliency, IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange
from torch.utils.data import Dataset, DataLoader

from helper_functions_toyUnet import *

device = 'cpu'

use_MS2_data = False
attach_MS2 = False

if use_MS2_data:
    C, D, y, fulltime_padded_states_all = Load_simulated_data(path='_data/dataset_for_Jacob.pkl')
    y = torch.stack(y).to(device)
    X = torch.stack(C).to(device)  # [N, 3, T_in] 
    D = torch.tensor(D, dtype=torch.float32).to(device)  # [N, T_in]
    X = torch.cat((X, D.unsqueeze(1)), dim=1)  # [N, 4, T_in]
    if attach_MS2:
        X = torch.cat((X, y.unsqueeze(1)), dim=1)  # [N, 5, T_in]
    else:
        X = torch.cat((X, torch.zeros_like(y).unsqueeze(1)), dim=1)  # [N, 5, T_in]
    # permute to [N, T_in, 5]
    X = X.permute(0, 2, 1)
    y = y.unsqueeze(-1)
else:
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = generate_data(n_samples=10000)
    y = y#.squeeze(-1).long()

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# model
# model = TransformerModel(vocab_size=X.shape[-1],
#                          d_model=6, 
#                          nhead=2, 
#                          num_encoder_layers=1,
#                          num_decoder_layers=1, 
#                          dim_feedforward=2, 
#                          dropout=0.01,x 
#                          max_len=1000
#                          ).to(device)

model = UNet1DVariableDecoder_resnet(
                              X.shape[-1], 
                              encoder_depth=3, decoder_depth=3, base_channels=64,
                              init_thresh=0.15, init_alpha=100.0, learn_alpha=False,
                              output_length=X.shape[1],
                              use_DistanceGate_mask=False).to(device)




best_model_state = pickle.load(open('toy_transformer_model.pickle', 'rb'))
train_losses = pickle.load(open('toy_transformer_trainloss.pickle', 'rb'))
val_losses = pickle.load(open('toy_transformer_valloss.pickle', 'rb'))

plt.figure(figsize=(6, 3))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# Restore best model
model.load_state_dict(best_model_state)

# ----- Evaluation + Visualization -----
idx_chosen = 25 #np.random.randint(0, X_val.shape[0])
print('idx_chosen', idx_chosen)

# 156, 1897

model.eval()
input_ts = X_val[idx_chosen:idx_chosen+1].clone().detach().requires_grad_()
pred, logvar, _ = model(input_ts)
std = logvar.exp().sqrt().detach()

print(pred.shape, std.shape)
print(std.min(), std.max())

probs = torch.softmax(pred, dim=-1)

fig, ax = plt.subplots(3,1,figsize=(10, 6))
ax[0].plot(input_ts.squeeze().detach().cpu()[:,3], label='Input Time Series')
ax[0].set_title("Input Time Series")
ax[0].legend(fontsize=8, loc='upper left')

ax[1].plot(y_val[idx_chosen].squeeze().cpu(), label='Target')
ax[1].plot(pred.squeeze().detach().cpu(), label='Predicted Probabilities', color='C1')
ax[1].fill_between(
    np.arange(len(pred.squeeze().detach().cpu())), 
    pred.squeeze().detach().cpu()-std.squeeze().detach().cpu(),
    pred.squeeze().detach().cpu()+std.squeeze().detach().cpu(),
    color='C1', alpha=0.3)
ax[1].set_title("Target vs Predicted Probabilities")
ax[1].legend(fontsize=8, loc='upper left')
ax[1].set_ylim(-0.2, 1)

ax[2].plot(y_val[idx_chosen].squeeze().cpu(), label='Target')
ax[2].plot(input_ts.squeeze().detach().cpu()[:,3] < 0.05, label='<0.05', color='dimgrey')
ax[2].plot(pred.squeeze().detach().cpu(), label='Predicted Probabilities', color='C1')
ax[2].fill_between(
    np.arange(len(pred.squeeze().detach().cpu())), 
    pred.squeeze().detach().cpu()-std.squeeze().detach().cpu(),
    pred.squeeze().detach().cpu()+std.squeeze().detach().cpu(),
    color='C1', alpha=0.3)
ax[2].set_title("Target vs Predicted Probabilities")
ax[2].legend(fontsize=8, loc='upper left')
ax[2].set_ylim(-0.2, 1)

plt.tight_layout()

# %%

"""
----- 

### Captum, (attention if transformer or have attention), and uncertain estimate evaluation playground ###

Below is a code for various methods for primary attribution within Captum, as well as noise tunnel.
also metrics to estimate the trustworthiness of model explanations, 
i.e., infidelity and sensitivity metrics to estimate the goodness of explanations.

Source: https://captum.ai/docs/attribution_algorithms

----- 
"""

# Most methods require a wrapper for the model forward
class TimeStepClassifierWrapper(nn.Module):
    def __init__(self, model, timestep, class_idx):
        super().__init__()
        self.model = model
        self.timestep = timestep
        self.class_idx = class_idx

    def forward(self, x):
        logits = self.model(x)
        return logits[:, self.timestep, self.class_idx]
    

class TimeStepRegressorWrapper(nn.Module):
    def __init__(self, model, timestep):
        super().__init__()
        self.model = model
        self.timestep = timestep

    def forward(self, x):
        mean, _, _ = self.model(x)
        mean = mean.squeeze(1)
        return mean[:, self.timestep]
    
class TimeStepUncertaintyWrapper(nn.Module):
    def __init__(self, model, timestep):
        super().__init__()
        self.model = model
        self.timestep = timestep

    def forward(self, x):
        _, logvar, _ = self.model(x)
        logvar = logvar.squeeze(1)
        return logvar[:, self.timestep]
    

###### Metrics ######

# Infidelity
"""
# Infidelity
Infidelity measures the mean squared error between model explanations in the magnitudes 
of input perturbations and predictor function's changes to those input perturbtaions.

https://arxiv.org/abs/1901.09392
"""

# Sensitivity
"""
# Sensitivity
Sensitivity measures the degree of explanation changes to subtle 
input perturbations using Monte Carlo sampling-based approximation

https://arxiv.org/abs/1901.09392
"""


# Timepoint to evaluate saliency/feature importance
pred_timestep = 27

# Check if model has attention weights
model_has_attention = hasattr(model, 'attn_weights_all_layers')

#wrapped_model = TimeStepClassifierWrapper(model, pred_timestep, class_idx=1)
wrapped_model = TimeStepRegressorWrapper(model, pred_timestep)
wrapped_model_uncertainty = TimeStepUncertaintyWrapper(model, pred_timestep)



# ----- Integrated Gradients -----
"""
Integrated gradients represents the integral of gradients with respect to inputs along the path from a given baseline to input."
https://arxiv.org/abs/1703.01365

"""

ig = IntegratedGradients(wrapped_model)
ig_unc = IntegratedGradients(wrapped_model_uncertainty)
baseline = torch.zeros_like(input_ts)

attr_ig = ig.attribute(input_ts, baselines=baseline).detach().squeeze().cpu().numpy()
attr_ig = attr_ig/np.max(np.abs(attr_ig))

attr_ig_uncertainty = ig_unc.attribute(input_ts, baselines=baseline).detach().squeeze().cpu().numpy()
attr_ig_uncertainty = attr_ig_uncertainty/np.max(np.abs(attr_ig_uncertainty))

print(input_ts.shape)

fig, ax = plt.subplots(6,1,figsize=(10, 8))
ax[0].plot(input_ts[0,:,3].detach().cpu().numpy(), label='Input data')
ax[1].plot(y_val[idx_chosen].squeeze().cpu(), label='Input data')
ax[1].plot(pred.squeeze().detach().cpu(), label='Predicted Probabilities', color='C1')
ax[1].axvline(pred_timestep, color='r', lw=1)

ax[2].plot(np.abs(attr_ig[:,0])/np.max(np.abs(attr_ig)), label='Feat 0')
ax[2].plot(np.abs(attr_ig_uncertainty[:,0])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 0 Uncertainty', linestyle='--')
ax[2].set_ylim(0, 1)
ax[2].axvline(pred_timestep, color='r', lw=1)

ax[3].plot(np.abs(attr_ig[:,1])/np.max(np.abs(attr_ig)), label='Feat 1')
ax[3].plot(np.abs(attr_ig_uncertainty[:,1])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 1 Uncertainty', linestyle='--')
ax[3].set_ylim(0, 1)
ax[3].axvline(pred_timestep, color='r', lw=1)

ax[4].plot(np.abs(attr_ig[:,2])/np.max(np.abs(attr_ig)), label='Feat 2')
ax[4].plot(np.abs(attr_ig_uncertainty[:,2])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 2 Uncertainty', linestyle='--')
ax[4].set_ylim(0, 1)
ax[4].axvline(pred_timestep, color='r', lw=1)

ax[5].plot(np.abs(attr_ig[:,3])/np.max(np.abs(attr_ig)), label='Feat 3')
ax[5].plot(np.abs(attr_ig_uncertainty[:,3])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 3 Uncertainty', linestyle='--')
ax[5].set_ylim(0, 1)
ax[5].axvline(pred_timestep, color='r', lw=1)

ax[0].set_title("Integrated Gradients Attribution")

plt.tight_layout()
plt.show()


# ----- Noise Tunnel -----
"""
Noise Tunnel is a method that can be used on top of any of the attribution methods. 
Noise tunnel computes attribution multiple times, 
adding Gaussian noise to the input each time, and combines the calculated attributions based on the chosen type.

https://arxiv.org/abs/1706.03825
https://arxiv.org/abs/1810.03307
"""
ig = IntegratedGradients(wrapped_model)
ig_unc = IntegratedGradients(wrapped_model_uncertainty)
ig_nt = NoiseTunnel(ig)
ig_nt_unc = NoiseTunnel(ig_unc)


attr_ig = ig_nt.attribute(input_ts, baselines=baseline).detach().squeeze().cpu().numpy()
attr_ig = attr_ig/np.max(np.abs(attr_ig))

attr_ig_uncertainty = ig_nt_unc.attribute(input_ts, baselines=baseline).detach().squeeze().cpu().numpy()
attr_ig_uncertainty = attr_ig_uncertainty/np.max(np.abs(attr_ig_uncertainty))



fig, ax = plt.subplots(6,1,figsize=(10, 8))
ax[0].plot(input_ts[0,:,3].detach().cpu().numpy(), label='Input data')

ax[1].plot(y_val[idx_chosen].squeeze().cpu(), label='Input data')
ax[1].plot(pred.squeeze().detach().cpu(), label='Predicted Probabilities', color='C1')
ax[1].axvline(pred_timestep, color='r', lw=1)

ax[2].plot(np.abs(attr_ig[:,0])/np.max(np.abs(attr_ig)), label='Feat 0')
ax[2].plot(np.abs(attr_ig_uncertainty[:,0])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 0 Uncertainty', linestyle='--')
ax[2].set_ylim(0, 1)
ax[2].axvline(pred_timestep, color='r', lw=1)

ax[3].plot(np.abs(attr_ig[:,1])/np.max(np.abs(attr_ig)), label='Feat 1')
ax[3].plot(np.abs(attr_ig_uncertainty[:,1])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 1 Uncertainty', linestyle='--')
ax[3].set_ylim(0, 1)
ax[3].axvline(pred_timestep, color='r', lw=1)

ax[4].plot(np.abs(attr_ig[:,2])/np.max(np.abs(attr_ig)), label='Feat 2')
ax[4].plot(np.abs(attr_ig_uncertainty[:,2])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 2 Uncertainty', linestyle='--')
ax[4].set_ylim(0, 1)
ax[4].axvline(pred_timestep, color='r', lw=1)

ax[5].plot(np.abs(attr_ig[:,3])/np.max(np.abs(attr_ig)), label='Feat 3')
ax[5].plot(np.abs(attr_ig_uncertainty[:,3])/np.max(np.abs(attr_ig_uncertainty)), label='Feat 3 Uncertainty', linestyle='--')
ax[5].set_ylim(0, 1)
ax[5].axvline(pred_timestep, color='r', lw=1)

ax[0].set_title("Integrated Gradients Attribution")

plt.tight_layout()

# ----- Gradient SHAP -----
"""
Gradient SHAP is a gradient method to compute SHAP values, which are based on Shapley values proposed in cooperative game theory

The computed attributions approximate SHAP values under the assumptions that the input features are independent and that the explanation model is linear between the inputs and given baselines.

https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

"""
print('We probably can not argue input features are independent')



# ----- DeepLIFT -----
"""
DeepLIFT is a back-propagation based approach that attributes a change to inputs based on the differences 
between the inputs and corresponding references (or baselines) for non-linear activations. 
DeepLIFT seeks to explain the difference in the output from reference in terms of the difference in inputs from reference.
https://arxiv.org/abs/1704.02685
"""


# ----- DeepLIFT SHAP -----
"""
DeepLIFT SHAP is a method extending DeepLIFT to approximate SHAP values, which are based on Shapley values proposed in cooperative game theory

"""


# ----- Saliency -----
"""
Saliency is a simple approach for computing input attribution, returning the gradient of the output with respect to the input.
https://arxiv.org/abs/1312.6034
"""

saliency = Saliency(wrapped_model)
saliency_uncertainty = Saliency(wrapped_model_uncertainty)

attr_sal = saliency.attribute(input_ts)
attr_sal = attr_sal.abs().detach().squeeze().cpu().numpy()

attr_sal_uncertainty = saliency_uncertainty.attribute(input_ts)
attr_sal_uncertainty = attr_sal_uncertainty.abs().detach().squeeze().cpu().numpy()

fig, ax = plt.subplots(6,1,figsize=(10, 8))
ax[0].plot(input_ts[0,:,3].detach().cpu().numpy(), label='Input data')

ax[1].plot(y_val[idx_chosen].squeeze().cpu(), label='Input data')
ax[1].plot(pred.squeeze().detach().cpu(), label='Predicted Probabilities', color='C1')
ax[1].axvline(pred_timestep, color='r', lw=1)

ax[2].plot(np.abs(attr_sal[:,0])/np.max(np.abs(attr_sal)), label='Feat 0')
ax[2].plot(np.abs(attr_sal_uncertainty[:,0])/np.max(np.abs(attr_sal_uncertainty)), label='Feat 0 Uncertainty', linestyle='--')
ax[2].set_ylim(0, 1)
ax[2].axvline(pred_timestep, color='r', lw=1)

ax[3].plot(np.abs(attr_sal[:,1])/np.max(np.abs(attr_sal)), label='Feat 1')
ax[3].plot(np.abs(attr_sal_uncertainty[:,1])/np.max(np.abs(attr_sal_uncertainty)), label='Feat 1 Uncertainty', linestyle='--')
ax[3].set_ylim(0, 1)
ax[3].axvline(pred_timestep, color='r', lw=1)

ax[4].plot(np.abs(attr_sal[:,2])/np.max(np.abs(attr_sal)), label='Feat 2')
ax[4].plot(np.abs(attr_sal_uncertainty[:,2])/np.max(np.abs(attr_sal_uncertainty)), label='Feat 2 Uncertainty', linestyle='--')
ax[4].set_ylim(0, 1)
ax[4].axvline(pred_timestep, color='r', lw=1)

ax[5].plot(np.abs(attr_sal[:,3])/np.max(np.abs(attr_sal)), label='Feat 3')
ax[5].plot(np.abs(attr_sal_uncertainty[:,3])/np.max(np.abs(attr_sal_uncertainty)), label='Feat 3 Uncertainty', linestyle='--')
ax[5].set_ylim(0, 1)
ax[5].axvline(pred_timestep, color='r', lw=1)

ax[0].set_title("Saliency Attribution")

plt.tight_layout()


# ----- Input X Gradient -----
"""
Input X Gradient is an extension of the saliency approach, taking the gradients of the output with respect to the input and multiplying by the input feature values.
"""

# ----- Guided Backpropagation and Deconvolution -----
"""
Guided backpropagation and deconvolution compute the gradient of the target output with respect to the input, 
but backpropagation of ReLU functions is overridden so that only non-negative gradients are backpropagated. 
- In guided backpropagation, the ReLU function is applied to the input gradients, 
https://arxiv.org/abs/1412.6806

- In deconvolution, the ReLU function is applied to the output gradients and directly backpropagated.
https://arxiv.org/abs/1311.2901

Both approaches were proposed in the context of a convolutional network and are generally used for convolutional networks, although they can be applied generically.
"""

# ----- Guided GradCAM -----
"""
Guided GradCAM computes the element-wise product of guided backpropagation attributions with upsampled (layer) GradCAM attributions. 
GradCAM attributions are computed with respect to a given layer, and attributions are upsampled to match the input size. 

Notes:
- This approach is designed for convolutional neural networks. 
- The chosen layer is often the last convolutional layer in the network, but any layer that is spatially aligned with the input can be provided.

https://arxiv.org/abs/1610.02391
"""

# ----- Feature Ablation -----
"""
Feature ablation is a perturbation based approach to compute attribution, 
involving replacing each input feature with a given baseline / reference value (e.g. 0), and computing the difference in output.

Input features can also be grouped and ablated together rather than individually.

"""


# ----- Occlusion -----
"""
Occlusion is a perturbation based approach to compute attribution, involving replacing each contiguous 
rectangular region with a given baseline / reference, and computing the difference in output.

Occlusion is most useful in cases such as images, where pixels in a contiguous rectangular region are likely to be highly correlated.
probably also in correlated time series then

https://arxiv.org/abs/1311.2901

"""




if model_has_attention:
    # ----- Attention -----
    with torch.no_grad():
        _ = model(input_ts)

    # Plot attention for each layer and head
    for layer_idx, layer_attn in enumerate(model.attn_weights_all_layers):
        for head_idx in range(layer_attn.shape[1]):
            attn_map = layer_attn[0, head_idx].cpu()  # [T, T]
            plt.figure(figsize=(6, 5))
            sns.heatmap(attn_map, cmap='viridis')
            plt.title(f"Layer {layer_idx} - Head {head_idx} Attention")
            plt.xlabel("Source Timestep")
            plt.ylabel("Target Timestep")
            plt.tight_layout()
            plt.show()


    # -------- Additional Attention Visualizations --------
    def plot_average_attention(attn_weights):
        for l, attn in enumerate(attn_weights):
            for h in range(attn.shape[1]):
                avg_attn = attn[0, h].mean(dim=0).cpu().numpy()
                plt.figure()
                sns.heatmap(avg_attn[None, :], cmap='viridis', cbar=True)
                plt.title(f'Layer {l+1}, Head {h+1} - Avg Attention')
                plt.xlabel("Source Timestep")
                plt.yticks([])
                plt.tight_layout()

    plot_average_attention(model.attn_weights_all_layers)


    def compute_rollout(attn_weights):
        rollout = torch.eye(attn_weights[0].shape[-1]).to(attn_weights[0].device)
        for attn in attn_weights:
            attn_head_avg = attn.mean(dim=1)  # [B, T, T]
            rollout = attn_head_avg[0] @ rollout
        return rollout.cpu().numpy()

    def plot_attention_rollout(rollout_matrix):
        plt.figure(figsize=(6, 5))
        sns.heatmap(rollout_matrix, cmap="viridis")
        plt.title("Attention Rollout")
        plt.xlabel("Source timestep")
        plt.ylabel("Target timestep")
        plt.tight_layout()

    rollout_matrix = compute_rollout(model.attn_weights_all_layers)
    plot_attention_rollout(rollout_matrix)


    def plot_attention_entropy(attn_weights):
        for l, attn in enumerate(attn_weights):
            for h in range(attn.shape[1]):
                entropy = - (attn[0, h] * torch.log(attn[0, h] + 1e-8)).sum(dim=-1).cpu().numpy()
                plt.plot(entropy, label=f"Layer {l+1} Head {h+1}")
        plt.title("Attention Entropy per Head")
        plt.xlabel("Target Timestep")
        plt.ylabel("Entropy")
        plt.legend()
        plt.tight_layout()

    plot_attention_entropy(model.attn_weights_all_layers)


    def plot_attention_at_timestep(attn_weights, timestep):
        for l, attn in enumerate(attn_weights):
            for h in range(attn.shape[1]):
                plt.figure()
                sns.heatmap(attn[0, h][timestep].cpu().numpy()[None, :], cmap="viridis")
                plt.title(f"Layer {l+1}, Head {h+1} - Attention at Timestep {timestep}")
                plt.xlabel("Source Timestep")
                plt.tight_layout()

    plot_attention_at_timestep(model.attn_weights_all_layers, pred_timestep)


    def plot_aggregated_attention(attn_weights):
        stacked = torch.stack([attn[0].mean(dim=0) for attn in attn_weights], dim=0)  # [L, T, T]
        avg_attn = stacked.mean(dim=0).cpu().numpy()
        plt.figure(figsize=(6, 5))
        sns.heatmap(avg_attn, cmap='viridis')
        plt.title("Aggregated Attention (Layer+Head Avg)")
        plt.xlabel("Source")
        plt.ylabel("Target")
        plt.tight_layout()

    plot_aggregated_attention(model.attn_weights_all_layers)


    # Optional: Compare saliency with attention
    sal = attr_sal.squeeze() / np.abs(attr_sal).max()
    attn = model.attn_weights_all_layers[0][0].mean(dim=0)[pred_timestep].cpu().numpy() / np.abs(attr_sal).max()

    plt.figure(figsize=(10, 4))
    plt.plot(sal, label="Saliency")
    plt.plot(attn, label=f"Attention @ t={pred_timestep}")
    plt.legend()
    plt.title("Saliency vs Attention")
    plt.xlabel("Timestep")
    plt.tight_layout()


    # %%
