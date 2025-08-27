# XAI_Timeseries_model
Explainable AI model (XAI) for timeseries data.

The goal is to learn Y (MS2 intensities, binds nascant RNA to visualize transcription) from X (enhancer promoter distances) and in the process learn what in enhancer promoter interactions are important, thus uncovering important aspects of gene regulation. Potentially, utilize Y in input for self-supervision.

- Predicitons and uncertainty estimation per time point
- Visualize feature importance of deep learning models

## Main playground scripts:

playground_toyUnet_n_Tranformer.py
- WIP: script to train a transformer/Unet with mean and logvar predictions per timepoint for regression with uncertainty estimation. Loss function: Calibrated GaussianNLL
plot_playground_toyUnet_n_Tranformer.py
- WIP: Script to plot results from playground_toyUnet_n_Tranformer.py. Train/val loss curves, examples of predictions, and plotting + explanation of saliency/feature importance attribution.

playground_EP_to_MS2_XAI.py
- WIP
- Explores mapping from EP to MS2
- In Unet we mask all XY input where EP is not under a learned Parameter (proxy for contact radius). The idea being that enhancer promoter occur under some contact radius so all useful information should be under this threshold and having it as a learnable parameter we get an estimate from the model.
- Even if one feeds Y as input with the masking using the learned Parameter for contact radius then due to masking using distance the model cant see Y completely and just copy but it does help prediction
  - accuracy of learned contact radius is sensitive to loss function and regularization but it works surprisingly well
 
<img width="1189" height="890" alt="image" src="https://github.com/user-attachments/assets/56f395b8-383a-406b-973e-ce5d1087dcdf" />


playground_EPMS2_to_MS2.py
- Explores mapping XY to Y using a Unet and resnet style decoder. In Unet we mask all XY input where EP is not under a learned Parameter (proxy for contact radius). The idea being that enhancer promoter occur under some contact radius so all useful information should be under this threshold and having it as a learnable parameter we get an estimate from the model.
Learnings:
- accuracy of learned contact radius is sensitive to loss function and regularization but it works surprisingly well
- without signal as input and/or without gating on contact radius the model still seems to pick up trends

## WIP / exploratory / dropped:

playground_MS2_to_MS2_VAEstyle.py
- Explores self-supervised task of encoding and reconstructing Y VAE-style. Unet encoder to a Bernoulli random variable. Resnet style decoder. Convolutions can be toggled between causual or standard. Investigated whether upsampling the latent space so it was larger than output could function as a more fine-grained time-resolution where model could place 1s. Checked if these 1s aligned with pol2 loading or other biology - needs more exploration.

playground_EPMS2_to_MS2_conditioned_decoder.py
- Explores mapping XY to Y using a Unet masking everything under a learned Parameter (proxy for contact radius), feeding Y to decoder to condition to aid in reconstruction.

playground_EPMS2_to_states.py
- Explores mapping XY to state (simulation GT) using a Unet masking everything under a learned Parameter (proxy for contact radius),

feeding Y to decoder to condition that is disconnected from the computational graph so the model can not backprop on it
