# XAI_Timeseries_model
Explainable AI model (XAI) for timeseries data.

The goal is to learn Y (MS2 intensities, binds nascant RNA to visualize transcription) from X (enhancer promoter distances) and in the process learn what in enhancer promoter interactions are important, thus uncovering important aspects of gene regulation. Potentially, utilize Y in input for self-supervision.

Main playground scripts:

playground_EPMS2_to_states.py
- Explores mapping XY to Y using a Unet and resnet style decoder. In Unet we mask all XY input where EP is not under a learned Parameter (proxy for contact radius). The idea being that enhancer promoter occur under some contact radius so all useful information should be under this threshold and having it as a learnable parameter we get an estimate from the model.
Learnings:
- accuracy of learned contact radius is sensitive to loss function and regularization but it works surprisingly well

<img width="1189" height="889" alt="image" src="https://github.com/user-attachments/assets/6aa76dec-b3f9-4a67-9051-abf32ade93d5" />



WIP / exploratory / dropped:

playground_MS2_to_MS2_VAEstyle.py
- Explores self-supervised task of encoding and reconstructing Y VAE-style. Unet encoder to a Bernoulli random variable. Resnet style decoder. Convolutions can be toggled between causual or standard. Investigated whether upsampling the latent space so it was larger than output could function as a more fine-grained time-resolution where model could place 1s. Checked if these 1s aligned with pol2 loading or other biology - needs more exploration.

playground_EPMS2_to_MS2_conditioned_decoder.py
- Explores mapping XY to Y using a Unet masking everything under a learned Parameter (proxy for contact radius), feeding Y to decoder to condition to aid in reconstruction.

playground_EPMS2_to_states.py
- Explores mapping XY to state (simulation GT) using a Unet masking everything under a learned Parameter (proxy for contact radius),

feeding Y to decoder to condition that is disconnected from the computational graph so the model can not backprop on it
