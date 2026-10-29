# ArogyaNet hypotheses (written before any ArogyaNet result)

| # | Design decision | What it is for | What I expect it to change |
|---|---|---|---|
| D1 | Residual (skip) connections around each block | Give gradients a short path so the network trains stably from scratch on a few thousand images. | Removing them will lower validation PR-AUC by under 1 point and make results vary more between seeds, but the average change will be small because the network is shallow. |
| D2 | Squeeze-and-excitation (channel attention) in each block | Let the network weight feature channels by how useful they are across the whole image, for example diffuse opacities over local edges. | Removing it will slightly lower recall on pneumonia, but I expect the difference to be inside the seed-to-seed noise. |
| T8 | Metadata branch (mean intensity, aspect ratio, image area) | Test whether acquisition details add information beyond the pixels. | If the audit shows these numbers separate the classes, the metadata model will score higher on the internal test set but drop more on the external hospital, because it is learning the scanner or site, not the disease. |