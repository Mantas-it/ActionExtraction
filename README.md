# Article repository

"Deep learning-based automatic action extraction from structured chemical synthesis procedures".

Available online: - doi: --

Required: python 3.8; tensorflow 2.4.1; RDkit 2021.03; The code is in Jupyter Notebook/Jupyter lab format for more convenient prototyping, Anaconda environment is recommended. GPU with support for CUDA 10.1 or higher is recommended

The prepared EPO/USPTO dataset of chemical procedures can be downloaded [here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EawEVnHXkg9FnxEB2LE1ujsBCsSe2NF2viC454L1Jaihmg?e=VGOOgz). (880 MB zipped, .csv file with tab as a separator)

The Translation (text generation) model can be downloaded [here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EcZb9F_L75hBkZF8DVuQhkoBxEbdkuFI81jSWRWYD_6PtA?e=SGJvgg). (>500 MB zipped, openNMT model)

Convert raw paragraphs to structured format via cmd:

```python
onmt-main --checkpoint_path model_folder\ckpt-2600 --config run_file.yml --auto_config --mixed_precision infer --features_file input.txt --predictions_file output.txt
```

All text of organic chemistry patents will be available after the associated article has been published. The raw text is substancial in size (about 200GB). A more detailed guide for exploration of the datasets and a tutorial will be made public.  
