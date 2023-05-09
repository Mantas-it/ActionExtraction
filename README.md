# Article repository

"Deep learning-based automatic action extraction from structured chemical synthesis procedures".

Available online: - doi: --

Required: python 3.8; tensorflow 2.4.1; RDkit 2021.03; The code is in Jupyter Notebook/Jupyter lab format for more convenient prototyping, Anaconda environment is recommended. GPU with support for CUDA 10.1 or higher is recommended

### Basic use of optimal models

# Task 1 - paragraph classification
To classify a paragraph, download files in folder 'no 2 From organic chemistry patents to procedure paragraphs' and use SentencePiece library along with tensorflow and load the model:
If the predition is > 0.5, it's likely an organic synthesis procedure. 
```python
import sentencepiece as spm
import tensorflow as tf
sp = spm.SentencePieceProcessor(model_file='full_30k.model')
text_x = sp.encode('Text paragraph.')
model = tf.keras.models.load_model('classification_model.h5')

# Result:
print(model.predict([text_x])[0][0])
```

# Task 2 - convert unstructured paragraphs into a structured format

Download the optimal text generation model from the 'Large files' section below and use OpenNMT via the command line:
```python
onmt-main --checkpoint_path model_folder\ckpt-2600 --config run_file.yml --auto_config --mixed_precision infer --features_file input.txt --predictions_file output.txt
```

### Large files

The prepared EPO/USPTO dataset of chemical procedures can be downloaded [here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EawEVnHXkg9FnxEB2LE1ujsBCsSe2NF2viC454L1Jaihmg?e=VGOOgz). (880 MB zipped, .csv file with tab as a separator)

The Translation (text generation) model can be downloaded [here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EcZb9F_L75hBkZF8DVuQhkoBxEbdkuFI81jSWRWYD_6PtA?e=SGJvgg). (>500 MB zipped, openNMT model)

The training and fine-tuning datasets for translation (text generation task) can be downloaded [here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EWNOxzDRJCpCpxzuphGHZhgBc5SbO8A2jMSVRoncN8VkfQ?e=XzAxLE) (530 MB)

The fine-tuned T5 model can be downloaded [here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/ER0_aoQ-BThGqMkt5Y5t_pgBP2_yKKIPvD8knZyT2Z1mvw?e=Gu7x7y) (640 MB) 

All text of organic chemistry patents in raw format will be available after the associated article has been published. The raw text is substantial in size (about 200GB). A more detailed guide for exploration of the datasets and a tutorial will be made public.  

### Additional notes and advanced use
Methods and code for raw patent data processing can be found in the folder 'no 1 From raw patent data to organic chemistry patents'. 


