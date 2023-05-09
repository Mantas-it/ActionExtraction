import random
from torch.utils.data import Dataset, DataLoader
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import random
from torch.utils.data import Dataset, DataLoader
import torch
class ChemDataset(Dataset):
    def __init__(self, input_data, output_data, tokenizer, max_length=256):
        self.input_data = input_data
        self.output_data = output_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        

        input_encodings = self.tokenizer(input_data[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        output_encodings = self.tokenizer(output_data[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        
        
        return (input_encodings["input_ids"][0], input_encodings["attention_mask"][0],  output_encodings["input_ids"][0])

    def __len__(self):
        return len(self.input_data)
    
class ChemDataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([torch.where(item[2] != self.tokenizer.pad_token_id, item[2], -100) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}    
    
from typing import Sequence, List
from nltk.translate.bleu_score import corpus_bleu
def modified_bleu(truth: List[str], pred: List[str]) -> float:

    references = [sentence.split() for sentence in truth]
    candidates = [sentence.split() for sentence in pred]


    references = [r + max(0, 4 - len(r)) * [''] for r in references]
    candidates = [c + max(0, 4 - len(c)) * [''] for c in candidates]

    refs = [[r] for r in references]
    return corpus_bleu(refs, candidates)

if __name__ == '__main__':
    import evaluate
    metric = evaluate.load("sacrebleu")
    from transformers import AutoModelForSeq2SeqLM

    


    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    
    model.config.num_beams = 1
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer
    import torch
    import torch.nn as nn
    import os
    import random
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


    with open("src-train.txt", "r",encoding='utf-8') as f:
        input_data = [line.strip() for line in f.readlines()]
        #random.shuffle(input_data)
    with open("tgt-train.txt", "r",encoding='utf-8') as f:
        output_data = [line.strip() for line in f.readlines()]
        #random.shuffle(output_data)
    zipped = list(zip(input_data, output_data))
    random.shuffle(zipped)
    input_data, output_data = zip(*zipped)
    with open("src-valid_sm.txt", "r",encoding='utf-8') as f:
        val_input_data = [line.strip() for line in f.readlines()][:200]
    with open("tgt-valid_sm.txt", "r",encoding='utf-8') as f:
        val_output_data = [line.strip() for line in f.readlines()][:200]

    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    
    data_collator = ChemDataCollator(tokenizer)
    
    train_dataset = ChemDataset(input_data, output_data, tokenizer)
    val_dataset = ChemDataset(val_input_data, val_output_data, tokenizer)


    
    
    from transformers import Seq2SeqTrainer, TrainingArguments,Seq2SeqTrainingArguments
    import numpy as np
    
    def postprocess_text(preds, labels):
        preds = [list(pred.split('</s>'))[0].replace('<unk>','') for pred in preds]
        labels = [list(label.split('</s>'))[0].replace('<unk>','') for label in labels]

        return preds, labels
    
    
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        

        preds = np.argmax(predictions[0], axis=2)

 
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
 
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)


        decoded_preds1, decoded_labels1 = postprocess_text(decoded_preds, decoded_labels)

        decoded_preds2, decoded_labels2 = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds1, references=decoded_labels1)
        result = {"bleu": result["score"]}
        result["bleu_modified"]=modified_bleu(decoded_labels1,decoded_preds1)*100

        result = {k: round(v, 4) for k, v in result.items()}
        print(decoded_preds2[0])
        print(decoded_labels2[0])
        print(decoded_preds2[1])
        print(decoded_labels2[1])
        print(decoded_preds2[2])
        print(decoded_labels2[2])
        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir="/run1",
        num_train_epochs=10,
        per_device_train_batch_size=4,

        save_steps=70,
        learning_rate=0.000001,
        evaluation_strategy = 'no',
        #fp16 can be turned to False
        fp16=True,                 
        dataloader_num_workers=0,
        logging_steps=20, 
        gradient_accumulation_steps=64,
 
    )
    trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset =val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
    trainer.train()
