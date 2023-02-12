import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel


# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class BertCETA(nn.Module):

  def __init__(self, model_config, bert_backbone, args):
    super(BertCETA, self).__init__()
    self.num_classes = model_config['num_classes']

    assert bert_backbone is None, 'we do not support training based on provided checkpoints yet'
    # options such as bert_base_uncased and bert_base_cased
    self.bert = BertModel.from_pretrained(args.model_name)
    # use just the BERT embeddings
    if args.freeze_bert:
      for param in self.bert.parameters():
        param.requires_grad = False
    # dropout layer 
    self.drop = nn.Dropout(p=model_config['drop_rate'])
    # two linear classifiers 
    self.out1 = nn.Linear(self.bert.config.hidden_size, self.num_classes)
    self.out2 = nn.Linear(self.bert.config.hidden_size, self.num_classes)


  def forward(self, input_ids, attention_mask):
    bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    cls_repr = bert_out[0][:, 0, :]
    pooler_repr = bert_out['pooler_output']
    output = self.drop(pooler_repr)
    # we get two different logits from two classifers 
    logits1 = self.out1(output)
    logits2 = self.out2(output)

    return {'logits1': logits1, 'logits2': logits2,'cls_repr': cls_repr, 'pooler_repr': pooler_repr}

