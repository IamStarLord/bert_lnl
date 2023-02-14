# create ensembles 
# perform loss sorting 
import torch
import os 
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from models.text_bert import TextBert
from models.bert_ceta import BertCETA

class Ensembler:
    def __init__(self, logger, log_dir, model_config, args, full_dataset):
        self.logger = logger 
        self.model_config = model_config 
        self.log_dir = log_dir 
        self.args = args
        self.full_dataset = full_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self, model="TextBert"):
        if model == "TextBert":
            model = TextBert(self.model_config, None, self.args)
        elif model == "BertCETA":
            model = BertCETA(self.model_config, None, self.args)
        return model

    def load_model(self, path):
        model = self.create_model()
        # print(model)
        model.load_state_dict(torch.load(path))
        # put model in eval mode to disable to dropout and batchnorm 
        # model.eval()
        return model 

    def eval_model(self, t_loader, model, device):
        # model predictions
        model_preds = []
        # clean labels
        clean_labels = []
        # model logits 
        model_logits = []
        # put the model in eval mode here 
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for idx, t_batch in enumerate(t_loader):
                input_ids = t_batch['input_ids'].to(device)
                attention_mask = t_batch['attention_mask'].to(device)
                c_targets = t_batch['c_labels'].to(device)
                
                # model logits 
                model_pred = model(input_ids, attention_mask)['logits']
                # predicted labels
                predicted = torch.max(model_pred.cpu(), 1)[1]
                model_preds.extend(predicted.numpy())
                clean_labels.extend(list(c_targets.cpu()))
                # print(f"shape of logits is {y_pred.shape}")
                model_logits.append(model_pred)

            # classification_score_dict = classification_report(all_y, np.array(all_preds).flatten(),
            #                                                   target_names=self.label_list, output_dict=True)
            # classification_score_str = classification_report(all_y, np.array(all_preds).flatten(),
                                                            #  target_names=self.label_list, output_dict=False)
        model_logits = torch.cat(model_logits)
        print(f"shape of logits after cating {model_logits.shape}")
        return {'model_predictions': model_preds, 'clean_labels': clean_labels, 'model_logits': model_logits}
    
    def model_averaging(self):
        # args are string arguments 
        # load models 
        # 0th model => first model 
        loaded_models = []
        for member in self.args.members:
            # path to the trained model 
            path = os.path.join(os.path.abspath("[LOG_ROOT]"), member, "early_stopper_model/model_dict.pt")
            loaded_models.append(self.load_model(path))
        # get the test dataset  
        _, _, _, t_set, _, _ = self.full_dataset
        # make a test loader 
        # evaluate the model on the test data 
        t_loader = torch.utils.data.DataLoader(t_set, batch_size=self.args.eval_batch_size,
                                               shuffle=False,
                                               num_workers=0)
        # dictionary to save model inference

        inferences = {}
        # evaluate the data on each model 
        for idx, model in enumerate(loaded_models):
            inferences[idx] = self.eval_model(t_loader, model, self.device)
        
        # average all model logits
        avg_logits = inferences[0]["model_logits"]
        # print("first logit: ")
        # print(avg_logits)
        for m, inference in inferences.items():
            if m == 0:
                continue
            avg_logits += inference["model_logits"]
        # print(avg_logits)
        avg_logits = avg_logits / len(loaded_models)

        avg_predictions = torch.max(avg_logits.cpu(), 1)[1]
        clean_labels = [lb.item() for lb in inferences[0]["clean_labels"]]

        # print accuracy of individual models 
        # model1_preds = inferences[0]['model_predictions']
        # print(f"Model 1 accuracy is {self.accuracy(clean_labels, model1_preds)}")

        # model2_preds = inferences[1]['model_predictions']
        # print(f"Model 2 accuracy is {self.accuracy(clean_labels, model2_preds)}")

        # model3_preds = inferences[2]['model_predictions']
        # print(f"Model 3 accuracy is {self.accuracy(clean_labels, model3_preds)}")  

        # model4_preds = inferences[3]['model_predictions']
        # print(f"Model 4 accuracy is {self.accuracy(clean_labels, model4_preds)}")  

        # model5_preds = inferences[4]['model_predictions']
        # print(f"Model 5 accuracy is {self.accuracy(clean_labels, model5_preds)}")    

        return avg_predictions, clean_labels

    def accuracy(self, targets, predictions):
        """ 
        Takes 1D array of predictions and targets
        """
        # return (np.array(targets) == np.array(predictions)).sum() / len(targets)
        return accuracy_score(targets, predictions)


        