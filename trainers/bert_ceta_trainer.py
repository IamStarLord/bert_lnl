import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from trainers.trainer import Trainer
from tqdm import tqdm
import wandb
import sys 

from trainers.early_stopper import EarlyStopper
from trainers.loss_noise_tracker import LossNoiseTracker
from trainers.wasserstein import wasserstein_distance
from trainers.wasserstein_2_metric import calculate_2_wasserstein_dist

from sklearn.metrics import classification_report

import numpy as np

class BertCETA_Trainer(Trainer):
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        super(BertCETA_Trainer, self).__init__(args, logger, log_dir, model_config, full_dataset, random_state)
        # enabling a store_model_flag here
        self.store_model_flag = True if args.store_model == 1 else False

    def train(self, args, logger, full_dataset):
        logger.info('Bert CETA Trainer: training started')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nl_set, ul_set, v_set, t_set, l2id, id2l = full_dataset
        logger.info(f'training size: {len(nl_set)}',)
        logger.info(f'validation size: {len(v_set)}' )
        logger.info(f'test size: {len(t_set)}')

        model = self.create_model(args, model="BertCETA")
        model = model.to(device)

        assert args.nl_batch_size % args.gradient_accumulation_steps == 0
        # compute after how many batch to perform gradient update. 
        # for gradient accumulation. 
        nl_sub_batch_size = args.nl_batch_size // args.gradient_accumulation_steps
        # training data loader
        nl_bucket = torch.utils.data.DataLoader(nl_set, batch_size=nl_sub_batch_size,
                                                shuffle=True,
                                                num_workers=0)
        # convert the bucket into an iter object
        nl_iter = iter(nl_bucket)

        # testing data loader
        t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.eval_batch_size,
                                               shuffle=False,
                                               num_workers=0)
        # if there is no validation dataset
        if v_set is None:
            logger.info('No validation set is used here')
            v_loader = None
        else:
            logger.info('Validation set is used here')
            # validation data loader
            v_loader = torch.utils.data.DataLoader(v_set, batch_size=args.eval_batch_size,
                                                   shuffle=False,
                                                   num_workers=0)

        num_training_steps = args.num_training_steps

        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, model)
        # print(optimizer_grouped_parameters)
        # sys.exit()
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                              num_training_steps=num_training_steps)
        ce_loss_fn = nn.CrossEntropyLoss()
        # w_loss_fn = wasserstein_distance
        w_loss_fn = calculate_2_wasserstein_dist

        if self.store_model_flag:
            # if store model flag is on, then save early_stopper_model
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            # if directory does not exist create it
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        # We log the validation accuracy, so, large_is_better should be set to True
        early_stopper = EarlyStopper(patience=args.patience, delta=0, save_dir=early_stopper_save_dir,
                                     large_is_better=True, verbose=False, trace_func=logger.info)

        noise_tracker_dir = os.path.join(self.log_dir, 'loss_noise_tracker')
        loss_noise_tracker = LossNoiseTracker(args, logger, nl_set, noise_tracker_dir)

        global_step = 0
        # parameter for ceta training 
        beta = 2.1
        # for n, p in model.bert.encoder.named_parameters():
        #     print(n)
        # sys.exit()

        for idx in tqdm(range(num_training_steps), desc=f'[BERT CETA Trainer] training'):
            # ce_loss_mean = 0.0

            # for these many steps accumulate the gradients over batches 
            # for i in range(args.gradient_accumulation_steps):
            model.train()
            try:
                # get next bach of data every gradient_accumulation_steps
                nl_batch = next(nl_iter)
            except:
                nl_iter = iter(nl_bucket)
                nl_batch = next(nl_iter)

            # computing the loss, performing a backward pass and returning the loss
            nll_loss = \
                self.forward_backward_noisy_batch(model, {'nl_batch': nl_batch}, ce_loss_fn, args, device)
            # accumulate gradients for the number of gradient_accumulation_steps
            # total loss, w.r.t encoder as well as both classifiers
            # ce_loss_mean += nll_loss  
            # after backward, compute optimizer step and zero out the gradients

            # clipping current gradients 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            # multiply wasserstein loss with -1 so ascent is calculated
            # compute backward 
            # multiply encoder parameters with -1 * beta
            lwd_loss = self.forward_backward_noisy_batch_wasserstein(model, {'nl_batch': nl_batch}, w_loss_fn, args, beta, device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step() 
            optimizer_scheduler.step()
            model.zero_grad()
            # use after a batch has been called
            # Update learning rate schedule
            
            global_step += 1

            wandb.log({'train/batch_loss/cross entropy loss': nll_loss,
                       'train/batch_loss/wasserstein loss': lwd_loss})

            # if evaluation is needed 
            if self.needs_eval(args, global_step):
                val_score = self.eval_model_with_both_labels(model, v_loader, device, fast_mode=args.fast_eval)
                test_score = self.eval_model(args, logger, t_loader, model, device, fast_mode=args.fast_eval)

                early_stopper.register(val_score['score_dict_n']['accuracy'], model, optimizer)

                wandb.log({'eval/loss/val_c_loss1': val_score['val_c_loss1'],
                           'eval/loss/val_c_loss2': val_score['val_c_loss2'],
                           'eval/loss/val_n_loss1': val_score['val_n_loss1'],
                           'eval/loss/val_n_loss2': val_score['val_n_loss2'],
                           'eval/score/val_c_acc': val_score['score_dict_c']['accuracy'],
                           'eval/score/val_n_acc': val_score['score_dict_n']['accuracy'],
                           'eval/score/test_acc': test_score['score_dict']['accuracy']}, step=global_step)

                loss_noise_tracker.log_loss(model, global_step, device)
                loss_noise_tracker.log_last_histogram_to_wandb(step=global_step, normalize=True, tag='eval/loss')

            if early_stopper.early_stop:
                break

        # save the best model 
        if args.save_loss_tracker_information:
            loss_noise_tracker.save_logged_information()
            self.logger.info("[WN Trainer]: loss history saved")
        best_model = self.create_model(args, model="BertCETA")
        best_model_weights = early_stopper.get_final_res()["es_best_model"]
        best_model.load_state_dict(best_model_weights)
        best_model = best_model.to(device)

        val_score = self.eval_model_with_both_labels(best_model, v_loader, device, fast_mode=False)
        test_score = self.eval_model(args, logger, t_loader, best_model, device, fast_mode=False)
        wandb.run.summary["best_score_on_val_n"] = test_score['score_dict']['accuracy']
        wandb.run.summary["best_val_n"] = val_score['score_dict_n']['accuracy']
        wandb.run.summary["best_val_c_on_val_n"] = val_score['score_dict_c']['accuracy']

    def eval_model_with_both_labels(self, model, v_loader, device, fast_mode):
        all_preds = []
        # all clean labels 
        all_y_c = []
        # all noisy labels 
        all_y_n = []
        # run evaluation
        model.eval()
        # clean data validation loss sum
        c_val_loss_sum1 = 0
        c_val_loss_sum2 = 0
        # noisy data validation loss sum
        n_val_loss_sum1 = 0
        n_val_loss_sum2 = 0
        # use the cross entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()

        # if running in fast_mode, use smaller batches
        if fast_mode:
            n_batch = len(v_loader)/10

        with torch.no_grad():
            for idx, t_batch in enumerate(v_loader):
                # get input ids, attention_mask, clean labels, noisy labels for the batch
                input_ids = t_batch['input_ids'].to(device)
                attention_mask = t_batch['attention_mask'].to(device)
                c_labels = t_batch['c_labels'].to(device)
                n_labels = t_batch['n_labels'].to(device)

                # get the logits
                y_pred1 = model(input_ids, attention_mask)['logits1']
                y_pred2 = model(input_ids, attention_mask)['logits2']
                # average both probabilities 
                total_pred = (y_pred1 + y_pred2) / 2
                # get predicted classes from the logits 
                predicted = torch.max(total_pred.cpu(), 1)[1]                # add to the loss for both clean and noisy labels
                # print(f"c_labels are {c_labels}")
                # print(f"predicted labels are {predicted}")
                # print(f"noisy labels are {n_labels}")
                # loss on classifier 1
                c_val_loss_sum1 += loss_fn(y_pred1, c_labels).item()
                # loss on classifier 2 
                c_val_loss_sum2 += loss_fn(y_pred2, c_labels).item()
                # loss on classifier 1
                n_val_loss_sum1 += loss_fn(y_pred1, n_labels).item()
                # loss on classifier 2
                n_val_loss_sum2 += loss_fn(y_pred2, n_labels).item()

                all_preds.extend(predicted.numpy())
                all_y_c.extend(list(c_labels.cpu()))
                all_y_n.extend(list(n_labels.cpu()))


                if fast_mode and idx > n_batch:
                    break

            num_val_samples = len(all_y_c)

            classification_score_dict_n = classification_report(all_y_n, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str_n = classification_report(all_y_n, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)

            classification_score_dict_c = classification_report(all_y_c, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str_c = classification_report(all_y_c, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)
            # c_val is the validation with respect to clean labels 
            # n_val is the validation with respect to noisy labels 
            
            # average loss on classifier 1
            c_val_loss_avg1 = c_val_loss_sum1/num_val_samples
            # average loss on classifier 2 
            c_val_loss_avg2 = c_val_loss_sum2/num_val_samples
            # average loss on classifer 1
            n_val_loss_avg1 = n_val_loss_sum1/num_val_samples
            # average loss on classifer 2 
            n_val_loss_avg2 = n_val_loss_sum2/num_val_samples

        return {'score_dict_n': classification_score_dict_n,
                'score_str_n': classification_score_str_n,
                'score_dict_c': classification_score_dict_c,
                'score_str_c': classification_score_str_c,
                'val_c_loss1': c_val_loss_avg1,
                'val_c_loss2': c_val_loss_avg2, 
                'val_n_loss1': n_val_loss_avg1,
                'val_n_loss2': n_val_loss_avg2}

    def eval_model(self, args, logger, t_loader, model, device, fast_mode):
        all_preds = []
        all_y = []
        model.eval()

        if fast_mode:
            # divide batches by 10 
            n_batch = len(t_loader)/10

        with torch.no_grad():
            for idx, t_batch in enumerate(t_loader):
                input_ids = t_batch['input_ids'].to(device)
                attention_mask = t_batch['attention_mask'].to(device)
                c_targets = t_batch['c_labels'].to(device)

                y_pred1 = model(input_ids, attention_mask)['logits1']
                y_pred2 = model(input_ids, attention_mask)['logits2']
                y_pred_total = (y_pred1 + y_pred2) / 2
                predicted = torch.max(y_pred_total.cpu(), 1)[1]
                all_preds.extend(predicted.numpy())
                all_y.extend(list(c_targets.cpu()))
                if fast_mode and idx > n_batch:
                    break

            classification_score_dict = classification_report(all_y, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str = classification_report(all_y, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)

        return {'score_dict': classification_score_dict, 'score_str': classification_score_str}

    def forward_backward_noisy_batch_wasserstein(self, model, data_dict, w_loss_fn, args, beta, device):

        nl_databatch = data_dict['nl_batch']
        input_ids = nl_databatch['input_ids']
        attention_mask = nl_databatch['attention_mask']
        n_labels = nl_databatch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)

        # forward pass of the model 
        outputs1 = model(input_ids, attention_mask)['logits1']
        outputs2 = model(input_ids, attention_mask)['logits2']

        # 1 wasserstein loss 
        # outputs1_flat = torch.flatten(outputs1)
        # outputs2_flat = torch.flatten(outputs2)

        # compute wasserstein loss between both probability distributions 
        # lwd = w_loss_fn(outputs1_flat, outputs2_flat)

        # calculation with 2- wasserstein loss metric 
        lwd = w_loss_fn(outputs1, outputs2)
        
        # farzi step 
        if args.gradient_accumulation_steps > 1:
            lwd = lwd / args.gradient_accumulation_steps 
        # multiply loss with -1 to perform gradient accumulation  
        lwd = -1*lwd
        lwd.backward()

        # multiply encoder parameters with -1 and beta to perform 
        for param in model.bert.encoder.layer.parameters():
            param.grad = -1*beta*param.grad 

        # return the loss
        return (-1*lwd)

    # freeze parameter to tell what to freeze during training
    def forward_backward_noisy_batch(self, model, data_dict, loss_fn, args, device, freeze=None):

        nl_databatch = data_dict['nl_batch']
        input_ids = nl_databatch['input_ids']
        attention_mask = nl_databatch['attention_mask']
        n_labels = nl_databatch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)

        # forward pass of the model 
        outputs1 = model(input_ids, attention_mask)['logits1']
        outputs2 = model(input_ids, attention_mask)['logits2']
        loss1 = loss_fn(outputs1, n_labels)
        loss2 = loss_fn(outputs2, n_labels)

        if args.gradient_accumulation_steps > 1:
            loss1 = loss1 / args.gradient_accumulation_steps
            loss2 = loss2 / args.gradient_accumulation_steps

        total_loss = loss1 + loss2
        # this should conceptually compute gradients with respect to 
        # classifier1 
        # classifier2 
        # and the encoder 
        # i.e. CETA stage 1 
        total_loss.backward()

        return total_loss.item()

        
        

        

    