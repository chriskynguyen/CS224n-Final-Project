'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

import quadprog # used to solve QP

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.sentiment_classifier = nn.Linear(config.hidden_size, 5)

        self.paraphrase_classifier = nn.Linear(config.hidden_size*2, 1)

        self.similarity_classifier = nn.Linear(config.hidden_size*2, 1)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        outputs = self.bert(input_ids, attention_mask) # {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
        pooled_output = outputs['pooler_output'] # hidden state of [CLS] token
        pooled_output = self.dropout(pooled_output)
        return pooled_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        output = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(output)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        combined_output = torch.cat((output_1, output_2), dim=1)
        logits = self.paraphrase_classifier(combined_output)
        return logits.float()

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        combined_output = torch.cat((output_1, output_2), dim=1)
        logits = self.similarity_classifier(combined_output)
        return logits


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    # Sentiment Analysis dataset (SST)
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Paraphrase Detection dataset (Quora)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    # Semantic Textual Similarity (STS)
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr

    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = [0. for i in range(3)]
    steps_per_epoch = 2000

    # Calculate warmup steps (10% of total steps)
    warmup_steps = steps_per_epoch * 0.1
    step = 0
    # Define a learning rate scheduler for linear warmup and decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda= lambda step:args.lr * min((step + 1) / warmup_steps, 1.0) * max(1.0 - (step + 1 - warmup_steps) / (steps_per_epoch - warmup_steps), 0)
    )

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        # sampling method: anneal
        probs = [8544, 141498, 6041] # number of training examples
        alpha = 1. - 0.8 * epoch / (args.epochs - 1)
        probs = [p**alpha for p in probs]
        tot = sum(probs)
        probs = [p/tot for p in probs]

        model.train()
        train_loss = [0. for i in range(3)] # separate loss for each task
        num_batches = [0. for i in range(3)]
        task_id = 0
        for step in tqdm(range(steps_per_epoch), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            task_id = np.random.choice(3, p=probs)

            # Load batch based on task ID
            if task_id == 0:
                batch = next(iter(sst_train_dataloader))
                input_ids, attention_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels']
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model.predict_sentiment(input_ids.to(device), attention_mask.to(device))
                loss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / args.batch_size
                
            elif task_id == 1:
                batch = next(iter(para_train_dataloader))
                input_ids1, attention_mask1, input_ids2, attention_mask2, labels = (
                    batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                labels = labels.to(device).float()
                optimizer.zero_grad()
                logits = model.predict_paraphrase(input_ids1.to(device), attention_mask1.to(device),
                                                input_ids2.to(device), attention_mask2.to(device))
                loss = F.binary_cross_entropy_with_logits(logits, labels.view(-1, 1), reduction='sum') / args.batch_size
            elif task_id == 2:
                batch = next(iter(sts_train_dataloader))
                input_ids1, attention_mask1, input_ids2, attention_mask2, labels = (
                    batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                labels = labels.to(device).float()
                optimizer.zero_grad()   
                logits = model.predict_similarity(input_ids1.to(device), attention_mask1.to(device),
                                                input_ids2.to(device), attention_mask2.to(device))
                loss = F.mse_loss(logits, labels.view(-1, 1), reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step() # update model parameters
            scheduler.step() # update learning rate

            train_loss[task_id] += loss.item() 
            num_batches[task_id] += 1

        train_avg_loss = [loss / num_batches[i] if num_batches[i] != 0 else 0 for i, loss in enumerate(train_loss)] # Average loss over all batches

        train_acc_sst, _, _, train_acc_para, _, _, train_acc_sts, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_acc_sst, _, _, dev_acc_para, _, _, dev_acc_sts, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        dev_acc = [dev_acc_sst, dev_acc_para, dev_acc_sts]
        """
        dev_count = 0
        # make sure all accuracies are increasing other 
        for i in range(len(dev_acc)):
            if dev_acc[i] > best_dev_acc[i]:
                best_dev_acc[i] = dev_acc[i]
                dev_count += 1
        if dev_count == 3:
            save_model(model, optimizer, args, config, args.filepath)
        """
        if any(dev_acc[i] > best_dev_acc[i] for i in range(len(dev_acc))):
            best_dev_acc = [max(dev_acc[i], best_dev_acc[i]) for i in range(len(dev_acc))]
            save_model(model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch}: train avg loss sst:: {train_avg_loss[0]:.3f}, train avg loss para:: {train_avg_loss[1]:.3f}, train avg loss sts:: {train_avg_loss[2]:.3f}")


class GEM(nn.Module):
    def __init__(self, args, model, device, optimizer, memory_size, num_tasks):
        """
        model: MultitaskBERT
        memory_size: number of examples to store in memory
        num_tasks: number of tasks
        """
        super(GEM, self).__init__()
        self.model = model # BERT model
        self.memory_size = memory_size
        self.args = args
        self.opt = optimizer
        self.device = device
        self.margin = self.args.memory_strength
        # allocate episodic memory
        self.memory_data = [[] for _ in range(num_tasks)] # each element stores the batched data dictionary

        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), num_tasks)
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        if self.args.use_gpu:
            self.memory_data = self.memory_data
            self.grads = self.grads.to(device)

    def forward(self, task_id, batch):
        if task_id == 0:
            input_ids, attention_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels']
            labels = labels.to(self.device)
            logits = self.model.predict_sentiment(input_ids.to(self.device), attention_mask.to(self.device))
        elif task_id == 1:
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels = (
                batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
            labels = labels.to(self.device).float()
            logits = self.model.predict_paraphrase(input_ids1.to(self.device), attention_mask1.to(self.device),
                                            input_ids2.to(self.device), attention_mask2.to(self.device))
        elif task_id == 2:
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels = (
                batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
            labels = labels.to(self.device).float()
            logits = self.model.predict_similarity(input_ids1.to(self.device), attention_mask1.to(self.device),
                                            input_ids2.to(self.device), attention_mask2.to(self.device))
        return logits, labels

    def store_grad(self, parameters, grads, grad_dims, task_id):
        """
            This stores parameter gradients of past tasks. 
            grads: gradients
            grad_dims: list with number of parameters per layers
        """
        # store the gradients
        grads[:, task_id].fill_(0.0)
        count = 0
        for param in parameters():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                grads[begin: end, task_id].copy_(param.grad.data.view(-1))
            count += 1

    def overwrite_grad(self, parameters, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        count = 0
        for param in parameters():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1

    def project2cone2(self, gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))

    def observe(self, task_id, batch):
        # update memory
        if task_id != self.old_task:
            self.observed_tasks.append(task_id)
            self.old_task = task_id

        # update memory to store examples from current task
        if len(self.memory_data[task_id]) < self.memory_size:
            self.memory_data[task_id].append(batch)
        else:
            self.memory_data[task_id][self.mem_cnt] = batch
            self.mem_cnt = (self.mem_cnt + 1) % self.memory_size # resets mem_cnt if equal to memory_size

        # compute gradient on previous tasks 
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1): # only get past tasks not recently added task
                self.opt.zero_grad()
                #fwd/bwd on examples in memory
                past_task = self.observed_tasks[tt]
                logits, labels = self.forward(past_task, self.memory_data[past_task][0])
                if past_task == 0:
                    ptloss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / self.args.batch_size
                elif past_task == 1:
                    ptloss = F.binary_cross_entropy_with_logits(logits, labels.view(-1, 1), reduction='sum') / self.args.batch_size
                else:
                    ptloss = F.mse_loss(logits, labels.view(-1, 1), reduction='sum') / self.args.batch_size
                ptloss.backward()
                # store gradients
                self.store_grad(self.model.bert.parameters, self.grads, self.grad_dims, past_task)

        # compute gradient on current batch
        self.opt.zero_grad()
        logits, labels = self.forward(task_id, batch)
        if task_id == 0:
            loss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / self.args.batch_size
        elif task_id == 1:
            loss = F.binary_cross_entropy_with_logits(logits, labels.view(-1, 1), reduction='sum') / self.args.batch_size
        else:
            loss = F.mse_loss(logits, labels.view(-1, 1), reduction='sum') / self.args.batch_size
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            self.store_grad(self.model.bert.parameters, self.grads, self.grad_dims, task_id)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.args.use_gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, task_id].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                self.project2cone2(self.grads[:, task_id].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                self.overwrite_grad(self.model.bert.parameters, self.grads[:, task_id], self.grad_dims)
        
        self.opt.step()
        return loss.item()

def train_multitask_GEM(args):
    '''Train MultitaskBERT using GEM.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    # Sentiment Analysis dataset (SST)
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Paraphrase Detection dataset (Quora)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    # Semantic Textual Similarity (STS)
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    
    
    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    tasks = ["SST", "QQP", "STS"]
    loaders = [sst_train_dataloader, para_train_dataloader, sts_train_dataloader]
    gem = GEM(args, model, device, optimizer, 1, 3)

     # Calculate warmup steps (10% of total steps)
    steps_per_task = [len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader)]
    warmup_steps_per_task = [steps * 0.1 for steps in steps_per_task]
    step = 0
    # Define a learning rate scheduler for linear warmup and decay
    schedulers = [torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step, warmup_steps=warmup_steps, total_steps=total_steps: args.lr * min((step + 1) / warmup_steps, 1.0) * max(1.0 - (step + 1 - warmup_steps) / (total_steps - warmup_steps), 0)
    ) for warmup_steps, total_steps in zip(warmup_steps_per_task, steps_per_task)]
    for epoch in range(args.epochs):
        model.train()
        train_loss = [0. for i in range(3)] # separate loss for each task
        num_batches = [0. for i in range(3)]
        for task_id, dataloader in enumerate(loaders): # train task sequentially
            scheduler = schedulers[task_id]
            for batch in tqdm(dataloader, desc=f'train-{epoch}, {tasks[task_id]}', disable=TQDM_DISABLE):
                train_loss[task_id] += gem.observe(task_id, batch)
                scheduler.step()
                num_batches[task_id] += 1

        train_avg_loss = [loss / num_batches[i] for i, loss in enumerate(train_loss)] # Average loss over all batches

        train_acc_sst, _, _, train_acc_para, _, _, train_acc_sts, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_acc_sst, _, _, dev_acc_para, _, _, dev_acc_sts, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        dev_acc = [dev_acc_sst, dev_acc_para, dev_acc_sts]
        if any(acc > best_dev_acc for acc in dev_acc):
            best_dev_acc = max(dev_acc)
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train avg loss sst:: {train_avg_loss[0]:.3f}, train avg loss para:: {train_avg_loss[1]:.3f}, train avg loss sts:: {train_avg_loss[2]:.3f}")
                

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=16)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    parser.add_argument("--use_gem", action='store_true')
    parser.add_argument("--memory_strength", type=float, default=0.5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    if args.option == 'finetune' and args.use_gem:
        train_multitask_GEM(args)
    else:
        train_multitask(args)
    test_multitask(args)
