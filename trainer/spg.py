from tqdm import tqdm

import torch
from torch import Tensor, nn


from trainer.spu import SPU, zerolike_params_dict




class OtherTasksLoss(nn.Module):
    def forward(self, t: Tensor) -> Tensor:
        return t.sum()
    
    
    
class SPG(SPU):
    
    @classmethod
    def standardize(cls, x: Tensor) -> Tensor:
        sh = x.shape
        x = x.view(-1)

        ret = (x - x.mean()) / x.std()

        return ret.view(*sh)
    
    def standardize_pm1(self, x: Tensor) -> Tensor:
        if torch.all(x == 0):
            pass
        else:
            x = self.standardize(x)
        ret = torch.tanh(x)

        return ret
    
    def compute_update_importance(self, model, dataloader, recent_task):

        # Get importance
        curr_importance = self._get_importance(model, dataloader,recent_task)
        if not self.importance_computed:
            self.importance = curr_importance
            self.importance_computed = True
            return
        else:
            # Update importance
            for name in self.importance.keys():
                self.importance[name] = torch.max(self.importance[name],
                                      curr_importance[name].data)
    
    def _get_importance(self, model, dataloader,  recent_task):


        # Initialize importance matrix
        importance = dict(zerolike_params_dict(model, device=self.args.device))

        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        size = 0
        
        for num_batch, batch in enumerate(tqdm(dataloader)):
            # Get batch
            images, _, texts = batch
            images = images.to(self.args.device)
            texts = texts.to(self.args.device)

            # Forward pass
            model.zero_grad()
            logits_per_image, logits_per_text = model(images, texts)


            ground_truth = torch.arange(len(images), dtype=torch.long, device=self.args.device)    
            if recent_task:
                
                loss_img = nn.CrossEntropyLoss()
                loss_txt = nn.CrossEntropyLoss()
                loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            else:
                loss_img = OtherTasksLoss()
                loss = loss_img(logits_per_image)

            loss.backward()

            # Accumulate importance
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        importance[name].data += param.grad.clone() * len(images)
            size += len(images)
            

        # Normalize importance
        importance = {
            name: self.standardize_pm1(importance[name] / size).abs() 
            for name in importance.keys()
        }


        return importance
    
    def compute_importance(self, dataset, model, task):
        if task == 0:
            for n, w in model.named_parameters():
                self.mask[n] = torch.ones_like(w)
        else:

        

            prev_set = dataset.get_dataset(task, is_train=False, with_buffer=False)
            loader = self.get_loader(prev_set)
            print (f'Before start task {task}, compute importance for the task {task-1}...')
            self.compute_update_importance(model, loader, recent_task = True)
            self.compute_update_importance(model, loader, recent_task = False)

            for n,w in model.named_parameters():
                self.mask[n] = 1-self.importance[n]