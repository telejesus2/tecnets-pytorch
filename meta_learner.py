import torch
import torch.nn.functional as F
import numpy as np
from network.utils import plot_grad_flow, register_hooks

class MetaLearner(object):

    def __init__(self, train_gen, val_gen, emb_mod, ctr_mod, opt, eval_sim=None, eval_emb=None,
                 lambda_embedding=1.0, lambda_support=0.1, lambda_query=0.1):
        self.emb_mod = emb_mod
        self.ctr_mod = ctr_mod
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.opt = opt
        self.eval_sim = eval_sim
        self.eval_emb = eval_emb

        # loss weights
        self.lambda_embedding = lambda_embedding
        self.lambda_support = lambda_support
        self.lambda_query = lambda_query

    def meta_train(self, epoch, train_emb=True, train_ctr=True, control=True, writer=None, log_interval=10):
        if train_ctr and not control:
            raise RuntimeError('Cannot train the control module without evaluating it.')
        self.emb_mod.train(train_emb)
        self.ctr_mod.train(train_ctr)
        for param in self.emb_mod.parameters():
            param.requires_grad = train_emb
        for param in self.ctr_mod.parameters():
            param.requires_grad = train_ctr

        running_loss = 0
        running_size = 0
        for batch_idx, inputs in enumerate(self.train_gen):
            loss, loss_emb, loss_ctr_q, loss_ctr_U, acc_emb = 0, 0, 0, 0, 0

            # torch.autograd.set_detect_anomaly(True)

            emb_outputs = self.emb_mod.forward(inputs) 
            loss_emb = self.lambda_embedding * emb_outputs['loss_emb']
            acc_emb = emb_outputs['acc_emb']
            if train_emb:
                loss += loss_emb

            if control: 
                inputs.update(emb_outputs)
                ctr_outputs = self.ctr_mod.forward(inputs)    
                loss_ctr_U = self.lambda_support * ctr_outputs['loss_ctr_U']
                loss_ctr_q = self.lambda_query *  ctr_outputs['loss_ctr_q'] 
                if train_ctr:
                    loss += loss_ctr_U + loss_ctr_q

            self.opt.zero_grad()
            # loss.register_hook(lambda grad: print(grad))
            # register_hooks(loss)
            loss.backward()
            
            # plot_grad_flow(self.ctr_mod.ctr_net.named_parameters(), 'ctr')
            # plot_grad_flow(self.emb_mod.emb_net.named_parameters(), 'emb')
            # for name, param in self.ctr_mod.ctr_net.named_parameters():
            #     print(name, param.grad.norm())
            # clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), val)
            # for param in model.parameters():
            #     param.register_hook(lambda grad: grad.clamp_(-val, val))
            self.opt.step()

            # torch.autograd.set_detect_anomaly(False)

            batch_size = len(inputs[list(inputs)[0]])
            running_size += batch_size
            loss_total = loss_emb + loss_ctr_U + loss_ctr_q
            if batch_idx % log_interval == 0:
                 print('Train Epoch: {} \t[{}/{} \t({:.0f}%)]\tLoss: {:.6f}\tTot: {:.6f}\tEmb: {:.6f}\tCtrQ: {:.6f}\tCtrU: {:.6f}'.format( 
                    epoch, running_size, len(self.train_gen.dataset), 100. * (batch_idx + 1) / len(self.train_gen),
                    loss.data.item(), loss_total.data.item(), loss_emb.data.item(), loss_ctr_q.data.item(), loss_ctr_U.data.item()))

            if writer is not None:
                writer.add_scalar('loss', loss, epoch)
                writer.add_scalar('loss_emb', loss_emb, epoch)
                writer.add_scalar('loss_ctr_U', loss_ctr_U, epoch)
                writer.add_scalar('loss_ctr_q', loss_ctr_q, epoch)
                writer.add_scalar('loss_all', loss_total, epoch)

    def meta_valid(self, epoch, train_emb=True, train_ctr=True, control=True, writer=None):
        self.emb_mod.eval()
        self.ctr_mod.eval()
        with torch.no_grad():
            
            val_loss, val_loss_emb, val_loss_ctr_q, val_loss_ctr_U, val_acc_emb = 0, 0, 0, 0, 0
            for batch_idx, inputs in enumerate(self.val_gen):
                loss, loss_emb, loss_ctr_q, loss_ctr_U, acc_emb = 0, 0, 0, 0, 0

                emb_outputs = self.emb_mod.forward(inputs) 
                loss_emb = self.lambda_embedding * emb_outputs['loss_emb']
                acc_emb = emb_outputs['acc_emb']
                if train_emb:
                    loss += loss_emb
        
                if control:
                    inputs.update(emb_outputs)
                    ctr_outputs = self.ctr_mod.forward(inputs)    
                    loss_ctr_U = self.lambda_support * ctr_outputs['loss_ctr_U']
                    loss_ctr_q = self.lambda_query *  ctr_outputs['loss_ctr_q']
                    if train_ctr:
                        loss += loss_ctr_U + loss_ctr_q

                batch_size = len(inputs[list(inputs)[0]])
                val_loss += loss * batch_size
                val_loss_emb += loss_emb * batch_size
                val_loss_ctr_U += loss_ctr_U * batch_size
                val_loss_ctr_q += loss_ctr_q * batch_size
                val_acc_emb += acc_emb * batch_size

            val_loss /= len(self.val_gen.dataset)
            val_loss_emb /= len(self.val_gen.dataset)
            val_loss_ctr_U /= len(self.val_gen.dataset)
            val_loss_ctr_q /= len(self.val_gen.dataset)
            val_acc_emb /= len(self.val_gen.dataset)
            val_loss_total = val_loss_emb + val_loss_ctr_U + val_loss_ctr_q
            print('\nValidation set: Average loss: {:.4f} (Tot: {:.4f}, Emb: {:.4f}, CtrQ: {:.4f}, CtrU: {:.4f}), Embedding Accuracy: {:.4f}\n'.format(
                val_loss, val_loss_total, val_loss_emb, val_loss_ctr_q, val_loss_ctr_U, val_acc_emb))

            if writer is not None:
                writer.add_scalar('loss', val_loss, epoch)
                writer.add_scalar('loss_emb', val_loss_emb, epoch)
                writer.add_scalar('loss_ctr_U', val_loss_ctr_U, epoch)
                writer.add_scalar('loss_ctr_q', val_loss_ctr_q, epoch)
                writer.add_scalar('loss_all', val_loss_total, epoch)
                writer.add_scalar('acc_emb', val_acc_emb, epoch)

    def resume(self, log_dir, epoch, device):
        self.emb_mod.load(log_dir + '/model_emb_' + str(epoch) + '.pt', device)
        self.ctr_mod.load(log_dir + '/model_ctr_' + str(epoch) + '.pt', device)
        self.opt.load_state_dict(torch.load(log_dir + '/model_opt_' + str(epoch) + '.pt', map_location=device))

    def save(self, log_dir, epoch):
        self.emb_mod.save(log_dir + '/model_emb_' + str(epoch) + '.pt')
        self.ctr_mod.save(log_dir + '/model_ctr_' + str(epoch) + '.pt')
        torch.save(self.opt.state_dict(), log_dir + '/model_opt_' + str(epoch) + '.pt')

    def evaluate(self, epoch, writer=None):
        if self.eval_emb is not None:
            self.eval_emb.evaluate(epoch, self.emb_mod, writer=writer)
        if self.eval_sim is not None:
            _ = self.eval_sim.evaluate(epoch, self.emb_mod, self.ctr_mod)
