import torch
import torch.nn.functional as F
import numpy as np
from network.utils import plot_grad_flow, register_hooks

class MetaLearner(object):
    def __init__(
        self,
        train_gen,
        val_gen,
        emb_mod,
        opt_emb,
        ctr_mod=None,
        opt_ctr=None, 
        eval_sim=None,
        eval_emb=None,
        lambda_embedding=1.0,
        lambda_support=0.1,
        lambda_query=0.1,
        train_emb=True,
        train_ctr=True
    ):
        # embedding
        self.emb_mod = emb_mod
        self.opt_emb = opt_emb
        self.train_emb = train_emb

        # control
        self.ctr_mod = ctr_mod
        self.opt_ctr = opt_ctr
        self.train_ctr = train_ctr

        # data loaders
        self.train_gen = train_gen
        self.val_gen = val_gen

        # eval
        self.eval_sim = eval_sim
        self.eval_emb = eval_emb

        # loss weights
        self.lambda_embedding = lambda_embedding
        self.lambda_support = lambda_support
        self.lambda_query = lambda_query

    def meta_train(self, epoch, log_interval=10, writer=None, log_histograms=False):
        # freeze/unfreeze embnet
        self.emb_mod.train(self.train_emb) # TODO
        for param in self.emb_mod.parameters():
            param.requires_grad = self.train_emb
        # freeze/unfreeze ctrnet
        if self.ctr_mod is not None:
            self.ctr_mod.train(self.train_ctr)
            for param in self.ctr_mod.parameters():
                param.requires_grad = self.train_ctr

        running_loss = 0
        running_size = 0
        for batch_idx, inputs in enumerate(self.train_gen):
            loss, loss_emb, loss_ctr_q, loss_ctr_U, acc_emb = 0, 0, 0, 0, 0

            # torch.autograd.set_detect_anomaly(True)

            emb_outputs = self.emb_mod.forward(inputs) 
            loss_emb = self.lambda_embedding * emb_outputs['loss_emb']
            acc_emb = emb_outputs['acc_emb']
            if self.train_emb:
                loss += loss_emb

            if self.ctr_mod is not None:
                inputs.update(emb_outputs)
                ctr_outputs = self.ctr_mod.forward(inputs)    
                loss_ctr_U = self.lambda_support * ctr_outputs['loss_ctr_U']
                loss_ctr_q = self.lambda_query *  ctr_outputs['loss_ctr_q'] 
                if self.train_ctr:
                    loss += loss_ctr_U + loss_ctr_q

            self.opt_emb.zero_grad()
            if self.ctr_mod is not None:
                self.opt_ctr.zero_grad()
            # loss.register_hook(lambda grad: print(grad))
            # register_hooks(loss)
            loss.backward()     
            # plot_grad_flow(self.ctr_mod.net.named_parameters(), 'ctr')
            # plot_grad_flow(self.emb_mod.net.named_parameters(), 'emb')

            if log_histograms and batch_idx == 0:
                if writer is not None:
                    if self.train_emb:
                        for name, param in self.emb_mod.net.named_parameters():
                            writer.add_histogram('emb/' + name, param, epoch)
                            writer.add_histogram('emb/' + name + '/gradient', param.grad, epoch)
                    if self.ctr_mod is not None and self.train_ctr:
                        for name, param in self.ctr_mod.net.named_parameters():
                            writer.add_histogram('ctr/' + name, param, epoch)
                            writer.add_histogram('ctr/' + name + '/gradient', param.grad, epoch)


            # clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), val)
            # for param in model.parameters():
            #     param.register_hook(lambda grad: grad.clamp_(-val, val))
            self.opt_emb.step()
            if self.ctr_mod is not None:
                self.opt_ctr.step()

            # torch.autograd.set_detect_anomaly(False)

            batch_size = len(inputs[list(inputs)[0]])
            running_size += batch_size
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} \t[{}/{} \t({:.0f}%)]\tLoss: {:.6f}\tEmb: {:.6f}\tCtrQ: {:.6f}\tCtrU: {:.6f}\tEmbAcc: {:.6f}'.format( 
                    epoch, running_size, len(self.train_gen.dataset), 100. * (batch_idx + 1) / len(self.train_gen),
                    loss, loss_emb, loss_ctr_q, loss_ctr_U, acc_emb))
                if writer is not None:
                    writer.add_scalar('loss', loss, epoch)
                    writer.add_scalar('loss_emb', loss_emb, epoch)
                    writer.add_scalar('loss_ctr_U', loss_ctr_U, epoch)
                    writer.add_scalar('loss_ctr_q', loss_ctr_q, epoch)
                    writer.add_scalar('acc_emb', acc_emb, epoch)

    def meta_valid(self, epoch, writer=None):
        self.emb_mod.eval()
        if self.ctr_mod is not None:
            self.ctr_mod.eval()
        with torch.no_grad():
            
            val_loss, val_loss_emb, val_loss_ctr_q, val_loss_ctr_U, val_acc_emb = 0, 0, 0, 0, 0
            for batch_idx, inputs in enumerate(self.val_gen):
                loss, loss_emb, loss_ctr_q, loss_ctr_U, acc_emb = 0, 0, 0, 0, 0

                emb_outputs = self.emb_mod.forward(inputs) 
                loss_emb = self.lambda_embedding * emb_outputs['loss_emb']
                acc_emb = emb_outputs['acc_emb']
                if self.train_emb:
                    loss += loss_emb
        
                if self.ctr_mod is not None:
                    inputs.update(emb_outputs)
                    ctr_outputs = self.ctr_mod.forward(inputs)    
                    loss_ctr_U = self.lambda_support * ctr_outputs['loss_ctr_U']
                    loss_ctr_q = self.lambda_query *  ctr_outputs['loss_ctr_q']
                    if self.train_ctr:
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
            print('\nValidation set: Average loss: {:.4f} Emb: {:.4f}, CtrQ: {:.4f}, CtrU: {:.4f}), Emb Accuracy: {:.4f}\n'.format(
                val_loss, val_loss_emb, val_loss_ctr_q, val_loss_ctr_U, val_acc_emb))

            if writer is not None:
                writer.add_scalar('loss', val_loss, epoch)
                writer.add_scalar('loss_emb', val_loss_emb, epoch)
                writer.add_scalar('loss_ctr_U', val_loss_ctr_U, epoch)
                writer.add_scalar('loss_ctr_q', val_loss_ctr_q, epoch)
                writer.add_scalar('acc_emb', val_acc_emb, epoch)

    def resume(self, log_dir, epoch, device):
        self.emb_mod.load(log_dir + '/model_emb_' + str(epoch) + '.pt', device)
        self.opt_emb.load_state_dict(torch.load(log_dir + '/model_opt_emb_' + str(epoch) + '.pt', map_location=device))
        if self.ctr_mod is not None:
            self.ctr_mod.load(log_dir + '/model_ctr_' + str(epoch) + '.pt', device)
            self.opt_ctr.load_state_dict(torch.load(log_dir + '/model_opt_ctr_' + str(epoch) + '.pt', map_location=device))

    def save(self, log_dir, epoch):
        self.emb_mod.save(log_dir + '/model_emb_' + str(epoch) + '.pt')
        torch.save(self.opt_emb.state_dict(), log_dir + '/model_opt_emb_' + str(epoch) + '.pt')
        if self.ctr_mod is not None:
            self.ctr_mod.save(log_dir + '/model_ctr_' + str(epoch) + '.pt')       
            torch.save(self.opt_ctr.state_dict(), log_dir + '/model_opt_ctr_' + str(epoch) + '.pt')

    def evaluate(self, epoch, writer=None):
        if self.ctr_mod is not None and self.eval_sim is not None:
            acc = self.eval_sim.evaluate(epoch, self.emb_mod, self.ctr_mod)
            if writer is not None:
                writer.add_scalar('eval_acc', acc, epoch)
        if writer is not None and self.eval_emb is not None:
            self.eval_emb.evaluate(epoch, self.emb_mod, writer=writer)