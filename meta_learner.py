import torch
import torch.nn.functional as F
import numpy as np

class MetaLearner(object):

    def __init__(self, train_gen, val_gen, emb_mod, ctr_mod, opt,
                 lambda_embedding=1.0, lambda_support=0.1, lambda_query=0.1):
        self.emb_mod = emb_mod
        self.ctr_mod = ctr_mod
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.opt = opt

        # loss weights
        self.lambda_embedding = lambda_embedding
        self.lambda_support = lambda_support
        self.lambda_query = lambda_query

    def meta_train(self, epoch, train_emb=True, train_ctr=True, control=True, writer=None, log_interval=5):
        if train_ctr and not control:
            raise RuntimeError('Cannot train the control module without evaluating it.')
        self.emb_mod.eval()
        self.ctr_mod.eval()
        if train_emb:
            self.emb_mod.train()
        if train_ctr:
            self.ctr_mod.train()

        running_loss = 0
        running_size = 0
        for batch_idx, inputs in enumerate(self.train_gen):
            self.opt.zero_grad()
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

            loss.backward()
            self.opt.step()

            batch_size = len(inputs[list(inputs)[0]])
            running_size += batch_size
            loss_total = loss_emb + loss_ctr_U + loss_ctr_q
            if batch_idx % log_interval == 0:
                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t\tTot: {:.6f}\tEmb: {:.6f}\tCtrQ: {:.6f}\tCtrU: {:.6f}'.format( 
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

    def evaluate(self):
        self.eval.evaluate(self.emb_mod, self.ctr_mod)