import torch
import torch.nn as nn
from models import nsa
from models import nsm

class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, mode='nsa'):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz

        self.mode=mode

        print("FF", self.mode)

        if self.mode=='nsa':
            self.nsa1 =  nsa.ProgammedController(program_shape=[self.in_dim, hidden_dim],
                                        program_interface_size=self.in_dim,
                                        pkey_dim=5, rnn_step=0, has_res_w='n',
                                        num_program=50,
                                        bias=False, svd_num_features=10, top_lu=5,
                                        kc_mode='cb')
            self.nsa2 = nsa.ProgammedController(program_shape=[hidden_dim, hidden_dim],
                                        program_interface_size=hidden_dim,
                                        pkey_dim=5, rnn_step=0, has_res_w='n',
                                        num_program=50,
                                        bias=False, svd_num_features=10, top_lu=5,
                                         kc_mode='cb')

            self.nsa1.initialize()
            self.nsa2.initialize()
            self.linear = nn.Sequential(
                self.nsa1,
                nn.ReLU(inplace=True),
                self.nsa2,
                nn.ReLU(inplace=True),
            )

        elif self.mode == "nsm":
            self.nsm1 = nsm.ProgammedController(program_shape=[self.in_dim, hidden_dim],
                                        program_interface_size=self.in_dim,
                                    pkey_dim=5,
                                    num_program=5,
                                    bias=False, svd_num_features=15,
                                    att_mode="kv", program_read_mode="linear")
            self.nsm2 = nsm.ProgammedController(program_shape=[hidden_dim, hidden_dim],
                                                program_interface_size=hidden_dim,
                                                pkey_dim=5,
                                                num_program=5,
                                                bias=False, svd_num_features=15,
                                                att_mode="kv", program_read_mode="linear")
            self.nsm1.initialize()
            self.nsm2.initialize()
            self.linear = nn.Sequential(
                self.nsm1,
                nn.ReLU(inplace=True),
                self.nsm2,
                nn.ReLU(inplace=True),
            )

        else:
            self.linear = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.ReLU(inplace=True),
            )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task


    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def get_last(self, nin, nout):
        if self.mode=="mlp":
            return nn.Linear(nin, nout)
        elif self.mode=="nsa":
            self.nsa_last = nsa.ProgammedController(program_shape=[nin, nout],
                                        program_interface_size=nin,
                                        pkey_dim=5, rnn_step=0, has_res_w='n',
                                        num_program=50,
                                        bias=False, svd_num_features=10, top_lu=5,
                                         kc_mode='cb')
            self.nsa_last.initialize()
            return self.nsa_last
        elif self.mode=="nsm":
            nsa_last = nsm.ProgammedController(program_shape=[nin, nout],
                                        program_interface_size=nin,
                                        pkey_dim=5,  num_program=5,
                                        bias=False, svd_num_features=15,
                                        att_mode="kv", program_read_mode="linear")
            nsa_last.initialize()
            return nsa_last


    def get_ploss(self):
        if self.mode=="nsa":
            loss = self.nsa1.get_reg_loss() + self.nsa2.get_reg_loss()+self.nsa_last.get_reg_loss()
            return loss
        else:
            return torch.tensor(0.0)
        # print(self.last)

def MLP100():
    return MLP(hidden_dim=100)


def MLP400(mode="mlp"):
    return MLP(hidden_dim=400, mode=mode)


def MLP1000():
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)