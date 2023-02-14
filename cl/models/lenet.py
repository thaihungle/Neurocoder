import torch
import torch.nn as nn
from models import nsa
from models import  nsm
import torchvision.models as models

class LeNet(nn.Module):

    def __init__(self, out_dim=10, in_channel=3, img_sz=32, mode='mlp'):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.mode = mode

        # self.n_feat = 50 * feat_map_sz * feat_map_sz
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channel, 20, 5, padding=2),
        #     # nn.BatchNorm2d(20),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(20, 50, 5, padding=2),
        #     # nn.BatchNorm2d(50),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2)
        # )

        self.n_feat = 1000
        self.conv = models.resnet18(pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = True

        if self.mode=='nsa':
            self.nsa1 =  nsa.ProgammedController(program_shape=[self.n_feat, 500],
                                        program_interface_size=self.n_feat,
                                        pkey_dim=5, rnn_step=0, has_res_w='n',
                                        num_program=50,
                                        bias=False, svd_num_features=10, top_lu=5,
                                        att_mode="kv", program_read_mode="linear", kc_mode='cb')

            self.nsa1.initialize()
            self.nsa2 = nsa.ProgammedController(program_shape=[500, 500],
                                                program_interface_size=500,
                                                pkey_dim=5, rnn_step=0, has_res_w='n',
                                                num_program=50,
                                                bias=False, svd_num_features=10, top_lu=5,
                                                att_mode="kv", program_read_mode="linear", kc_mode='cb')

            self.nsa2.initialize()
            self.nsa3 = nsa.ProgammedController(program_shape=[500, 500],
                                               program_interface_size=500,
                                               pkey_dim=5, rnn_step=0, has_res_w='n',
                                               num_program=50,
                                               bias=False, svd_num_features=10, top_lu=5,
                                               att_mode="kv", program_read_mode="linear", kc_mode='cb')

            self.nsa3.initialize()
            self.linear = nn.Sequential(
                self.nsa1,
                # nn.BatchNorm1d(500),
                nn.ReLU(inplace=True),
                self.nsa2,
                # nn.BatchNorm1d(500),
                nn.ReLU(inplace=True),
                self.nsa3,
                # nn.BatchNorm1d(500),
                nn.ReLU(inplace=True),
            )

        elif self.mode == "nsm":
            self.nsm1 = nsm.ProgammedController(program_shape=[self.n_feat, 500],
                                        program_interface_size=self.n_feat,
                                    pkey_dim=5,
                                    num_program=5,
                                    bias=False, svd_num_features=15,
                                    att_mode="kv", program_read_mode="linear")

            self.nsm1.initialize()
            self.linear = nn.Sequential(
                self.nsm1,
                # nn.BatchNorm1d(500),
                nn.ReLU(inplace=True),
            )

        else:
            self.linear = nn.Sequential(
                nn.Linear(self.n_feat, 500, bias=False),
                # nn.BatchNorm1d(500),
                nn.ReLU(inplace=True),
                nn.Linear(500, 500, bias=False),
                # nn.BatchNorm1d(500),
                nn.ReLU(inplace=True),
                nn.Linear(500, 500, bias=False),
                # nn.BatchNorm1d(500),
                nn.ReLU(inplace=True),
            )
        self.last = nn.Linear(500, out_dim)  # Subject to be replaced dependent on task

    def get_last(self, nin, nout):
        # return nn.Linear(nin, nout)

        if self.mode=="mlp":
            return nn.Linear(nin, nout)
        elif self.mode=="nsa":
            nsa_last = nsa.ProgammedController(program_shape=[nin, nout],
                                        program_interface_size=nin,
                                        pkey_dim=10, rnn_step=0, has_res_w='n',
                                        num_program=50,
                                        bias=False, svd_num_features=10, top_lu=5,
                                        att_mode="kv", program_read_mode="linear", kc_mode='cb')
            nsa_last.initialize()
            self.nsa_last = nsa_last
            return nsa_last
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
            loss = self.nsa1.get_reg_loss()+self.nsa2.get_reg_loss()+self.nsa3.get_reg_loss()+self.nsa_last.get_reg_loss()
            return 10*loss
        else:
            return torch.tensor(0.0)
        # print(self.last)

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def LeNetC(out_dim=10):  # LeNet with color input
    return LeNet(out_dim=out_dim, in_channel=3, img_sz=32)