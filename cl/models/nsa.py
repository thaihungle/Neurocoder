import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.controller import LSTMController


δ = 1e-6
def θ(a, b, dimA=2, dimB=2, normBy=2):
    """Batchwise Cosine distance

    Cosine distance

    Arguments:
        a {Tensor} -- A 3D Tensor (b * m * w)
        b {Tensor} -- A 3D Tensor (b * r * w)

    Keyword Arguments:
        dimA {number} -- exponent value of the norm for `a` (default: {2})
        dimB {number} -- exponent value of the norm for `b` (default: {1})

    Returns:
        Tensor -- Batchwise cosine distance (b * r * m)
    """
    a_norm = torch.norm(a, normBy, dimA, keepdim=True).expand_as(a) + δ
    b_norm = torch.norm(b, normBy, dimB, keepdim=True).expand_as(b) + δ

    x = torch.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (
            torch.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + δ)
    return x


class ProgammedController(nn.Module):
    def __init__(self, program_shape, program_interface_size, pkey_dim=5, num_program=20,
                 bias=False, svd_num_features=8, top_lu=10, has_res_w="n", low_rank_factor=0,
                 kc_mode="lk", rnn_step=10):
        super(ProgammedController, self).__init__()
        self.pkey_dim = pkey_dim
        self.program_size = num_program
        self.program_shape = program_shape
        self.has_bias = bias
        self.rnn_step = rnn_step
        self.top_lu = top_lu
        self.lrf = low_rank_factor
        self.has_res_w = has_res_w
        self.svd_num_features = svd_num_features
        self.kc_mode = kc_mode

        self.PM_U = nn.Parameter(torch.zeros(self.program_size,
                                             self.pkey_dim + program_shape[0],
                                  requires_grad=True))

        self.PM_V = nn.Parameter(torch.zeros(self.program_size,
                                             self.pkey_dim + program_shape[1],
                                             requires_grad=True))
        self.PM_S = nn.Parameter(torch.zeros(self.program_size,self.pkey_dim+1,
                                             requires_grad=True))

        if self.has_res_w == "y":
            if self.lrf<2:
                self.res_weight = nn.Parameter(torch.zeros(program_shape[0],
                                          program_shape[1],
                                          requires_grad=True))
            else:
                self.res_weight_l = nn.Parameter(torch.zeros(program_shape[0],
                                                           program_shape[1]//self.lrf,
                                                           requires_grad=True))
                self.res_weight_r = nn.Parameter(torch.zeros(program_shape[1]//self.lrf,
                                                           program_shape[1],
                                                           requires_grad=True))

        if self.rnn_step == 0:

            self.program_key_u = nn.Linear(program_interface_size,self.svd_num_features*self.pkey_dim)
            self.program_key_v = nn.Linear(program_interface_size, self.svd_num_features*self.pkey_dim)
            self.program_key_s= nn.Linear(program_interface_size, self.svd_num_features*self.pkey_dim)
            self.res_s = nn.Linear(program_interface_size, 1)

        else:

            self.rnn_program_controller = LSTMController(num_inputs=program_interface_size,
                                                         num_outputs=self.svd_num_features*self.pkey_dim*3+1,
                                                         num_layers=1)
            self.component_map = nn.Linear(self.svd_num_features*self.pkey_dim*3+1, program_interface_size)

            if self.top_lu>0:
                self.read_mode = nn.Linear(program_interface_size, 3*num_program)

        if self.kc_mode == "cb":
            self.p2ku = nn.Linear(program_shape[0], self.pkey_dim)
            self.p2kv = nn.Linear(program_shape[1], self.pkey_dim)

        for name, param in self.named_parameters():
            if "PM" not in name:
                param.requires_grad = False

        self.bias = nn.Parameter(torch.zeros(program_shape[1],
                                             requires_grad=True))
        self.record_Ua = []
        self.record_Va = []
        self.record_Sa = []



    def initialize(self):
        nn.init.xavier_uniform_(self.PM_U, gain=1)
        nn.init.xavier_uniform_(self.PM_V, gain=1)
        nn.init.xavier_uniform_(self.PM_S, gain=1)
        if self.has_res_w == "y":
            if self.lrf<2:
                nn.init.xavier_uniform_(self.res_weight, gain=1)
            else:
                nn.init.xavier_uniform_(self.res_weight_l, gain=1)
                nn.init.xavier_uniform_(self.res_weight_r, gain=1)
            nn.init.normal_(self.bias, std=0.01)

        if self.rnn_step == 0:
            nn.init.xavier_uniform_(self.program_key_u.weight, gain=1)
            nn.init.normal_(self.program_key_u.bias, std=0.01)
            nn.init.xavier_uniform_(self.program_key_v.weight, gain=1)
            nn.init.normal_(self.program_key_v.bias, std=0.01)
            nn.init.xavier_uniform_(self.program_key_s.weight, gain=1)
            nn.init.normal_(self.program_key_s.bias, std=0.01)
            nn.init.xavier_uniform_(self.res_s.weight, gain=1)
            nn.init.normal_(self.res_s.bias, std=0.01)
        else:
            self.rnn_program_controller.reset_parameters()

    def init_seq(self):
        self.record_Ua=[]
        self.record_Va=[]
        self.record_Sa=[]

    def updateMK(self, kc_mode=None):
        if self.kc_mode == "lk":
            self.PK_U = self.PM_U[:,:self.pkey_dim]
            self.PK_V = self.PM_V[:, :self.pkey_dim]
            self.PK_S = self.PM_S[:, :self.pkey_dim]
        elif self.kc_mode == "cb":
            self.PK_U = self.p2ku(self.PM_U[:, self.pkey_dim:])
            self.PK_V = self.p2kv(self.PM_V[:, self.pkey_dim:])
            self.PK_S = self.PM_S[:, :self.pkey_dim]

    def get_reg_loss(self):
        if torch.cuda.is_available():
            I = torch.eye(self.program_size).cuda()
        else:
            I = torch.eye(self.program_size)

        ploss1 = torch.norm(torch.matmul(self.PM_U, self.PM_U.t()) - I)
        ploss2 = torch.norm(torch.matmul(self.PM_V, self.PM_V.t()) - I)
        return ploss1 + ploss2

    def read_Us(self, x):
        MK = self.PK_U.repeat(x.shape[0], 1, 1)
        MP = self.PM_U[:,self.pkey_dim:].repeat(x.shape[0], 1, 1)
        ku = self.program_key_u(x).view(x.shape[0],self.svd_num_features,-1)
        dU = θ(MK, ku)
        self.record_Ua.append(dU)
        dU = F.softmax(dU, dim=-1)
        Us = torch.matmul(dU, MP)
        return Us

    def read_Vs(self, x):
        MK = self.PK_V.repeat(x.shape[0], 1, 1)
        MP = self.PM_V[:,self.pkey_dim:].repeat(x.shape[0], 1, 1)
        kv = self.program_key_v(x).view(x.shape[0],self.svd_num_features,-1)
        dV = θ(MK, kv)
        self.record_Va.append(dV)
        dV = F.softmax(dV, dim=-1)
        Vs = torch.matmul(dV, MP)
        return Vs

    def read_Ss(self, x):
        MK = self.PK_S.repeat(x.shape[0], 1, 1)
        MP = self.PM_S[:,self.pkey_dim:].repeat(x.shape[0], 1, 1)
        ks = self.program_key_s(x).view(x.shape[0],self.svd_num_features,-1)
        dS = θ(MK, ks)
        self.record_Sa.append(dS)
        dS = F.softmax(dS, dim=-1)
        Ss = torch.matmul(dS, MP)
        Ss = F.softplus(Ss)
        Ss = torch.cumsum(Ss, dim=1).squeeze(-1)
        Ss = torch.diag_embed(Ss, offset=0, dim1=-2, dim2=-1)
        return Ss

    def read_component(self, MP, MK, k, luw=None, rm=0):
        d = θ(MK, k)
        d = F.softmax(d*10, dim=-1)


        if self.top_lu>0:
            m, _ = torch.max(d, dim=-1)
            d = d*1/m.unsqueeze(2)
            d = d*(1-rm) + luw*rm
        M = torch.matmul(d, MP)
        return M, d

    def readPM_recurrent(self, x):
        MKu = self.PK_U.repeat(x.shape[0], 1, 1)
        MPu = self.PM_U[:, self.pkey_dim:].repeat(x.shape[0], 1, 1)
        MKv = self.PK_V.repeat(x.shape[0], 1, 1)
        MPv = self.PM_V[:, self.pkey_dim:].repeat(x.shape[0], 1, 1)
        MKs = self.PK_S.repeat(x.shape[0], 1, 1)
        MPs = self.PM_S[:, self.pkey_dim:].repeat(x.shape[0], 1, 1)

        U = []
        V = []
        S = []

        dUs = []
        dVs = []
        dSs = []

        state = self.rnn_program_controller.create_new_state(x.shape[0])
        if self.top_lu>0:
            luw_u = torch.zeros(x.shape[0], 1, self.program_size-self.top_lu)
            luw_u2 = torch.ones(x.shape[0], 1, self.top_lu)
            luw_u = torch.cat([luw_u2, luw_u], dim=-1)
            luw_v = torch.zeros(x.shape[0], 1, self.program_size-self.top_lu)
            luw_v2 = torch.ones(x.shape[0], 1, self.top_lu)
            luw_v = torch.cat([luw_v2, luw_v], dim=-1)
            luw_s = torch.zeros(x.shape[0], 1, self.program_size-self.top_lu)
            luw_s2 = torch.ones(x.shape[0], 1, self.top_lu)
            luw_s = torch.cat([luw_s2, luw_s], dim=-1)

            if torch.cuda.is_available():
                luw_u = luw_u.cuda()
                luw_v = luw_v.cuda()
                luw_s = luw_s.cuda()


        for step in range(self.rnn_step):
            interface, state = self.rnn_program_controller(x, state)
            # x = self.component_map(interface)
            key_u, key_v, key_s, rs = interface[:,:self.svd_num_features*self.pkey_dim],\
                                      interface[:,self.svd_num_features*self.pkey_dim:self.svd_num_features*self.pkey_dim*2], \
                                      interface[:, self.svd_num_features*self.pkey_dim*2:self.svd_num_features*self.pkey_dim*3],\
                                      interface[:,self.svd_num_features*self.pkey_dim*3:]

            if self.top_lu==0:
                Ut, dU = self.read_component(MPu, MKu, key_u.view(x.shape[0],self.svd_num_features,-1))
                Vt, dV = self.read_component(MPv, MKv, key_v.view(x.shape[0],self.svd_num_features,-1))
                St, dS = self.read_component(MPs, MKs, key_s.view(x.shape[0],self.svd_num_features,-1))
            else:
                rm = F.sigmoid(self.read_mode(x))
                rm_u = rm[:,:self.program_size].unsqueeze(1)
                rm_v = rm[:, self.program_size:self.program_size*2].unsqueeze(1)
                rm_s = rm[:, self.program_size*2:].unsqueeze(1)
                Ut, dU = self.read_component(MPu, MKu, key_u.view(x.shape[0], self.svd_num_features, -1),
                                             luw_u, rm_u)
                Vt, dV = self.read_component(MPv, MKv, key_v.view(x.shape[0], self.svd_num_features, -1),
                                             luw_v, rm_v)
                St, dS = self.read_component(MPs, MKs, key_s.view(x.shape[0], self.svd_num_features, -1),
                                             luw_s, rm_s)

            U.append(Ut)
            V.append(Vt)
            S.append(St)

            dUs.append(dU)
            dVs.append(dV)
            dSs.append(dS)

            dU = torch.cat(dUs, dim=1)
            dV = torch.cat(dVs, dim=1)
            dS = torch.cat(dSs, dim=1)

            if self.top_lu>0:
                max_useu, _ = torch.max(dU, dim=1)
                upperu, _ = torch.max(max_useu, dim=-1)
                luw_u = 1 -max_useu
                luw_u_sort, _ = luw_u.sort(dim=-1, descending=True)
                th = luw_u_sort[:,self.top_lu].unsqueeze(1)
                luw_u = (luw_u*(luw_u>th).float()).unsqueeze(1)
                max_usev, _ = torch.max(dV, dim=1)
                upperv, _ = torch.max(max_usev, dim=-1)
                luw_v = 1-max_usev
                luw_v_sort, _ = luw_v.sort(dim=-1, descending=True)
                th = luw_v_sort[:, self.top_lu].unsqueeze(1)
                luw_v = (luw_v*(luw_v > th).float()).unsqueeze(1)
                max_uses, _ = torch.max(dS, dim=1)
                uppers, _ = torch.max(max_uses, dim=-1)
                luw_s = 1 - max_uses
                luw_s_sort, _ = luw_s.sort(dim=-1, descending=True)
                th = luw_s_sort[:, self.top_lu].unsqueeze(1)
                luw_s = (luw_s*(luw_s > th).float()).unsqueeze(1)



        U = torch.cat(U, dim=1)
        V = torch.cat(V, dim=1)
        S = torch.cat(S, dim=1)



        self.record_Ua.append(dU)
        self.record_Va.append(dV)
        self.record_Sa.append(dS)


        S = F.softplus(S)
        S = torch.cumsum(S, dim=1).squeeze(-1)
        S = torch.flip(S, dims=[1])
        S = torch.diag_embed(S, offset=0, dim1=-2, dim2=-1)
        W = self.composeSVD(U, V, S)
        rs = F.sigmoid(rs)
        return W, rs, S[:, -1, -1]


    def composeSVD(self, U, V, S):
        US = torch.matmul(U.permute(0, 2, 1), S)
        USV = torch.matmul(US, V)
        return USV

    def forward(self, x, res_weight=None):
        self.updateMK()
        self.init_seq()
        if  self.has_res_w == "y":
            if self.lrf<2:
                res_weight = self.res_weight
            else:
                res_weight = torch.matmul(self.res_weight_l, self.res_weight_r)

        if self.rnn_step == 0:
            U = self.read_Us(x)
            V = self.read_Vs(x)
            S = self.read_Ss(x)
            W = self.composeSVD(U, V, S)
            rs = F.sigmoid(self.res_s(x))
            s = S[:, 0, 0]
        else:
            W, rs, s = self.readPM_recurrent(x)


        if self.has_res_w == "y" or res_weight is not None:
            a = s.unsqueeze(1).unsqueeze(2) * rs.unsqueeze(2)
            W = W + a*res_weight.repeat(x.shape[0], 1, 1)

        y = torch.matmul(x.unsqueeze(1), W).squeeze(1)
        if self.has_bias:
            y = y + self.bias
        return y