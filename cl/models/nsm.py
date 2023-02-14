import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, program_shape, program_interface_size, pkey_dim=10, num_program=2,
                 bias=False, svd_num_features=8, program_read_mode="linear", att_mode="kv", kc_mode="lk"):
        super(ProgammedController, self).__init__()
        self.pkey_dim = pkey_dim
        self.program_size = num_program
        self.program_shape = program_shape
        self.has_bias = bias
        self.svd_num_features = svd_num_features
        self.program_read_mode = program_read_mode
        self.att_mode = att_mode
        self.kc_mode = kc_mode

        a=0
        if bias:
            a =1

        self.mprogram_weights = nn.Parameter(torch.zeros(self.program_size,
                                                         self.pkey_dim +
                                                         (program_shape[0]+a) * program_shape[1],
                                                         requires_grad=True))

        # if att_mode == "kv":
        self.program_key = nn.Linear(program_interface_size, self.pkey_dim)
        self.program_strength = nn.Linear(program_interface_size, 1)
        # elif att_mode == "da":
        self.address_net = nn.Linear(program_interface_size, self.program_size)
        # elif  att_mode == "dasvd":
        self.address_svd_net = nn.Linear(program_interface_size, self.program_size*svd_num_features)
        self.address_svd_net2 = nn.Linear(program_interface_size, self.program_size)

        self.program_descriptor = nn.Linear(min(program_shape[0]+a,program_shape[1])*self.svd_num_features//2*3, self.pkey_dim)
        self.program_descriptor2 = nn.Linear(program_shape[1]*2+1,
                                             self.pkey_dim)
        # self.program_descriptor2.weight.require_grad=False
        # self.program_descriptor2.bias.require_grad=False

        self.program_key2 = nn.Linear(program_interface_size, self.pkey_dim*(self.svd_num_features+1))
        # self.program_key2.weight.require_grad = False
        # self.program_key2.bias.require_grad = False
        self.code_len_net = nn.Linear(program_interface_size, program_shape[1])

        self.pad_size = program_shape[0]+a-self.svd_num_features
        # stdv = 1. / math.sqrt(self.mprogram_weights.size(1))
        # self.mprogram_weights.data.uniform_(-stdv, stdv)
        self.relu = nn.ReLU()
        self.MK = None
        self.Us = None
        self.Ss = None
        self.Vs = None
        for name, param in self.named_parameters():
            if "mprogram_weights" not in name:
                param.requires_grad = True

        # self.updateMK()

    def initialize(self):
        nn.init.xavier_uniform_(self.mprogram_weights, gain=1.4)
        nn.init.xavier_uniform_(self.program_key.weight, gain=1.4)
        nn.init.normal_(self.program_key.bias, std=0.01)
        nn.init.xavier_uniform_(self.program_strength.weight, gain=1.4)
        nn.init.normal_(self.program_strength.bias, std=0.01)
        nn.init.xavier_uniform_(self.address_net.weight, gain=1.4)
        nn.init.normal_(self.address_net.bias, std=0.01)
        nn.init.xavier_uniform_(self.address_svd_net.weight, gain=1.4)
        nn.init.normal_(self.address_svd_net.bias, std=0.01)
        nn.init.xavier_uniform_(self.address_svd_net2.weight, gain=1.4)
        nn.init.normal_(self.address_svd_net2.bias, std=0.01)
        nn.init.xavier_uniform_(self.program_descriptor.weight, gain=1.4)
        nn.init.normal_(self.program_descriptor.bias, std=0.01)
        nn.init.xavier_uniform_(self.program_key2.weight, gain=1.4)
        nn.init.normal_(self.program_key2.bias, std=0.01)
        nn.init.xavier_uniform_(self.code_len_net.weight, gain=1.4)
        nn.init.normal_(self.code_len_net.bias, std=0.01)

    def get_mprogram_weight(self, p):
        return self.mprogram_weights[p,self.pkey_dim:self.pkey_dim+self.program_shape[0]*self.program_shape[1]]

    def attend_program(self, input):

        if self.att_mode == "kv":
            keys = F.tanh(self.program_key(input))
            strength = F.softplus(self.program_strength(input))
            K = keys.unsqueeze(1)[:, :, :self.pkey_dim]

            d = θ(self.MK.repeat(keys.shape[0], 1, 1), K)
            content_weights = F.softmax(d * strength.unsqueeze(2), dim=-1)
            return content_weights, keys, strength
        elif self.att_mode == "da":
            a = F.softmax(self.address_net(input), dim=-1)
            return a, None, None
        elif self.att_mode == "dasvd":
            a = self.address_svd_net(input).view(input.shape[0], self.svd_num_features, self.program_size)
            a2 = self.address_svd_net2(input).view(input.shape[0], 1, self.program_size)

            a = F.softmax(a, dim=-1)
            a2 = F.softmax(a2, dim=-1)

            pad = torch.ones(input.shape[0], self.pad_size, self.program_size)
            if torch.cuda.is_available():
                pad = pad.cuda()
            a = torch.cat([a, pad*a2], dim=1)
            return a, None, None
        elif self.att_mode == "kvsvd":
            K = F.tanh(self.program_key2(input)).view(input.shape[0], self.svd_num_features+1, self.pkey_dim)

            MK = self.MK.permute(1,0,2)

            MK = MK.repeat(input.shape[0], 1, 1, 1)
            d = θ(MK.view(-1, self.program_size, self.pkey_dim), K.view(-1,1,self.pkey_dim))
            content_weights = F.softmax(d , dim=-1).view(input.shape[0], self.svd_num_features+1,self.program_size)
            pad = torch.ones(input.shape[0], self.pad_size, self.program_size)
            if torch.cuda.is_available():
                pad = pad.cuda()
            pad = pad*content_weights[:,-1,:].unsqueeze(1)
            a = torch.cat([content_weights[:,:self.svd_num_features,:], pad], dim=1)
            return a, K[:,0,:], None

    def updateMK(self, kc_mode="lk"):
        if "svd" in kc_mode:
            try:
               MP = self.mprogram_weights[:, self.pkey_dim:]
               A = MP.view(MP.shape[0], -1, self.program_shape[1])
               MK = []
               Us = []
               Ss = []
               Vs = []
               for i in range(self.program_size):
                   U,S,V = torch.svd(A[i])
                   if kc_mode=="svds":
                       MK.append(S[:self.pkey_dim])
                   elif kc_mode == "svda":
                       pfeature = torch.cat([U[:self.svd_num_features+1, :],
                                             V[:self.svd_num_features+1, :],
                                             S[:self.svd_num_features+1].unsqueeze(1)], dim=1)
                       MK.append(self.program_descriptor2(pfeature))
                   else:
                       pfeature = torch.cat([U[:self.svd_num_features//2,:].contiguous().view(-1),
                                             V[:self.svd_num_features //2, :].contiguous().view(-1),
                                             S[:self.svd_num_features//2]])
                       MK.append(self.program_descriptor(pfeature))
                   if self.program_read_mode!="linear":
                       Us.append(U.contiguous().view(-1))
                       Ss.append(S)
                       Vs.append(V.contiguous().view(-1))
               self.MK = F.tanh(torch.stack(MK, dim=0))
               if self.program_read_mode != "linear":
                   self.Us = torch.stack(Us, dim=0)
                   self.Ss = torch.stack(Ss, dim=0)
                   self.Vs = torch.stack(Vs, dim=0)

            except Exception as e:
               print(f"svd err {e}")
        elif kc_mode == "lk":
            self.MK = F.tanh(self.mprogram_weights[:,:self.pkey_dim])
            if self.program_read_mode!="linear":
                MP = self.mprogram_weights[:, self.pkey_dim:]
                A = MP.view(MP.shape[0], -1, self.program_shape[1])
                MK = []
                Us = []
                Ss = []
                Vs = []
                for i in range(self.program_size):
                    U, S, V = torch.svd(A[i])
                    Us.append(U.contiguous().view(-1))
                    Ss.append(S)
                    Vs.append(V.contiguous().view(-1))
                if self.program_read_mode != "linear":
                    self.Us = torch.stack(Us, dim=0)
                    self.Ss = torch.stack(Ss, dim=0)
                    self.Vs = torch.stack(Vs, dim=0)


    def linear_read(self, MP, weights):
        return  torch.matmul(weights, MP)

    def linear_svd_read(self, weights):
        U = torch.matmul(weights, self.Us.repeat(weights.shape[0], 1, 1)).view(weights.shape[0],-1, self.program_shape[1])
        # U = torch.sum(self.Us, dim=0).repeat(weights.shape[0], 1, 1).view(weights.shape[0],-1, self.program_shape[1])
        # V = torch.sum(self.Vs, dim=0).repeat(weights.shape[0], 1, 1).view(weights.shape[0],-1, self.program_shape[1])
        V = torch.matmul(weights, self.Vs.repeat(weights.shape[0], 1, 1)).view(weights.shape[0],-1, self.program_shape[1])

        S = torch.matmul(weights, self.Ss.repeat(weights.shape[0], 1, 1))
        S = torch.torch.diag_embed(S, offset=0, dim1=-2, dim2=-1).squeeze(1)
        US = torch.matmul(U, S)

        USV = torch.matmul(US, V.permute(0,2,1))

        return USV

    def linear_svd_read1(self, weights):
        S = torch.matmul(weights, self.Ss.repeat(weights.shape[0], 1, 1))
        S = torch.diag_embed(S, offset=0, dim1=-2, dim2=-1).squeeze(1)

        U = torch.sum(self.Us, dim=0).repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])
        V = torch.sum(self.Vs, dim=0).repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])
        US = torch.matmul(U, S)
        USV = torch.matmul(US, V.permute(0, 2, 1))



        return USV

    def linear_svd_read2(self, weights):
        S = torch.matmul(weights, self.Ss.repeat(weights.shape[0], 1, 1))
        S = torch.torch.diag_embed(S, offset=0, dim1=-2, dim2=-1).squeeze(1)

        U = self.Us[0].repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])
        V = self.Vs[0].repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])
        US = torch.matmul(U, S)
        USV = torch.matmul(US, V.permute(0, 2, 1))*weights[:,:,0].unsqueeze(2)

        for i in range(self.program_size-1):
            U = self.Us[i+1].repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1,self.program_shape[1])
            V = self.Vs[i+1].repeat(weights.shape[0], 1, 1).view(weights.shape[0],-1, self.program_shape[1])
            US = torch.matmul(U, S)
            USV += torch.matmul(US, V.permute(0, 2, 1))*weights[:,:,i+1].unsqueeze(2)


        return USV

    def linear_svd_read3(self, weights):
        S = torch.matmul(weights, self.Ss.repeat(weights.shape[0], 1, 1))
        S = torch.diag_embed(S, offset=0, dim1=-2, dim2=-1).squeeze(1)

        U = self.Us[0].repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])
        # V = self.Vs[0].repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])
        V = torch.matmul(weights, self.Vs.repeat(weights.shape[0], 1, 1)).view(weights.shape[0],-1, self.program_shape[1])

        US = torch.matmul(U, S)
        USV = torch.matmul(US, V.permute(0, 2, 1))

        for i in range(self.program_size-1):
            U = self.Us[i+1].repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1,self.program_shape[1])
            V = self.Vs[i+1].repeat(weights.shape[0], 1, 1).view(weights.shape[0],-1, self.program_shape[1])
            US = torch.matmul(U, S)
            USV += torch.matmul(US, V.permute(0, 2, 1))


        return USV


    def linear_svd_read_da(self, weights, cg):
        S = self.Ss.repeat(weights.shape[0], 1, 1)#*cg.unsqueeze(1)
        S = torch.diag_embed(S, offset=0, dim1=-2, dim2=-1)

        U = (self.Us[0]).repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])*weights[:,:, 0].unsqueeze(2)
        V = self.Vs[0].repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1, self.program_shape[1])
        US = torch.matmul(U, S[:,0,:,:])
        USV = torch.matmul(US, V.permute(0, 2, 1))

        for i in range(self.program_size-1):
            U = (self.Us[i+1]).repeat(weights.shape[0], 1, 1).view(weights.shape[0], -1,self.program_shape[1])*weights[:,:, i+1].unsqueeze(2)
            V = self.Vs[i+1].repeat(weights.shape[0], 1, 1).view(weights.shape[0],-1, self.program_shape[1])
            US = torch.matmul(U, S[:,i+1,:,:])
            USV += torch.matmul(US, V.permute(0, 2, 1))

        return USV

    def read_program(self, input):
        key_size = self.pkey_dim


        content_weights, keys, strengths = self.attend_program(input)

        # print(input.shape)
        # print(memory.shape)
        # print(keys.shape)
        # MK = F.tanh(memory.repeat(keys.shape[0], 1, 1))[:, :, :key_size]
        # MK = self.getMK().repeat(keys.shape[0], 1, 1)
        if self.has_bias:
            biases = self.mprogram_weights.repeat(input.shape[0], 1, 1)[:, :,-self.program_shape[1]:]

        MP = self.mprogram_weights.repeat(input.shape[0], 1, 1)[:, :, key_size:key_size+self.program_shape[0]*self.program_shape[1]]


        # print(MP.shape)
        # print(content_weights.shape)
        if self.program_read_mode == "linear":
            working_weight = self.linear_read(MP, content_weights)
        if self.program_read_mode == "svd":
            working_weight = self.linear_svd_read(content_weights)
        elif self.program_read_mode == "svd1":
            working_weight = self.linear_svd_read1(content_weights)
        elif self.program_read_mode == "svd2":
            working_weight = self.linear_svd_read2(content_weights)
        elif self.program_read_mode == "svd3":
            working_weight = self.linear_svd_read3(content_weights)
        elif self.program_read_mode == "svdda":
            cg = F.sigmoid(self.code_len_net(input))
            working_weight = self.linear_svd_read_da(content_weights, cg)
            content_weights = content_weights[:,0,:]
        if len(content_weights.shape)==2:
            content_weights = content_weights.unsqueeze(1)
        # instruction = content_weights.view(content_weights.shape[0],self.program_size)[:,0].unsqueeze(1) * MP[:,0,:]
        # for i in range(self.program_size-1):
        #     instruction*= content_weights.view(content_weights.shape[0],self.program_size)[:,i+1].unsqueeze(1)*MP[:,i+1,:]

        o = (torch.matmul(input.unsqueeze(1), working_weight.view(input.shape[0], self.program_shape[0], self.program_shape[1]))).squeeze(1)
        if self.has_bias:
            bias = torch.matmul(content_weights, biases).squeeze(1)
            o = o+bias
        program_scales = []
        for p in range(self.program_size):
            s = torch.mean(torch.exp(-torch.abs(working_weight.view(input.shape[0],-1)-MP[:,p]))
                           , dim=0)
            program_scales.append(s)
        return o

    def forward(self, x):
        self.updateMK()
        return self.read_program(x)
