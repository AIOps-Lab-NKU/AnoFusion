import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.F = F.softmax
        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))
        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        h = self.W(inputs)
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(0)
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))
        attention = self.F(outputs, dim=2)
        return torch.bmm(attention, h) + self.b


class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()
        self.attention_module = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)
        self.act = nn.LeakyReLU()

    def forward(self, inputs, graph):
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)
        outputs = self.act(outputs)
        outputs = self.out_att(outputs, graph)
        return self.act(outputs)


class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, data):
        graph = data["graph"][0].to(device)
        flow = data["flow_x"]
        flow = flow.to(device)

        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1)
        prediction = self.subnet(flow, graph)
        return prediction