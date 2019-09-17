from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return my(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, inChannels,outChannels,reduction=16):
        super(CALayer, self).__init__()
        self.conv1 =nn.Conv2d(inChannels,outChannels,kernel_size=1,padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(outChannels, outChannels//reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(outChannels//reduction, outChannels, 1, padding=0, bias=True),
                nn.Sigmoid()
                )
    def forward(self,x):
        out = self.conv1(x)
        y = self.avg_pool(out)
        y = self.conv_du(y)
        
        return y*out


class BasicModule(nn.Module):
    def __init__(self,in_channel):
        super(BasicModule, self).__init__()

        self.conv1 =nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1)
        self.act = nn.ReLU(True)
        self.conv2 =nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(in_channel)
    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        out = torch.add(x,res)
        return out

class BasicUnit(nn.Module):
    def __init__(self, n_feats, n_resblocks):
        super(BasicUnit, self).__init__()
        body = []
        for i in range(n_resblocks):
            body.append(BasicModule(n_feats))
        self.body = nn.Sequential(*body)
   
    def forward(self, x):
        res = self.body(x)
        out = torch.add(x,res)
        return out
        
class my(nn.Module):
    def __init__(self, args,conv=common.default_conv):
        super(my,self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        reduction = args.reduction 
        scale = args.scale[0]
        growthrate = args.growthrate
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)  
        
        # define head module
        self.head = conv(args.n_colors, n_feats, kernel_size=3)

        # define body module                 

        self.unit1 = BasicUnit(n_feats,n_resblocks)
        
        self.unit2 = BasicUnit(n_feats,n_resblocks)
        
        self.node1 = CALayer(n_feats*2,n_feats,reduction)
        
        self.unit3 = BasicUnit(n_feats, n_resblocks)
        
        self.unit4 = BasicUnit(n_feats, n_resblocks)
        
        self.node2 = CALayer(n_feats*2,n_feats,reduction)
        
        self.unit5 = BasicUnit(n_feats, n_resblocks)
        
        self.unit6 = BasicUnit(n_feats, n_resblocks)
        
        self.node3 = CALayer(n_feats*4,n_feats*2,reduction)
       
        self.unit7 = BasicUnit(n_feats*2, n_resblocks)
        
        self.unit8 = BasicUnit(n_feats*2, n_resblocks)
        
        self.node4 = CALayer(n_feats*4,n_feats*2,reduction)
        
        self.unit9 = BasicUnit(n_feats*2, n_resblocks)
        
        self.unit10 = BasicUnit(n_feats*2, n_resblocks)
        
        self.node5 = CALayer(n_feats*2*3,n_feats*3,reduction)
        
        self.unit11 = BasicUnit(n_feats*3, n_resblocks)
        
        self.unit12 = BasicUnit(n_feats*3, n_resblocks)
        
        self.node6 = CALayer(n_feats*6,n_feats*3,reduction)
        
        self.unit13 = BasicUnit(n_feats*3, n_resblocks)
        
        self.unit14 = BasicUnit(n_feats*3, n_resblocks)
        
        self.node7 = CALayer(n_feats*15,480, reduction)
        
        # define tail module
        modules_tail_1 = [
            conv(480, n_feats, kernel_size=3),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size=3)]
            
        modules_tail_2 = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size=3)] 
            
        self.tail_1 =nn.Sequential(*modules_tail_1) 
        self.tail_2 =nn.Sequential(*modules_tail_2)  
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self,x):
        x = self.sub_mean(x)
        first_out = self.head(x)
        out1 = self.unit1(first_out)
        out2 = self.unit2(out1)
        n_in1 = torch.cat([out1,out2],1)
        n_out1 = self.node1(n_in1)
    
        out3 = self.unit3(n_out1)
        out4 = self.unit4(out3)
        n_in2 = torch.cat([out3,out4],1)
        n_out2 = self.node2(n_in2)
        
        out5 = self.unit5(n_out2)
        out6 = self.unit6(out5)
        n_in3 = torch.cat([out5,out6,n_out1,n_out2],1)
        n_out3 = self.node3(n_in3)
        
        out7 = self.unit7(n_out3)
        out8 = self.unit8(out7)
        n_in4 = torch.cat([out7,out8],1)
        n_out4 = self.node4(n_in4)
        
        out9 = self.unit9(n_out4)
        out10 = self.unit10(out9)
        n_in5 = torch.cat([n_out4,out9,out10],1)
        n_out5 = self.node5(n_in5)
        
        out11 = self.unit11(n_out5)
        out12 = self.unit12(out11)
        n_in6 = torch.cat([out11,out12],1)
        n_out6 = self.node6(n_in6)
        
        out13 = self.unit13(n_out6)
        out14 = self.unit14(out13)
        n_in7 = torch.cat([out13,out14,n_out6,n_out5,n_out3,n_out1],1)
        n_out7 = self.node7(n_in7)
        
        
        res = self.tail_1(n_out7)
        out = self.tail_2(first_out)
        HR = torch.add(out,res) 
        HR = self.add_mean(HR) 
                             
        return HR
        
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))