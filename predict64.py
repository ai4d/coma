#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chang Shu
"""


## create the model
## load the last checkpoint 
## evalaute the model, and get the outptut
## save the output and plot output and input. 
##   ------------------------------------------------------------------------
print("test")
##****  create the model; copied from main.py
import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh

from models import AE
from datasets import MeshData
from utils import utils, writer, train_eval, DataLoader, mesh_sampling

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16, 16, 16, 32],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--K', type=int, default=6)

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=8e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset
template_fp = osp.join('template', 'template.obj')
meshdata = MeshData(args.data_fp,
                    template_fp,
                    split=args.split,
                    test_exp=args.test_exp)
train_loader = DataLoader(meshdata.train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size)

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [4, 4, 4, 4]
    _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

model = AE(args.in_channels,
           args.out_channels,
           args.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args.K).to(device)
print(model)

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=0.9)
else:
    raise RuntimeError('Use optimizers of SGD or Adam')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)

## -------------- test code start here --------------------
## Load data from ply file and make it ready
import openmesh as om 
mesh = om.read_trimesh('data/CoMA/raw/FaceTalk_170725_00137_TA/mouth_up/mouth_up.000001.ply')
face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
x = torch.tensor(mesh.points().astype('float32'))
type(x)
x.shape # torch.Size([5023, 3])
x_reshaped = (x.reshape(1, 5023,3)-meshdata.mean)/meshdata.std

tensor_x02 = torch.tensor(x_reshaped, dtype=torch.double, device='cuda:0')
tensor_x02=tensor_x02.float() ## model(tensor_x02.float()) to predict the output 
## write the input to ply file 
meshdata.save_mesh('x1.ply',tensor_x02.reshape((tensor_x02.size()[1], 3)).cpu())

for i, data in enumerate(test_loader):
            x_test = data.x.to(device)
            if i==3:
                break 
x_test2=x_test[1,:,:].reshape(1, 5023,3).float()             
meshdata.save_mesh('x2.ply',x_test2.reshape((tensor_x02.size()[1], 3)).cpu())
            
# =============================================================================
checkpoint = torch.load("out/interpolation_exp/checkpoints/checkpoint_300_64.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

## The output as follows:

model.eval() #turn off model weights inconsistence behavior 
with torch.no_grad(): #turn off graident to update wieghts https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    pred = model(x_test2)

### save pred in .ply to visulize it
vertices = pred.reshape((pred.size()[1], 3)).float()    #*self.std + self.mean 

meshdata.save_mesh('pred2.ply',vertices.cpu())

# encode a mesh
z = model.encoder(x_test2)
print("z values:", z)

# decode
def decode(model, z):
    pred_dc = model.decoder(z)
    vertices_dc = pred_dc.reshape((pred_dc.size()[1], 3)).float()
    return vertices_dc.cpu().detach()

meshdata.save_mesh('pred2_dc.ply', decode(model, z))

# z = z + 0.3*j
j = 1
z = z + 0.3*j
print("z = z+0.3*1", z)
meshdata.save_mesh('pred2_03_1.ply', decode(model, z))

# z = 0
z = z*0.0
print("z = z*0.0", z)
meshdata.save_mesh('pred2_0.ply', decode(model, z))

train_eval.eval_error(model, test_loader, device, meshdata, args.out_dir)
## Save the output data.
