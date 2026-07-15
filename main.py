# ============================================================================
# Deployment Environment and Resource Profiles:
# The DTFL and the baselines are deployed on a server with the following specifications:
# - Dual-sockets Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz 
# - Four NVIDIA GeForce GTX 1080 Ti GPUs
# - 64 GB of memory

# Each client in the simulation is assigned a distinct simulated CPU and communication resource
# to replicate heterogeneous resources, simulating varying training times based on CPU/network profiles.
# We simulate a heterogeneous environment with varying client capacity in both cross-solo and cross-device FL settings.

# We consider 5 resource profiles:
# 1. 4 CPUs with 100 Mbps
# 2. 2 CPUs with 30 Mbps
# 3. 1 CPU with 30 Mbps
# 4. 0.2 CPU with 30 Mbps
# 5. 0.1 CPU with 10 Mbps communication speed to the server.

# In this implementaion number of tiers is 6 (M=6)
# ============================================================================

import torch
from torch import nn
import torch.nn.functional as F
import math
import glob
import os.path
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

import random
import numpy as np
import os

import time
import sys
import wandb
import argparse
import logging

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from model.resnet import resnet56_SFL_local_tier_7
from model.resnet import resnet110_SFL_fedavg_base
from model.resnet import resnet110_SFL_local_tier_7

from utils.loss import PatchShuffle
from utils.loss import dis_corr
from utils.fedavg import aggregated_fedavg
from utils.privacy import (
    noise_schedule,
    clip_features,
    add_gaussian_noise,
    compute_snr,
    compute_corr_drop,
    get_dp_epsilon,
)

from utils.TierScheduler import TierScheduler
from utils.deats import DEATSConfig, DEATSScheduler
from api.data_preprocessing.ham10000.data_loader import load_partition_data_ham10000
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

import matplotlib
matplotlib.use('Agg')
import copy
# from multiprocessing import Process
# import torch.multiprocessing as mp
# from multiprocessing import Pool


SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "Multi-Tier Splitfed Local Loss"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

def add_args(parser):
    
    parser.add_argument('--running_name', default="DTFL", type=str)
    
    # Optimization related arguments
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_factor', default=0.9, type=float)
    parser.add_argument('--lr_patience', default=10, type=float)
    parser.add_argument('--lr_min', default=0, type=float)
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
 
    # Model related arguments
    parser.add_argument('--model', type=str, default='resnet110', metavar='N',
                        help='neural network used in training')
    
    
    # Data loading and preprocessing related arguments
    parser.add_argument('--dataset', type=str, default='ham10000', metavar='N',
                        help='dataset used for training: ham10000, flower, cifar10, cifar100, cinic10')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
        
    # Federated learning related arguments
    parser.add_argument('--client_epoch', default=1, type=int)
    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--rounds', default=300, type=int)
    parser.add_argument('--whether_local_loss', default=True, type=bool)
    parser.add_argument('--tier', default=5, type=int)
        
    
    # Decorrelation / obfuscation arguments (pre-existing)
    parser.add_argument('--whether_dcor', default=False, type=bool)
    parser.add_argument('--dcor_coefficient', default=0.5, type=float)  # same as alpha in paper
    parser.add_argument('--PatchShuffle', default=0, type=int)

    # -----------------------------------------------------------------------
    # Differential-privacy / smashed-tensor protection arguments
    # (see privacy-implementation-plan.md §1)
    # -----------------------------------------------------------------------
    # Master switch: set True to enable clipping + noise on the smashed tensor
    parser.add_argument('--privacy_enable', default=False, type=bool,
                        help='Enable L2-clipping and Gaussian noise on smashed tensor (DP-SL).')
    # L2 sensitivity / clipping bound C
    parser.add_argument('--clip_C', default=1.0, type=float,
                        help='L2 norm clipping bound for per-sample smashed features.')
    # Initial noise multiplier (sigma at round 0)
    parser.add_argument('--noise_sigma0', default=1.0, type=float,
                        help='Initial Gaussian noise multiplier sigma_0 (noise std = sigma_t * clip_C).')
    # Minimum noise multiplier (floor of the decay schedule)
    parser.add_argument('--noise_sigma_min', default=0.05, type=float,
                        help='Minimum noise multiplier; noise never falls below sigma_min * clip_C.')
    # Multiplicative decay applied each round: sigma_t = max(sigma_min, sigma0 * decay^t)
    parser.add_argument('--noise_decay', default=0.97, type=float,
                        help='Multiplicative per-round noise decay factor in (0, 1).')
    # Target delta for (epsilon, delta)-DP accounting
    parser.add_argument('--privacy_delta', default=1e-5, type=float,
                        help='Target delta for (eps, delta)-DP accounting via opacus.')
    # Enable computation and logging of SNR and correlation-drop metrics
    parser.add_argument('--privacy_metrics', default=True, type=bool,
                        help='Compute and log SNR and correlation-drop privacy metrics to W&B.')
    # How often (in rounds) to log detailed privacy metrics
    parser.add_argument('--privacy_log_interval', default=1, type=int,
                        help='Log privacy metrics every N rounds.')
    
    
    
    # Add the argument for simulation like net_speed_list
    parser.add_argument('--net_speed_list', type=str, default=[100, 30, 30, 30, 10], 
                    metavar='N', help='list of net speeds in mega bytes')
    parser.add_argument('--delay_coefficient_list', type=str, default=[16, 20, 34, 130, 250],
                    metavar='N', help='list of delay coefficients')

    # DEATS — Dynamic Energy-Aware Tier Scheduling (thesis Section 3.3)
    parser.add_argument(
        '--use_deats',
        action='store_true',
        default=True,
        help='Enable battery-aware DEATS tier scheduling (Section 3.3)',
    )
    parser.add_argument(
        '--no_deats',
        dest='use_deats',
        action='store_false',
        help='Use DTFL latency-only tiering without battery model',
    )
    parser.add_argument('--deats_ema_alpha', type=float, default=0.3,
                        help='EMA smoothing for energy rate (Eq. 3.1)')
    parser.add_argument('--deats_battery_min', type=float, default=5.0,
                        help='Battery safety floor %% (Eq. 3.4)')
    parser.add_argument('--deats_energy_scale', type=float, default=0.004,
                        help='Scale mapping compute/comm load to battery drain')
    parser.add_argument('--deats_battery_init_min', type=float, default=65.0,
                        help='Minimum initial battery %% for simulated clients')
    parser.add_argument('--deats_battery_init_max', type=float, default=100.0,
                        help='Maximum initial battery %% for simulated clients')
    
    args = parser.parse_args()
    return args


HAM10000_ALIASES = {"ham10000", "flower", "flower_framework"}


def normalize_dataset_name(dataset_name):
    dataset_key = dataset_name.lower().replace(" ", "_")
    if dataset_key in HAM10000_ALIASES:
        return "ham10000"
    return dataset_key


def resolve_ham10000_data_dir(data_dir):
    candidate_dirs = []
    if data_dir:
        candidate_dirs.append(os.path.normpath(data_dir))
    candidate_dirs.append(os.path.normpath("./Flower Framework"))

    for candidate in candidate_dirs:
        if os.path.isdir(os.path.join(candidate, "Dataset")):
            return candidate

    searched_paths = ", ".join(candidate_dirs)
    raise FileNotFoundError(
        f"Unable to locate the HAM10000 dataset root. Looked for a 'Dataset' directory in: {searched_paths}"
    )


def discover_ham10000_client_count(data_dir):
    resolved_data_dir = resolve_ham10000_data_dir(data_dir)
    client_csvs = sorted(glob.glob(os.path.join(resolved_data_dir, "Dataset", "client_*_train.csv")))
    if not client_csvs:
        raise FileNotFoundError(
            f"No HAM10000 client shard CSVs were found in '{os.path.join(resolved_data_dir, 'Dataset')}'."
        )
    return resolved_data_dir, len(client_csvs)


def build_criterion(dataset_name, train_loader, num_classes, target_device):
    if dataset_name != "ham10000":
        return nn.CrossEntropyLoss()

    labels = np.asarray(train_loader.dataset.target, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = labels.size / np.maximum(class_counts, 1) / num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=target_device)
    return nn.CrossEntropyLoss(weight=class_weights)

DYNAMIC_LR_THRESHOLD = 0.0001
DEFAULT_FRAC = 1.0        # participation of clients


#### Initialization
T_max = 1000

NUM_CPUs = os.cpu_count()

parser = argparse.ArgumentParser()
args = add_args(parser)
args.dataset = normalize_dataset_name(args.dataset)
if args.dataset == "ham10000":
    args.data_dir, detected_client_number = discover_ham10000_client_count(args.data_dir)
    if args.client_number != detected_client_number:
        print(
            f"HAM10000 client count auto-adjusted from {args.client_number} to {detected_client_number} "
            f"based on the discovered Flower client shards."
        )
        args.client_number = detected_client_number
logging.info(args)

    
wandb.init(
    mode="online",
    project="DTFL",
    name="DTFL",# + str(args.tier),
    config=args,
    # tags="Tier1_5",
    # group="ResNet56",
)



SFL_local_tier = resnet56_SFL_local_tier_7

### model selection


if args.dataset == 'cifar10':
    class_num = 10
elif args.dataset == 'cifar100' or args.dataset == 'cinic10':
    class_num = 100
elif args.dataset == 'ham10000':
    class_num = 7
else:
    raise ValueError(f"Unsupported dataset '{args.dataset}'.")


    
if args.model == 'resnet110':
    SFL_local_tier = resnet110_SFL_local_tier_7
    num_tiers = 7
    init_glob_model = resnet110_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)





whether_local_loss = args.whether_local_loss
whether_dcor = args.whether_dcor
dcor_coefficient = args.dcor_coefficient
tier = args.tier
client_epoch = args.client_epoch

# ---------------------------------------------------------------------------
# Privacy globals  (privacy-implementation-plan.md §1 + §2)
# These are read once from args and referenced throughout training.
# ---------------------------------------------------------------------------
privacy_enable       = args.privacy_enable          # master on/off switch
clip_C               = args.clip_C                  # L2 clipping bound C
noise_sigma0         = args.noise_sigma0             # initial noise multiplier
noise_sigma_min      = args.noise_sigma_min          # floor of decay schedule
noise_decay          = args.noise_decay              # per-round decay factor
privacy_delta        = args.privacy_delta            # δ for (ε, δ)-DP
privacy_metrics      = args.privacy_metrics          # enable metric computation
privacy_log_interval = args.privacy_log_interval     # log every N rounds
# sigma_t is updated at the start of each round in the main loop
sigma_t = noise_sigma0  # initialise to sigma_0

if privacy_enable and privacy_metrics:
    print("[Privacy] SNR and correlation-drop metrics enabled.")
elif privacy_metrics:
    print("[Privacy] SNR/correlation metrics requested, but privacy_enable=False; metrics will not be computed.")
client_epoch = np.ones(args.client_number,dtype=int) * client_epoch

client_type_percent = [0.0, 0.0, 0.0, 0.0, 1.0]

if num_tiers == 7:
    client_type_percent = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    tier = 1

    

client_number_tier = (np.dot(args.client_number , client_type_percent))

########## network speed profile of clients

def compute_delay(data_transmitted_client:float, net_speed:float, delay_coefficient:float, duration) -> float:
    net_delay = data_transmitted_client / net_speed
    computation_delay = duration * delay_coefficient
    total_delay = net_delay + computation_delay
    simulated_delay = total_delay
    return simulated_delay

net_speed_list = np.array([100, 200, 500]) * 1024000 ** 2  # MB/s: speed for transmitting data
net_speed_weights = [0.5, 0.25, 0.25]  # weights for each speed level
net_speed = random.choices(net_speed_list, weights=net_speed_weights, k=args.client_number)


net_speed_list = list(np.array(args.net_speed_list) * 1024 ** 2)

net_speed = net_speed_list * (args.client_number // 5 + 1)

delay_coefficient_list = list(np.array(args.delay_coefficient_list) / 14.5)  # to scale on the GPU 

delay_coefficient = delay_coefficient_list * (args.client_number // 5 + 1)  # coeffieient list for simulation computational power
delay_coefficient = list(np.array(delay_coefficient))

############### DEATS initialization (Section 3.3) ###############
deats_scheduler = None
if args.use_deats:
    deats_config = DEATSConfig(
        ema_alpha=args.deats_ema_alpha,
        battery_min=args.deats_battery_min,
        energy_scale=args.deats_energy_scale,
        battery_init_min=args.deats_battery_init_min,
        battery_init_max=args.deats_battery_init_max,
    )
    deats_scheduler = DEATSScheduler(
        num_clients=args.client_number,
        num_tiers=num_tiers,
        config=deats_config,
        seed=SEED,
    )
    print(
        f"DEATS enabled — battery-aware dynamic tiering "
        f"(safety floor {args.deats_battery_min}%)"
    )
else:
    print("DTFL latency-only dynamic tiering (DEATS disabled via --no_deats)")

############### Client profiles definitions ###############
client_cpus_gpus = [(0, 0.5, 0), (1, 0.5, 0), (2, 0, 1), (3, 2, 0), (4, 1, 0),
                    (5, 0.5, 0), (6, 0.5, 0), (7, 0, 1), (8, 2, 0), (9, 1, 0)]





total_time = 0 
# global avg_tier_time_list
avg_tier_time_list = []
max_time_list = pd.DataFrame({'time' : []})
    
client_delay_computing = 0.1
client_delay_net = 0.1



# tier = 1
#===================================================================
# No. of users
num_users = args.client_number
epochs = args.rounds
lr = args.lr

# data transmmission
global data_transmit
model_parameter_data_size = 0 # model parameter 
intermediate_data_size = 0 # intermediate data

# =====
#   load dataset
# ====



def load_data(args, dataset_name):

    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100
    elif dataset_name == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    elif dataset_name == "ham10000":
        data_loader = load_partition_data_ham10000
        args.data_dir = resolve_ham10000_data_dir(args.data_dir)
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    if dataset_name == "cinic10":
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts]
        
    else:
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    return dataset

if args.dataset != "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    
if args.dataset == "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts] = dataset

    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    
    dataset_size = {}
    for i in range(0,len(traindata_cls_counts)):
        dataset_size[i] = sum(traindata_cls_counts[i].values())
    avg_dataset = sum(dataset_size.values()) / len(dataset_size)

dataset_size = {}
if args.dataset != "cinic10":
    for i in range(0,args.client_number):
        dataset_size[i] = len(dataset_train[i].dataset.target)
    avg_dataset = sum(dataset_size.values()) / len(dataset_size)
    


# Functions

########################### Client Selection ##########################

def get_random_user_indices(num_users, DEFAULT_FRAC=0.1):
    m = max(int(DEFAULT_FRAC * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    return idxs_users, m


def calculate_data_size(w_model):
    """
    Calculate the data size (memory usage) of tensors in the w_glob_client_tier for a specific model

    Parameters:
        w_model (dict): Dictionary containing tensors for each model.

    Returns:
        int: Data size (memory usage) of tensors in bytes.
    """
    data_size = 0
    for k in w_model:
        data_size += sys.getsizeof(w_model[k].storage())
        # tensor = w_model[k]
        # data_size += tensor.numel() * tensor.element_size() # this calculate the tensor size, but a little smaller than with using sys
    return data_size

#####
#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side

    # def __init__(self):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_client_side, self).__init__()
        self.input_planes = 64
        self.layer1 = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
            )
        
        # Aux network  fedgkt
        
        self.layer2 = self._layer(block, 16, 1) # layers[0] =1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(16 * 1, classes )  # block.expansion = 1 , classes

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
            
        return nn.Sequential(*netLayers)
        
        
    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))   # here from fedgkt code extracted_features = x without maxpool
        
        # Aux Network output
        # extracted_features = resudial1

        x = self.layer2(resudial1)  # B x 16 x 32 x 32
        # x = self.layer2(x)  # B x 32 x 16 x 16
        # x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        extracted_features = self.fc(x_f)  # B x num_classes

        return extracted_features, resudial1
    
net_glob_client_tier = {}


net_glob_client_tier[1],_ = SFL_local_tier(classes=class_num,tier=5)
net_glob_client,_ = SFL_local_tier(classes=class_num,tier=tier)
for i in range(1,num_tiers+1):
    net_glob_client_tier[i],_ = SFL_local_tier(classes=class_num,tier=i)

    


"""
    Note that we only initialize the client feature extractor to mitigate the difficulty of alternating optimization
"""

if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    #net_glob_client = nn.parallel.DistributedDataParallel(net_glob_client)
    net_glob_client = nn.DataParallel(net_glob_client, device_ids=list(range(torch.cuda.device_count())))  
    for i in range(1, num_tiers+1):
        net_glob_client_tier[i] = nn.DataParallel(net_glob_client_tier[i], device_ids=list(range(torch.cuda.device_count())))

for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)


    
net_glob_client.to(device)


net_glob_server_tier = {}

_, net_glob_server = SFL_local_tier(classes=class_num,tier=tier) # local loss SplitFed
for i in range(1,num_tiers+1):
    _, net_glob_server_tier[i] = SFL_local_tier(classes=class_num,tier=i)
    
    
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server, device_ids=list(range(torch.cuda.device_count())))   # to use the multiple GPUs 
    for i in range(1, num_tiers+1):
        net_glob_server_tier[i] = nn.DataParallel(net_glob_server_tier[i], device_ids=list(range(torch.cuda.device_count())))
        
        
for i in range(1, num_tiers+1):
    net_glob_server_tier[i].to(device)

net_glob_server.to(device)


#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []


criterion = build_criterion(args.dataset, train_data_global, class_num, device)
count1 = 0
count2 = 0

time_train_server_train = 0
time_train_server_train_all = 0

#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    len_min = float('inf')
    index_len_min = 0
    for j in range(0, len(w)):
        if len(w[j]) < len_min:
            len_min = len(w[j])
            index_len_min = j
    w[0],w[index_len_min] = w[index_len_min],w[0]
            
            
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        c = 1
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            c += 1
        w_avg[k] = torch.div(w_avg[k], c)
    return w_avg

def FedAvg_wighted(w, client_sample):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # w_avg[k] += w[i][k] * client_sample[i]  # to solve long error
            w_avg[k] += w[i][k] * client_sample[i].to(w_avg[k].dtype)  # maybe other method can be used
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
best_acc = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_glob_server_tier ={}
net_glob_server_tier[tier].load_state_dict(w_glob_server)
for i in range(1, num_tiers+1):
   w_glob_server_tier[i] = net_glob_server_tier[i].state_dict()
w_locals_server = []
w_locals_server_tier = {}
for i in range(1,num_tiers+1):
    w_locals_server_tier[i]=[]


#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server_tier = {}
net_model_client_tier = {}
client_tier = {}
for i in range (0, num_users):
    client_tier[i] = num_tiers
k = 0
net_model_server = [net_glob_server for i in range(num_users)]
for i in range(len(client_number_tier)):
    for j in range(int(client_number_tier[i])):
        net_model_server_tier[k] = net_glob_server_tier[i+1]
        # net_model_client_tier[k] = net_glob_client_tier[i+1]
        client_tier[k] = i+1
        k +=1
net_server = copy.deepcopy(net_model_server[0]).to(device)
net_server = copy.deepcopy(net_model_server_tier[0]).to(device)
        


#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)
optimizer_server_glob =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
scheduler_server = ReduceLROnPlateau(optimizer_server_glob, 'max', factor=0.8, patience=0, threshold=0.0000001)
patience = args.lr_patience
factor= args.lr_factor
wait=0
new_lr = lr
min_lr = args.lr_min

times_in_server = []


# Server-side function associated with Training
def train_server(fx_client, y, round_idx, l_epoch_count, l_epoch, idx, len_batch, extracted_features):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server, time_train_server_train, time_train_server_train_all, w_glob_server_tier, w_locals_server_tier, w_locals_tier
    global loss_train_collect_user, acc_train_collect_user, lr, total_time, times_in_server, new_lr

    net_server = copy.deepcopy(net_model_server_tier[idx]).to(device)
    net_server.train()

    lr = new_lr
    if args.optimizer == "Adam":
        optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True)
    elif args.optimizer == "SGD":
        optimizer_server = torch.optim.SGD(
            net_server.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.wd,
        )

    time_train_server_s = time.time()
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    fx_server = net_server(fx_client)
    y = y.to(torch.long)
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    net_model_server[idx] = copy.deepcopy(net_server)
    net_model_server_tier[idx] = copy.deepcopy(net_server)
    time_train_server_train += time.time() - time_train_server_s

    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        times_in_server.append(time_train_server_train)
        time_train_server_train_all += time_train_server_train
        total_time += time_train_server_train
        time_train_server_train = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.2f} \tLoss: {:.3f}'.format(
            idx, l_epoch_count, acc_avg_train, loss_avg_train
        ))

        w_server = net_server.state_dict()

        if l_epoch_count == l_epoch - 1:
            l_epoch_check = True
            w_locals_server.append(copy.deepcopy(w_server))
            w_locals_server_tier[client_tier[idx]].append(copy.deepcopy(w_server))

            loss_train_collect_user.append(loss_avg_train)
            acc_train_collect_user.append(acc_avg_train)

            if idx not in idx_collect:
                idx_collect.append(idx)

        if len(idx_collect) == m:
            fed_check = True
            w_locals_tier = w_locals_server
            w_locals_server = []
            w_locals_server_tier = {}
            for i in range(1, num_tiers + 1):
                w_locals_server_tier[i] = []
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

            wandb.log({"Server_Training_Time": time_train_server_train_all, "epoch": round_idx}, commit=False)
            print("Server LR: ", optimizer_server.param_groups[0]["lr"])
            new_lr = optimizer_server.param_groups[0]["lr"]
            wandb.log({"Server_LR": optimizer_server.param_groups[0]["lr"], "epoch": round_idx}, commit=False)

    return dfx_client


def evaluate_global_split_model(client_net, server_net, test_loader, round_idx, eval_tier):
    """Evaluate the aggregated global split model once per FL round."""
    global loss_test_collect, acc_test_collect, acc_avg_all_user_train, loss_avg_all_user_train
    global best_acc, wait, new_lr, fed_check, l_epoch_check

    client_eval = copy.deepcopy(client_net).to(device)
    server_eval = copy.deepcopy(server_net).to(device)
    client_eval.eval()
    server_eval.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            client_output = client_eval(images)
            if isinstance(client_output, tuple):
                _, fx = client_output
            else:
                fx = client_output

            logits = server_eval(fx.to(device))
            loss = criterion(logits, labels)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += logits.argmax(dim=1).eq(labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Global evaluation received an empty test loader.")

    loss_avg_all_user = total_loss / total_samples
    acc_avg_all_user = 100.0 * total_correct / total_samples

    loss_test_collect.append(loss_avg_all_user)
    acc_test_collect.append(acc_avg_all_user)

    if (acc_avg_all_user / 100) > best_acc * (1 + DYNAMIC_LR_THRESHOLD):
        print("- Found better accuracy")
        best_acc = acc_avg_all_user / 100
        wait = 0
    else:
        wait += 1
        print("wait", wait)

    if wait > patience:
        new_lr = max(float(new_lr) * factor, min_lr)
        wait = 0

    fed_check = False
    l_epoch_check = False

    print("==========================================================")
    print("{:^58}".format("DTFL Performance"))
    print("----------------------------------------------------------")
    print(" Eval Tier: {:3d}".format(eval_tier))
    print(" Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}".format(
        round_idx, acc_avg_all_user_train, loss_avg_all_user_train
    ))
    print(" Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}".format(
        round_idx, acc_avg_all_user, loss_avg_all_user
    ))
    print("==========================================================")

    wandb.log({
        "Server_Training_Accuracy": acc_avg_all_user_train,
        "Server_Test_Accuracy": acc_avg_all_user,
        "Server_Test_Loss": loss_avg_all_user,
        "Global_Eval_Tier": eval_tier,
        "epoch": round_idx,
    }, commit=False)

    return loss_avg_all_user, acc_avg_all_user


#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================


# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = client_epoch[idx]
        self.ldr_train = dataset_train[idx]
        self.ldr_test = dataset_test[idx]
            
        

    def train(self, net, round_idx):
        net.train()
        self.lr , lr = new_lr, new_lr

        if args.optimizer == "Adam":
            optimizer_client =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
        elif args.optimizer == "SGD":
            optimizer_client =  torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                                      nesterov=True,
                                                      weight_decay=args.wd)

        time_client=0
        client_intermediate_data_size = 0
        CEloss_client_train = []
        Dcorloss_client_train = []

        # ---------------------------------------------------------------
        # Privacy metric accumulators (privacy-implementation-plan.md §4,§5)
        # Collected per-batch and averaged at end of training call.
        # ---------------------------------------------------------------
        batch_snr_list       = []   # SNR values across batches
        batch_corr_drop_list = []   # correlation-drop values across batches

        for local_epoch in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_s = time.time()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()

                #---------forward prop-------------
                extracted_features, fx = net(images)

                # -----------------------------------------------------------
                # Privacy mechanism on the smashed tensor fx
                # (privacy-implementation-plan.md §3)
                #
                # Execution order:
                #   1. Optional PatchShuffle obfuscation (pre-existing).
                #   2. L2 per-sample norm clipping to bound sensitivity.
                #   3. Gaussian noise injection calibrated to sigma_t * clip_C.
                #   4. The noisy tensor is sent to the server — the server
                #      receives *only* the protected version.
                # -----------------------------------------------------------
                if args.PatchShuffle == 1:
                    # Apply patch shuffle before any further processing
                    fx_for_server = fx.clone().detach().requires_grad_(False)
                    fx_for_server = PatchShuffle(fx_for_server)
                else:
                    fx_for_server = fx.clone().detach()

                if privacy_enable:
                    # Step 2: L2 per-sample clipping (§3 step 2)
                    fx_clipped = clip_features(fx_for_server, clip_C)

                    # Step 3: Gaussian noise injection (§3 step 3)
                    #   noise std per element = sigma_t * clip_C
                    fx_noisy = add_gaussian_noise(fx_clipped, sigma_t, clip_C)

                    # The server receives the noisy (protected) activation
                    client_fx = fx_noisy.clone().detach().requires_grad_(True)

                    # Step 4: Compute and accumulate privacy metrics (§4, §5)
                    # Only compute when privacy_metrics is enabled and on the
                    # configured logging interval (checked at epoch end below).
                    if privacy_metrics:
                        # §4  SNR: signal power / noise power
                        snr_val = compute_snr(fx_clipped, fx_noisy)
                        batch_snr_list.append(snr_val)

                        # §5  Correlation drop: relative dCor reduction
                        corr_drop_val = compute_corr_drop(
                            images, fx_clipped, fx_noisy
                        )
                        batch_corr_drop_list.append(corr_drop_val)
                else:
                    # Privacy disabled: send the (possibly shuffled) tensor
                    # unchanged — identical to the original behaviour.
                    client_fx = fx_for_server.clone().detach().requires_grad_(True)

                # -----------------------------------------------------------
                # Sending activations to server and receiving gradients
                # -----------------------------------------------------------
                time_client += time.time() - time_s
                dfx = train_server(client_fx, labels, round_idx, local_epoch, self.local_ep, self.idx, len_batch, _)

                #--------backward prop -------------
                time_s = time.time()

                labels = labels.to(torch.long)
                loss = criterion(extracted_features, labels) # to solve change dataset
                CEloss_client_train.append(((1 - dcor_coefficient)*loss.item()))

                if whether_dcor:
                    Dcor_value = dis_corr(images, fx)
                    loss = (1 - dcor_coefficient) * loss + dcor_coefficient * Dcor_value
                    Dcorloss_client_train.append(((dcor_coefficient) * Dcor_value))

                loss.backward()

                optimizer_client.step()
                time_client += time.time() - time_s

                client_intermediate_data_size += (sys.getsizeof(client_fx.storage()) +
                                      sys.getsizeof(labels.storage()))

        global intermediate_data_size
        intermediate_data_size += client_intermediate_data_size

        # -------------------------------------------------------------------
        # W&B logging — client-level metrics
        # -------------------------------------------------------------------
        # Pre-existing decorrelation loss log
        wandb.log({"Client{}_DcorLoss".format(self.idx): float(sum(Dcorloss_client_train)), "epoch": round_idx}, commit=False)
        wandb.log({"Client{}_time_not_scaled (s)".format(self.idx): time_client, "epoch": round_idx}, commit=False)

        # Privacy metrics: averaged over all batches in this training call
        # (privacy-implementation-plan.md §4, §5, §7)
        if privacy_enable and privacy_metrics:
            avg_snr = None
            avg_corr_drop = None
            if batch_snr_list:
                avg_snr = sum(batch_snr_list) / len(batch_snr_list)
                wandb.log(
                    {"Privacy/SNR_client{}".format(self.idx): avg_snr, "epoch": round_idx},
                    commit=False,
                )
            if batch_corr_drop_list:
                avg_corr_drop = sum(batch_corr_drop_list) / len(batch_corr_drop_list)
                wandb.log(
                    {"Privacy/corr_drop_client{}".format(self.idx): avg_corr_drop, "epoch": round_idx},
                    commit=False,
                )
            if avg_snr is not None and avg_corr_drop is not None:
                print(
                    f"[Privacy] Round {round_idx:3d} | Client {self.idx:2d} | "
                    f"SNR={avg_snr:.6f} | corr_drop={avg_corr_drop:.6f}"
                )
            elif not batch_snr_list and not batch_corr_drop_list:
                print(
                    f"[Privacy] Round {round_idx:3d} | Client {self.idx:2d} | "
                    "SNR/corr_drop unavailable: no privacy metric batches were collected."
                )

        return net.state_dict(), time_client, client_intermediate_data_size
    
    def evaluate_glob(self, net, ell): # I wrote this part
        net.eval()
        epoch_acc = []
        epoch_loss = []
           
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                labels = labels.to(torch.long)
                loss = criterion(fx, labels)
                acc = calculate_accuracy(fx, labels)
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
            prGreen('Model Test =>                     \tAcc: {:.3f} \tLoss: {:.4f}'
                    .format(epoch_acc[-1], epoch_loss[-1])) # After model update the test for all agent should be same. because the test dataset is same and after convergence all agents model are same
                
            return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
            
                
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                          

# Data transmission
client_tier_all = []
client_tier_all.append(copy.deepcopy(client_tier))
total_training_time = 0
time_train_server_train_all_list = []

client_sample = np.ones(num_users)

def calculate_client_samples(train_data_local_num_dict, idxs_users, dataset):
    """
    Calculates the number of samples for each client in a federated dataset.

    Args:
        train_data_local_num_dict (dict): A dictionary mapping client indices to the number
            of samples in their local training dataset.
        num_users (int): The total number of clients in the federated dataset.
        dataset (str): The name of the federated dataset.

    Returns:
        A list of length num_users, where the i-th element represents the number of samples
        that the i-th client should use for training.
    """
    num_users = len(idxs_users)
    client_sample = []
    total_samples = sum(train_data_local_num_dict.values())
    for idx in idxs_users:
        client_sample.append(train_data_local_num_dict[idx] / total_samples * num_users)
    return client_sample



for i in range(0, num_users):
    wandb.log({"Client{}_Tier".format(i): num_tiers - client_tier[i] + 1, "epoch": -1}, commit=False)
    if deats_scheduler is not None:
        wandb.log({
            f"Client{i}_Battery": deats_scheduler.battery.get_battery(i),
            "epoch": -1,
        }, commit=False)

#------------ Training And Testing  -----------------
net_glob_client.train()
w_glob_client_tier ={}

#copy weights
for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()

# net_glob_client_tier[tier].load_state_dict(w_glob_client)
w_glob_client_tier[tier] = net_glob_client_tier[tier].state_dict()


# to start with same weigths 
for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)
    
w_glob = copy.deepcopy(init_glob_model.state_dict())

for t in range(1, num_tiers+1):
    for k in w_glob_client_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            k1 = k1[7:] # remove the 'module.' prefix
            
            
        
        if (k1 == 'fc.bias' or k1 == 'fc.weight'):
            continue 
        
        w_glob_client_tier[t][k] = w_glob[k1]
    for k in w_glob_server_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            k1 = k1[7:]
        w_glob_server_tier[t][k] = w_glob[k1]
        
    net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
    net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])
    
w_locals_tier, w_locals_client, w_locals_server = [], [], []


# w_glob_client = init_glob_model.state_dict() # copy weights
# net_glob_client_tier[tier].load_state_dict(w_glob_client) # copy weights
net_model_client_tier = {}
for i in range(1, num_tiers+1):
    net_model_client_tier[i] = net_glob_client_tier[i]
    net_model_client_tier[i].train()
for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()


# optimizer for every elient
optimizer_client_tier = {}
for i in range(0, num_users): # one optimizer for every tier/ client
    if args.optimizer == "Adam":
        optimizer_client_tier[i] =  torch.optim.Adam(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
    elif args.optimizer == "SGD":
        optimizer_client_tier[i] =  torch.optim.SGD(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, momentum=0.9,
                                                          nesterov=True,
                                                          weight_decay=args.wd)


# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

simulated_delay_historical_df = pd.DataFrame()
start_time = time.time() 

client_observed_times = pd.DataFrame()
torch.manual_seed(SEED)
simulated_delay= np.zeros(num_users)

for i in range(0, num_users): # maybe remove this part
    continue
    data_server_to_client = 0
    for k in w_glob_client_tier[client_tier[i]]:
        data_server_to_client += sys.getsizeof(w_glob_client_tier[client_tier[i]][k].storage())
    simulated_delay[i] = data_server_to_client / net_speed[i]

# Generate a list of randomly chosen user indices based on the number of users and the default fraction
idxs_users, m = get_random_user_indices(num_users, DEFAULT_FRAC)

# record all data transmitted in last involeved epoch
data_transmitted_client_all = {}

computation_time_clients = {}
for k in range(num_users):
    computation_time_clients[k] = []

# Main loop over rounds
for iter in range(epochs):
    # -------------------------------------------------------------------
    # Privacy: adaptive noise schedule  (privacy-implementation-plan.md §2)
    # Compute sigma_t for this round: sigma_t = max(sigma_min, sigma0 * decay^t)
    # sigma_t is declared global so Client.train() can read it without passing
    # it as a parameter (consistent with how other globals like `new_lr` are used).
    # -------------------------------------------------------------------
    sigma_t = noise_schedule(iter, noise_sigma0, noise_sigma_min, noise_decay)  # noqa: F811
    if privacy_enable:
        wandb.log({"Privacy/sigma_t": sigma_t, "epoch": iter}, commit=False)
    if iter == int(50): # here we can change how the enviroement randomly change 
        continue
        delay_coefficient[0] = delay_coefficient_list[2]
        net_speed[0] = net_speed_list[2]
        delay_coefficient[1] = delay_coefficient_list[4]
        net_speed[1] = net_speed_list[4]
        delay_coefficient[2] = delay_coefficient_list[4]
        net_speed[2] = net_speed_list[4]
        delay_coefficient[3] = delay_coefficient_list[0]
        net_speed[3] = net_speed_list[0]
        delay_coefficient[4] = delay_coefficient_list[0]
        net_speed[4] = net_speed_list[0]
        delay_coefficient[5] = delay_coefficient_list[4]
        net_speed[5] = net_speed_list[4]
        delay_coefficient[6] = delay_coefficient_list[4]
        net_speed[6] = net_speed_list[4]
        delay_coefficient[7] = delay_coefficient_list[0]
        net_speed[7] = net_speed_list[0]
        delay_coefficient[8] = delay_coefficient_list[1]
        net_speed[8] = net_speed_list[1]
        delay_coefficient[9] = delay_coefficient_list[1]
        net_speed[9] = net_speed_list[1]

               
        
    
    # Initialize empty lists for client weights
    w_locals_client = []
    w_locals_client_tier = {}
    
    # Initialize a dictionary to store client weights based on their tiers
    w_locals_client_tier = {i: [] for i in range(1, num_tiers+1)}
    
    # Initialize a numpy array to store client time
    client_observed_time = np.zeros(num_users)
    
    processes = []
    
    simulated_delay= np.zeros(num_users)

    # ------------------------------------------------------------------
    # DEATS dropout handling (FIX)
  
    if deats_scheduler is not None:
        active_idxs_users = [i for i in idxs_users if deats_scheduler.battery.is_alive(i)]
        dropped_idxs_users = [i for i in idxs_users if not deats_scheduler.battery.is_alive(i)]
    else:
        active_idxs_users = list(idxs_users)
        dropped_idxs_users = []

    # Log dropped-out clients (battery at safety floor) for this round
    for idx in dropped_idxs_users:
        print(
            f"Client {idx} skipped — battery at safety floor "
            f"({args.deats_battery_min}%)"
        )

        # Recharge the idle client so it can rejoin in a later round, instead
        # of being lost permanently for the rest of training.
        deats_scheduler.battery.recharge_idle(idx)

        wandb.log({
            f"Client{idx}_Battery": deats_scheduler.battery.get_battery(idx),
            f"Client{idx}_DroppedOut": 1,
            "epoch": iter,
        }, commit=False)

    # Federation this round completes based on the clients that train.
    # `m` is read as a global inside train_server's federation trigger.
    m = len(active_idxs_users)

    # If every sampled client has dropped out, there is nothing to train or
    # aggregate this round — skip safely instead of deadlocking the loop.
    if m == 0:
        print(f"Round {iter}: all sampled clients dropped out — skipping round.")
        wandb.log({"Active_Clients": 0, "epoch": iter}, commit=True)
        continue

    wandb.log({"Active_Clients": m, "epoch": iter}, commit=False)

    for idx in active_idxs_users:
        # Log the client tier for each client in WandB
        wandb.log({"Client{}_Tier".format(idx): num_tiers - client_tier[idx] + 1, "epoch": iter}, commit=False) # tier 1 smallest model
        
        
        data_server_to_client = calculate_data_size(w_glob_client_tier[client_tier[idx]])
        simulated_delay[idx] = data_server_to_client / net_speed[idx]
            
            
        client_model_parameter_data_size = 0
        time_train_test_s = time.time()
        net_glob_client = net_model_client_tier[client_tier[idx]]
        w_glob_client_tier[client_tier[idx]] = net_glob_client_tier[client_tier[idx]].state_dict() # may be I can eliminate this line
        local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = [], idxs_test = [])
            

        # Training ------------------
        [w_client, duration, client_intermediate_data_size] = local.train(
            net=copy.deepcopy(net_glob_client).to(device),
            round_idx=iter,
        )
            
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_client_tier[client_tier[idx]].append(copy.deepcopy(w_client))
        
        client_observed_time[idx] = duration
        
        
        client_model_parameter_data_size = calculate_data_size(w_client)
        model_parameter_data_size += client_model_parameter_data_size         
        
        data_transmitted_client = client_intermediate_data_size + client_model_parameter_data_size
        
        # add to dic last observation
        data_transmitted_client_all[idx] = data_transmitted_client
        
        simulated_delay[idx] += compute_delay(data_transmitted_client, net_speed[idx]
                                              , delay_coefficient[idx], duration) # this is simulated delay

        if deats_scheduler is not None:
            energy_drain = deats_scheduler.battery.drain(
                client_id=idx,
                duration=duration,
                data_transmitted=data_transmitted_client,
                net_speed=net_speed[idx],
                delay_coefficient=delay_coefficient[idx],
                code_tier=client_tier[idx],
                num_tiers=num_tiers,
            )
            deats_scheduler.update_ema(idx, energy_drain)
            deats_scheduler.record_relative_energy(idx, energy_drain)
            wandb.log({
                f"Client{idx}_Battery": deats_scheduler.battery.get_battery(idx),
                f"Client{idx}_EnergyDrain": energy_drain,
                f"Client{idx}_EMA_Energy": deats_scheduler.battery.clients[idx].ema_energy_rate,
                "epoch": iter,
            }, commit=False)

        wandb.log({"Client{}_Total_Delay".format(idx): simulated_delay[idx], "epoch": iter}, commit=False)
        
    server_wait_first_to_last_client = (max(simulated_delay * client_epoch) - min(simulated_delay * client_epoch))
    training_time = (max(simulated_delay))
    total_training_time += training_time
    if iter == 0:
        first_training_time = training_time
    wandb.log({"Training_time_clients": total_training_time, "epoch": iter}, commit=False)
    times_in_server = []
    time_train_server_train_all_list.append(time_train_server_train_all)
    time_train_server_train_all = 0

    # -------------------------------------------------------------------
    # Privacy: per-round aggregate logging and (ε, δ)-DP accounting
    # (privacy-implementation-plan.md §6, §7)
    # -------------------------------------------------------------------
    if privacy_enable and privacy_metrics and (iter % privacy_log_interval == 0):
        # Log the current noise multiplier (already logged above, duplicated
        # here with commit=False so it groups with the DP epsilon in one step)
        wandb.log({"Privacy/noise_multiplier": sigma_t, "epoch": iter}, commit=False)

        # (ε, δ)-DP accounting via Opacus (§6)
        # Sampling rate q = batch_size / avg local dataset size
        q = args.batch_size / max(avg_dataset, 1)
        # Steps consumed this round: one optimizer step per batch per client epoch
        # (conservative: uses the maximum local epoch count).
        # Use the active-client count (m) so DP accounting reflects the
        # clients that actually trained after DEATS dropouts.
        steps_this_round = int(m * max(client_epoch))
        epsilon = get_dp_epsilon(
            sample_rate=q,
            noise_multiplier=sigma_t,
            num_steps=steps_this_round,
            delta=privacy_delta,
        )
        if epsilon is not None:
            wandb.log({"Privacy/epsilon": epsilon, "epoch": iter}, commit=False)
            print(f"[Privacy] Round {iter:3d} | sigma_t={sigma_t:.4f} | epsilon={epsilon:.4f} | delta={privacy_delta}")
        else:
            print(f"[Privacy] Round {iter:3d} | sigma_t={sigma_t:.4f} | (ε,δ) accounting unavailable (install opacus)")
     
    simulated_delay[simulated_delay==0] = np.nan  # convert zeros to nan, for when some clients not involved in the epoch
    simulated_delay_historical_df = pd.concat([simulated_delay_historical_df, pd.DataFrame(simulated_delay).T], ignore_index=True)
    client_observed_times = pd.concat([client_observed_times, pd.DataFrame(client_observed_time).T], ignore_index=True)
    client_epoch_last = client_epoch.copy()
    
    idxs_users, m = get_random_user_indices(num_users, DEFAULT_FRAC)
    
        
    [client_tier, T_max, computation_time_clients] = TierScheduler(
        computation_time_clients,
        T_max,
        client_tier_all=client_tier_all,
        delay_history=simulated_delay_historical_df,
        num_tiers=num_tiers,
        client_epoch=client_epoch,
        num_users=num_users,
        dataset_size=dataset_size,
        batch_size=args.batch_size,
        data_transmitted_client_all=data_transmitted_client_all,
        net_speed=net_speed,
        deats_scheduler=deats_scheduler,
        remaining_rounds=max(epochs - iter - 1, 1),
    )
    wandb.log({"max_time": T_max, "epoch": iter}, commit=False)
                                                    
    client_tier_all.append(copy.deepcopy(client_tier))
    

    
    for i in client_tier.keys():  # assign each server-side to its tier model
        net_model_server_tier[i] = net_glob_server_tier[client_tier[i]]

    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("{:^59}".format("Model Aggregation"))
    print("-----------------------------------------------------------")
    
    # calculate the number of samples in each client
    # Use the clients that actually trained this round (active_idxs_users),
    # in the same order their weights were appended to w_locals_client.
    # Using idxs_users here was incorrect because it is reassigned above to
    # the *next* round's sample, which misaligned the FedAvg weighting.
    client_sample = calculate_client_samples(train_data_local_num_dict, active_idxs_users, args.dataset) # same order as appended weights
        

    
    
            
    w_glob = aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, whether_local_loss, client_sample, idxs_users) # w_locals_tier is for server-side
    
    for t in range(1, num_tiers+1):
        for k in w_glob_client_tier[t].keys():
            if k in w_glob_server_tier[t].keys():  # This is local updading  // another method can be updating and supoose its similar to global model
                if w_locals_client_tier[t] != []:
                    w_glob_client_tier[t][k] = FedAvg(w_locals_client_tier[t])[k]
                    continue
                else:
                    continue 
            
            w_glob_client_tier[t][k] = w_glob[k]
        for k in w_glob_server_tier[t].keys():
            w_glob_server_tier[t][k] = w_glob[k]
            
        net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
        net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])

    global_eval_tier = num_tiers
    evaluate_global_split_model(
        net_glob_client_tier[global_eval_tier],
        net_glob_server_tier[global_eval_tier],
        test_data_global,
        iter,
        global_eval_tier,
    )
 

    
    print(f'Size of Total Model Parameter Data Transferred {(model_parameter_data_size/1024**2):,.2f} Mega Byte')
    print(f'Size of Total Intermediate Data Transferred {(intermediate_data_size/1024**2):,.2f} Mega Byte')

    wandb.log({"Model_Parameter_Data_Transmission(MB) ": model_parameter_data_size/1024**2, "epoch": iter}, commit=False)
    wandb.log({"Intermediate_Data_Transmission(MB) ": intermediate_data_size/1024**2, "epoch": iter}, commit=True)
    
    
elapsed = (time.time() - start_time)/60
    
#===================================================================================     

print("Training and Evaluation completed!")    
    

#=============================================================================
#                         Program Completed
#=============================================================================
