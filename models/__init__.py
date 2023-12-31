from models.bp.fc import BP_FC_MNIST, BP_FC_CIFAR10, BP_FC_CIFAR100
from models.bp.conv import BP_CONV_CIFAR10, BP_CONV_CIFAR100, BP_CONVBN_CIFAR10, BP_CONVBN_CIFAR100
from models.bp.resnet18 import BP_RESNET18_CIFAR10, BP_RESNET18_CIFAR100, BP_RESNET18_TINYIMAGENET

from models.dfa.fc import DFA_FC_MNIST, DFA_FC_CIFAR10, DFA_FC_CIFAR100
from models.dfa.conv import DFA_CONV_CIFAR10, DFA_CONV_CIFAR100, DFA_CONVBN_CIFAR10, DFA_CONVBN_CIFAR100
from models.dfa.resnet18 import DFA_RESNET18_CIFAR10, DFA_RESNET18_CIFAR100, DFA_RESNET18_TINYIMAGENET

from models.las.fc import LAS_FC_MNIST, LAS_FC_CIFAR10, LAS_FC_CIFAR100
from models.las.conv import LAS_CONV_CIFAR10, LAS_CONV_CIFAR100, LAS_CONVBN_CIFAR10, LAS_CONVBN_CIFAR100
from models.las.resnet18 import LAS_RESNET18_CIFAR10, LAS_RESNET18_CIFAR100, LAS_RESNET18_TINYIMAGENET

from models.ltas.fc import LTAS_FC_MNIST, LTAS_FC_CIFAR10, LTAS_FC_CIFAR100
from models.ltas.conv import LTAS_CONV_CIFAR10, LTAS_CONV_CIFAR100, LTAS_CONVBN_CIFAR10, LTAS_CONVBN_CIFAR100
from models.ltas.resnet18 import LTAS_RESNET18_CIFAR10, LTAS_RESNET18_CIFAR100, LTAS_RESNET18_TINYIMAGENET

from models.shllows.fc import SHALLOW_FC_MNIST, SHALLOW_FC_CIFAR10, SHALLOW_FC_CIFAR100
from models.shllows.conv import SHALLOW_CONV_CIFAR10, SHALLOW_CONV_CIFAR100


__all__ = [
    # BP Family
    "BP_FC_MNIST", "BP_FC_CIFAR10", "BP_FC_CIFAR100", 
    "BP_CONV_CIFAR10", "BP_CONV_CIFAR100", "BP_CONVBN_CIFAR10", "BP_CONVBN_CIFAR100",
    "BP_RESNET18_CIFAR10", "BP_RESNET18_CIFAR100", "BP_RESNET18_TINYIMAGENET",

    # DFA Family
    "DFA_FC_MNIST", "DFA_FC_CIFAR10", "DFA_FC_CIFAR100", 
    "DFA_CONV_CIFAR10", "DFA_CONV_CIFAR100", "DFA_CONVBN_CIFAR10", "DFA_CONVBN_CIFAR100",
    "DFA_RESNET18_CIFAR10", "DFA_RESNET18_CIFAR100", "DFA_RESNET18_TINYIMAGENET",

    # LAS Family
    "LAS_FC_MNIST", "LAS_FC_CIFAR10", "LAS_FC_CIFAR100", 
    "LAS_CONV_CIFAR10", "LAS_CONV_CIFAR100", "LAS_CONVBN_CIFAR10", "LAS_CONVBN_CIFAR100", 
    "LAS_RESNET18_CIFAR10", "LAS_RESNET18_CIFAR100", "LAS_RESNET18_TINYIMAGENET",

    # LTAS Family
    "LTAS_FC_MNIST", "LTAS_FC_CIFAR10", "LTAS_FC_CIFAR100", 
    "LTAS_CONV_CIFAR10", "LTAS_CONV_CIFAR100", "LTAS_CONVBN_CIFAR10", "LTAS_CONVBN_CIFAR100", 
    "LTAS_RESNET18_CIFAR10", "LTAS_RESNET18_CIFAR100", "LTAS_RESNET18_TINYIMAGENET",

    # Shallow Family
    "SHALLOW_FC_MNIST", "SHALLOW_FC_CIFAR10", "SHALLOW_FC_CIFAR100",
    "SHALLOW_CONV_CIFAR10", "SHALLOW_CONV_CIFAR100",
]