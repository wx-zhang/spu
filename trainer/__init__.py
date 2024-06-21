from trainer.frozenclip import FrozenCLIP
from trainer.finetune import Finetune
from trainer.spu import SPU
from trainer.distill import Distillation
from trainer.mas import MAS
from trainer.loraewc import LoRAEWC
from trainer.zscl import ZSCL
from trainer.slca import SLCA
from trainer.spg import SPG
from trainer.sparsecl import SparseCL

METHOD = {
    'frozenclip': FrozenCLIP,
    'flyp': Finetune,
    'er': Finetune,
    'lwf': Distillation,
    'prd': Distillation,
    'mas': MAS,
    'loraewc': LoRAEWC,
    'zscl': ZSCL,
    'slca': SLCA,
    'spg': SPG,
    'sparsecl': SparseCL,
    
    
    
    
    
    'spu': SPU,
    
}