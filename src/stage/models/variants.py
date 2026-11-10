from stage.models.stage import StageModel
from stage.models.variants_single import (
    GCN_Only_Model,
    LSTM_Only_Model,
)

from stage.models.variants_interaction import (
    Serial_GCN_LSTM_Model,
)

from stage.models.variants_fusion import (
    GCN_LSTM_Add,
    GCN_LSTM_Concat,
    GCN_LSTM_CrossAttn,
)

from stage.models.variants_backbone import (
    GCN_Trans_RVG_Model,
    GAT_LSTM_RVG_Model,
)


MODEL_REGISTRY = {
    "stage_final": StageModel,
    "stage": StageModel,

    "gcn_only": GCN_Only_Model,
    "lstm_only": LSTM_Only_Model,
    "inter_serial": Serial_GCN_LSTM_Model,

    "fusion_add": GCN_LSTM_Add,
    "fusion_concat": GCN_LSTM_Concat,
    "fusion_cross_attn": GCN_LSTM_CrossAttn,

    "gcn_trans_rvg": GCN_Trans_RVG_Model,
    "gat_lstm_rvg": GAT_LSTM_RVG_Model,
}


def get_model_class(name):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model variant: {name}. Available: {available}")
    return MODEL_REGISTRY[name]


def build_model(name, num_classes):
    model_cls = get_model_class(name)
    return model_cls(num_classes=num_classes)
