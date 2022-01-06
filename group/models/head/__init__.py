from .st_plus_tr_cross_cluster import ST_plus_TR_cross_cluster
from .global_sttr import global_sttr

model_zoo = {
    'global_sttr': global_sttr,
    'st_plus_tr_cross_cluster':ST_plus_TR_cross_cluster,
}

def HeadBuilder(config):
    model = model_zoo[config.type.lower()](config)
    return model
