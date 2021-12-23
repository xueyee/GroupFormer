from .base_fc import BaseFC
from .base_tr import BaseTR
from .full_tr import FullTR
from .st_tr import STTR
from .st_plus_tr import ST_plus_TR
from .st_plus_tr_para import ST_plus_TR_para
from .global_sttr import global_sttr
from .frame_seq_tr import Frame_TR
from .st_plus_tr_cross import ST_plus_TR_cross
from .st_plus_tr_knn import ST_plus_TR_knn
from .st_plus_tr_knn_v2 import ST_plus_TR_knn_v2
from .st_plus_tr_knn_v3 import ST_plus_TR_knn_v3
from .early_tr import early_tr_gl
from .st_plus_tr_no_global import ST_plus_TR_no_global
from .NT_tr import ST_local_TR
from .NT_tr_plus import ST_local_TR_plus
from .NT_tr_plus_no_global import ST_local_TR_plus_no_global
from .base_cluster_tr import Base_cluster_TR
from .st_plus_tr_cross_cluster import ST_plus_TR_cross_cluster
from .nt_sttr_cluster import cluster_nttr
from .st_plus_tr_cross_cluster_v2 import ST_plus_TR_cross_cluster_v2
from .nt_sttr_cluster_v2 import cluster_nttr_v2
model_zoo = {
    'base_fc': BaseFC,
    'base_tr': BaseTR,
    'full_tr': FullTR,
    'st_tr':STTR,
    'st_plus_tr': ST_plus_TR,
    'st_plus_tr_parra':ST_plus_TR_para,
    'global_sttr': global_sttr,
    'frame_tr':Frame_TR,
    'st_plus_tr_cross':ST_plus_TR_cross,
    'st_plus_tr_knn':ST_plus_TR_knn,
    'st_plus_tr_knn_v2':ST_plus_TR_knn_v2,
    'st_plus_tr_knn_v3':ST_plus_TR_knn_v3,
    'early_tr':early_tr_gl,
    'st_plus_tr_no_global':ST_plus_TR_no_global,
    'local_tr': ST_local_TR,
    'local_tr_plus':ST_local_TR_plus,
    'local_tr_plus_no_global': ST_local_TR_plus_no_global,
    'base_cluster_tr':Base_cluster_TR,
    'st_plus_tr_cross_cluster':ST_plus_TR_cross_cluster,
    'st_plus_tr_cross_cluster_v2':ST_plus_TR_cross_cluster_v2,
    'cluster_nttr':cluster_nttr,
    'cluster_nttr_v2': cluster_nttr_v2,
}

def HeadBuilder(config):
    model = model_zoo[config.type.lower()](config)
    return model
