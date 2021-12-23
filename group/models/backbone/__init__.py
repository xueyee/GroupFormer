from .inv3 import Inception_v3
from.inv3_version2 import Inception_v3_version2
from .i3d import i3d
from .i3d_v2 import i3d_flow_no_acy,i3d_flow_acy
model_zoo = {
    'inv3': Inception_v3,
    'inv3_version2':Inception_v3_version2,
    'i3d': i3d,
    'i3d_flow_no_acy':i3d_flow_no_acy,
    'i3d_flow_acy':i3d_flow_acy
}

def BackboneBuilder(config):
    model = model_zoo[config.type](config)
    return model
