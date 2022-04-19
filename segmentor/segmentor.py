from lib.registry import Registry

SEGMENTOR_REGISTRY = Registry("SEGMENTOR")


def get_segmentor(segmentor_name):
    name = segmentor_name.upper()
    return SEGMENTOR_REGISTRY.get(name)


class Segmentor:
    
    def build_dataset(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError



# @SEGMENTOR_REGISTRY.register()
# class your_own_segmentor(Segmentor):

