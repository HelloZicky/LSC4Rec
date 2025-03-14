from ..model_meta import MetaType, model


class ModelConfig(object):
    def __init__(self):
        self.id_dimension = 8
        self.id_vocab = 500
        self.classifier = [64, 32]
        self.add_plugin = False
        self.mlp_layers = 2


    @staticmethod
    @model("meta_gru4rec", MetaType.ConfigParser)
    @model("meta_gru4rec_large", MetaType.ConfigParser)
    def parse(json_obj):
        conf = ModelConfig()
        conf.id_dimension = json_obj.get("id_dimension")
        conf.id_vocab = json_obj.get("id_vocab")
        conf.classifier = json_obj.get("classifier")
        conf.add_plugin = json_obj.get("add_plugin")
        conf.mlp_layers = json_obj.get("mlp_layers")

        return conf
