from transformers.configuration_utils import PretrainedConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class ZImageTransformer2DModelConfig(PretrainedConfig):
    # Expected _class_name in Diffusers
    _class_name = "ZImageTransformer2DModel"

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        _class_name: str | None = None,  # Parsed from Diffuser config
        **kwargs,
    ):
        super().__init__()
        if _class_name is None:
            logger.warn(
                "Model config expected _class_name %s, but None was provided",
                ZImageTransformer2DModelConfig._class_name,
            )

        if _class_name != ZImageTransformer2DModelConfig._class_name:
            logger.warn(
                "Model config expected _class_name %s, but got %s",
                ZImageTransformer2DModelConfig._class_name,
                _class_name,
            )

        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.n_layers = n_layers
        self.n_refiner_layers = n_refiner_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.cap_feat_dim = cap_feat_dim
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
