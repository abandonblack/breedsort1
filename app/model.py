"""兼容入口：默认使用 SE-ResNet34。"""

from app.model_seresnet34 import SEResNet34, build_model

__all__ = ["SEResNet34", "build_model"]
