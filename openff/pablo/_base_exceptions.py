__all__ = ["PabloError", "ResidueValidationError"]


class PabloError(ValueError):
    pass


class ResidueValidationError(PabloError):
    pass
