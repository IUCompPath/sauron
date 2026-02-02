from aegis.feature_extraction.models.patch_encoders.factory import encoder_factory

# Updated imports to reflect the new structure
from aegis.mil_models.ABMIL import DAttention
from aegis.mil_models.DiffABMIL import DifferentiableAttentionMIL
from aegis.mil_models.dsmil import DSMIL
from aegis.mil_models.hgachc import HGACrossHeadCom
from aegis.mil_models.MaxMIL import MaxMIL
from aegis.mil_models.MeanMIL import MeanMIL
from aegis.mil_models.moemil import MoEMIL
from aegis.mil_models.rrtmil import RRT as rrtmil
from aegis.mil_models.S4MIL import S4Model
from aegis.mil_models.TransMIL import TransMIL
from aegis.mil_models.WIKGMIL import WiKG

try:
    from aegis.mil_models.mambaMIL import MambaMIL, HAS_MAMBA_SSM

    HAS_MAMBA = HAS_MAMBA_SSM
except ImportError:
    HAS_MAMBA = False
except Exception as e:
    print(f"Warning: MambaMIL could not be imported: {e}")
    HAS_MAMBA = False


def mil_model_factory(args, in_dim=None):
    """
    Factory function to create and return an instance of a MIL model.
    """
    # Determine input dimension from backbone if not provided
    if in_dim is None and hasattr(args, "backbone"):
        # This part assumes a feature extractor factory that can provide embedding dims.
        # If not available, in_dim must be provided in args.
        try:
            in_dim = encoder_factory(args.backbone).embedding_dim
        except Exception:
            print(
                f"Could not infer in_dim from backbone {args.backbone}. Using args.in_dim."
            )
            in_dim = args.in_dim

    # Determine if the task is survival analysis
    is_survival_task = (
        getattr(args, "task_type", "classification").lower() == "survival"
    )

    model_type = args.model_type.lower()

    if model_type == "att_mil":
        # Assuming DAttention is the intended class for 'att_mil'
        return DAttention(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            activation=getattr(args, "activation", "relu"),
            is_survival=is_survival_task,
        )
    elif model_type == "trans_mil":
        return TransMIL(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            activation=getattr(args, "activation", "gelu"),
            is_survival=is_survival_task,
        )
    elif model_type == "max_mil":
        return MaxMIL(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            activation=getattr(args, "activation", "relu"),
            is_survival=is_survival_task,
        )
    elif model_type == "mean_mil":
        return MeanMIL(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            activation=getattr(args, "activation", "relu"),
            is_survival=is_survival_task,
        )
    elif model_type == "s4model":
        return S4Model(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            activation=getattr(args, "activation", "gelu"),
            is_survival=is_survival_task,
        )
    elif model_type == "wikgmil":
        return WiKG(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            is_survival=is_survival_task,
        )
    elif model_type == "diffabmil":
        return DifferentiableAttentionMIL(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            is_survival=is_survival_task,
        )
    elif model_type == "hgachc":
        return HGACrossHeadCom(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            is_survival=is_survival_task,
        )
    elif model_type == "rrtmil":
        return rrtmil(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            is_survival=is_survival_task,
        )
    elif model_type == "dsmil":
        return DSMIL(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            is_survival=is_survival_task,
        )

    elif model_type == "moemil":
        return MoEMIL(
            in_dim=in_dim,
            n_classes=args.n_classes,
            embed_dim=getattr(args, "embed_dim", 512),
            num_experts=getattr(args, "num_experts", 4),
            dropout_rate=args.drop_out,
            is_survival=is_survival_task,
        )
    elif model_type == "mambamil":
        if not HAS_MAMBA:
            raise ImportError(
                "MambaMIL is not available. Please install mamba-ssm and causal-conv1d."
            )
        return MambaMIL(
            in_dim=in_dim,
            n_classes=args.n_classes,
            dropout_rate=args.drop_out,
            activation=getattr(args, "activation", "relu"),
            is_survival=is_survival_task,
            layer=getattr(args, "layer", 2),
            rate=getattr(args, "rate", 10),
            type=getattr(args, "mamba_type", "SRMamba"),
        )
    else:
        raise ValueError(f"Unknown MIL model type: {args.model_type}")
