import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

print("Testing imports...")

try:
    import aegis.mil_models.mambaMIL as mamba_module

    print("Successfully imported aegis.mil_models.mambaMIL")
    print(
        f"mamba_module.HAS_MAMBA_SSM: {getattr(mamba_module, 'HAS_MAMBA_SSM', 'Not Found')}"
    )
except ImportError as e:
    print(f"Failed to import aegis.mil_models.mambaMIL: {e}")
except Exception as e:
    print(f"Unexpected error importing aegis.mil_models.mambaMIL: {e}")

try:
    from aegis.mil_models.models_factory import mil_model_factory, HAS_MAMBA

    print(f"Successfully imported models_factory. HAS_MAMBA: {HAS_MAMBA}")
except Exception as e:
    print(f"Failed to import models_factory: {e}")


class MockArgs:
    model_type = "mambamil"
    n_classes = 2
    drop_out = 0.25
    task_type = "classification"
    backbone = "resnet50"
    in_dim = 1024


args = MockArgs()

print("\nTesting factory creation...")
try:
    model = mil_model_factory(args, in_dim=1024)
    print("Model created successfully (Unexpected if mamba is missing)")
except ImportError as e:
    print(f"Caught expected ImportError from factory: {e}")
except Exception as e:
    print(f"Caught unexpected exception from factory: {e}")
