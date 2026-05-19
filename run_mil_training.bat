@echo off
:: Script to run MIL training with specified parameters
::
:: NOTE: If using mambamil model, ensure mamba-ssm is installed:
::   pip install mamba-ssm
::   or
::   pip install causal-conv1d>=1.2.0
::   (CUDA extensions will be compiled automatically)
::
:: Available MIL model types (--model_type):
::   - att_mil        : Attention-based MIL (ABMIL/DAttention)
::   - trans_mil       : Transformer-based MIL (TransMIL)
::   - max_mil         : Max pooling MIL
::   - mean_mil        : Mean pooling MIL
::   - s4model         : S4-based MIL
::   - wikgmil         : WiKG (Graph-based MIL)
::   - diffabmil       : Differentiable Attention MIL
::   - hgachc          : Hierarchical Graph Attention Cross-Head Communication
::   - rrtmil          : RRT (Region-based Transformer MIL)
::   - dsmil           : Dual-stream MIL
::   - mambamil        : Mamba-based MIL (supports SRMamba, Mamba, BiMamba)
::   - moemil          : Mixture of Experts MIL
::
:: Model-specific parameters:
::   - mambamil: --mamba_layers (default: 2), --mamba_rate (default: 10), --mamba_type (SRMamba/Mamba/BiMamba)
::   - moemil: --embed_dim (default: 512), --num_experts (default: 4)
::   - trans_mil, att_mil, max_mil, mean_mil, s4model: --activation (relu/gelu, default varies by model)

setlocal enabledelayedexpansion

echo Starting MIL training script...
echo Current directory: %CD%

:: Initialize conda
:: Try to find conda activation script if conda command is not directly available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda not found in PATH. Trying standard locations...
    if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    ) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    ) else (
        echo Error: Could not find conda. Please ensure conda is in your PATH or installed in standard locations.
        exit /b 1
    )
)

:: Activate conda environment
echo Activating conda environment: aegis
call conda activate aegis
if %errorlevel% neq 0 (
    echo Error: Failed to activate conda environment 'aegis'
    echo Please ensure the environment exists: conda env list
    exit /b 1
)

echo Conda environment activated.
for /f "tokens=*" %%i in ('where python') do set PYTHON_PATH=%%i
echo Python path: %PYTHON_PATH%

:: Verify Python script exists
if not exist "train_mil_run.py" (
    echo Error: train_mil_run.py not found in current directory: %CD%
    echo Please ensure you're running this script from the project root directory
    exit /b 1
)

:: Set DATA_ROOT
set "DATA_ROOT=E:\features_uni_v2"

echo Launching training...

:: Build command arguments
:: Note: Using ^ for line continuation
set PYTHON_ARGS=--data_root_dir "E:\features_uni_v2" ^
 --train_csv Data/tcga-ot_train.csv ^
 --test_csv Data/tcga-ot_test.csv ^
 --val_csv Data/tcga-ot_val.csv ^
 --label_col OncoTreeCode ^
 --patient_id_col case_id ^
 --slide_id_col slide_id ^
 --results_dir ./results ^
 --task multiclass ^
 --task_type classification ^
 --exp_code tcga_ot_multiclass_s1 ^
 --seed 42 ^
 --num_workers 16 ^
 --log_data ^
 --testing ^
 --k 1 ^
 --k_start 0 ^
 --k_end 1 ^
 --model_type att_mil ^
 --backbone uni_v2 ^
 --in_dim 1536 ^
 --max_epochs 200 ^
 --lr 1e-4 ^
 --reg 1e-5 ^
 --opt adam ^
 --drop_out 0.25 ^
 --early_stopping ^
 --preloading no ^
 --weighted_sample ^
 --batch_size 4 ^
 --use_hdf5 ^
 --n_subsamples 2048 
@REM  --memmap_bin_path "E:\dataset.bin" ^
@REM  --memmap_json_path "E:\output.json"

:: Run the python script
:: Run the code on GPU 1 with CUDA_VISIBLE_DEVICES=1
set CUDA_VISIBLE_DEVICES=1

python -u train_mil_run.py %PYTHON_ARGS%

if %errorlevel% neq 0 (
    echo Training failed with error code %errorlevel%
    exit /b %errorlevel%
)

endlocal
