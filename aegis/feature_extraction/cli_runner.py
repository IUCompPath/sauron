# aegis/feature_extraction/cli_runner.py (This replaces your root feature_extract script)
import argparse
import logging
import os
import sys
from queue import Queue
from threading import Thread

import torch

from aegis.feature_extraction.batch_processing import batch_consumer, batch_producer
from aegis.feature_extraction.models.patch_encoders.factory import (
    encoder_factory as patch_encoder_factory,
)
from aegis.feature_extraction.models.segmentation.factory import (
    segmentation_model_factory,
)
from aegis.feature_extraction.models.slide_encoders.factory import (
    encoder_factory as slide_encoder_factory,
)

# Relative imports within the aegis package
from aegis.feature_extraction.processor import Processor

# IMPORTANT: Ensure aegis/parse/argparse.py is renamed to aegis/parse/cli_parsers.py
from aegis.parse.cli_parsers import parse_feature_extraction_arguments

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_processor(args, wsi_dir_for_processor: str):
    """
    Initialize the aegis Processor with arguments.
    `wsi_dir_for_processor` is typically `args.wsi_dir` or a batch cache directory.
    """
    return Processor(
        job_dir=args.job_dir,
        wsi_source=wsi_dir_for_processor,  # This is the directory the processor will *read from*
        wsi_ext=args.wsi_ext,
        wsi_cache=args.wsi_cache,
        clear_cache=args.clear_cache,
        skip_errors=args.skip_errors,
        custom_mpp_keys=args.custom_mpp_keys,
        custom_list_of_wsis=args.custom_list_of_wsis,
        max_workers=args.max_workers,
        reader_type=args.reader_type,
        search_nested=args.search_nested,
    )


def run_feature_extraction_task(processor: Processor, args):
    """
    Execute the specified task using the aegis Processor.
    """

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.task == "seg":
        # Instantiate segmentation model and artifact remover if requested by user
        segmentation_model = segmentation_model_factory(
            args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        artifact_remover_model = None
        if args.remove_artifacts:
            artifact_remover_model = segmentation_model_factory(
                "grandqc_artifact",  # This model handles general artifacts
                # remove_penmarks_only=False (default behavior)
            )
        elif args.remove_penmarks:  # Only if remove_artifacts is NOT set
            artifact_remover_model = segmentation_model_factory(
                "grandqc_artifact",
                remove_penmarks_only=True,  # Specialized for penmarks
            )

        # run segmentation
        processor.run_segmentation_job(
            segmentation_model=segmentation_model,
            seg_mag=segmentation_model.target_mag,  # Use model's recommended target_mag
            holes_are_tissue=not args.remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=(
                args.seg_batch_size
                if args.seg_batch_size is not None
                else args.batch_size
            ),
            device=device_str,
        )
    elif args.task == "coords":
        # Derive coords_dir_name if not provided
        coords_dir_name = (
            args.coords_dir_name
            or f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"
        )
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            patch_dir_name=coords_dir_name,  # Pass as `patch_dir_name`
            min_tissue_proportion=args.min_tissue_proportion,
        )
    elif args.task == "feat":
        # Derive coords_dir_name if not provided, for feature extraction
        coords_dir_for_feat = (
            args.coords_dir_name
            or f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"
        )

        if args.slide_encoder is None:
            # Patch Feature Extraction
            patch_encoder = patch_encoder_factory(
                args.patch_encoder, weights_path=args.patch_encoder_ckpt_path
            )
            processor.run_patch_feature_extraction_job(
                coords_h5_dir=os.path.join(
                    args.job_dir, coords_dir_for_feat, "patches"
                ),  # Path to the 'patches' subfolder
                patch_encoder=patch_encoder,
                device=device_str,
                saveas="h5",  # Hardcoded to h5 for now, as it's common for MIL
                batch_limit=(
                    args.feat_batch_size
                    if args.feat_batch_size is not None
                    else args.batch_size
                ),
            )
        else:
            # Slide Feature Extraction
            slide_encoder = slide_encoder_factory(args.slide_encoder)
            processor.run_slide_feature_extraction_job(
                coords_h5_dir=os.path.join(
                    args.job_dir, coords_dir_for_feat
                ),  # Path to the job_dir/patch_dir_name
                slide_encoder=slide_encoder,
                device=device_str,
                saveas="h5",  # Hardcoded to h5 for now
                batch_limit_for_patch_features=(
                    args.feat_batch_size
                    if args.feat_batch_size is not None
                    else args.batch_size
                ),
            )
    elif args.task == "cache":
        # In this mode, we only populate the cache, the main loop in main() handles it
        # The processor's populate_cache method will be called.
        processor.populate_cache()
    else:
        raise ValueError(f"Invalid task: {args.task}")


def run_feature_extraction_job(args: argparse.Namespace):  # Renamed `main` function
    """
    Main function to run the feature extraction job based on provided arguments.
    """
    # === Handle Caching / Parallel processing ===
    if (
        args.wsi_cache and args.task != "cache"
    ):  # If caching is enabled for processing tasks (not just populating)
        # We need to run a producer-consumer setup
        # from multiprocessing import Lock # No explicit Lock needed for this queue setup

        queue = Queue(maxsize=1)  # Queue to hold batch IDs for processing

        # Collect all valid slides first (from original source directory)
        # Use a temporary Processor instance just for WSI path collection without caching setup
        temp_processor_for_path_collection = Processor(
            job_dir=args.job_dir,  # required but not used for path collection
            wsi_source=args.wsi_dir,
            wsi_ext=args.wsi_ext,
            custom_list_of_wsis=args.custom_list_of_wsis,
            search_nested=args.search_nested,
            max_workers=args.max_workers,
            reader_type=args.reader_type,
            wsi_cache=None,  # Don't pass cache here, we are just collecting paths
            clear_cache=False,  # Not relevant for path collection
            skip_errors=True,  # Allow path collection to skip errors and report.
        )
        all_valid_slides_from_source = temp_processor_for_path_collection.wsis[
            0
        ]  # Get the list of WSI objects

        # Extract just the original paths from the WSI objects
        original_wsi_paths = [w.original_path for w in all_valid_slides_from_source]

        logger.info(
            f"[Feature Extraction Runner] Found {len(original_wsi_paths)} valid slides in {args.wsi_dir}."
        )

        # Explicitly warm up the first batch outside the producer thread
        # The producer needs a processor instance that is configured to copy files
        # from `args.wsi_dir` to `args.wsi_cache`.
        producer_file_copier_processor = Processor(
            job_dir=args.job_dir,  # Used by Processor generally
            wsi_source=args.wsi_dir,  # This is the source directory for `populate_cache`
            wsi_ext=args.wsi_ext,
            wsi_cache=args.wsi_cache,  # This is the destination for `populate_cache`
            clear_cache=False,  # Don't clear during initial populate
            skip_errors=args.skip_errors,
            custom_mpp_keys=args.custom_mpp_keys,
            custom_list_of_wsis=args.custom_list_of_wsis,  # This will filter `self.wsis` for copying
            max_workers=args.max_workers,
            reader_type=args.reader_type,
            search_nested=args.search_nested,
        )
        # `populate_cache` on this producer instance will copy the initial batch.
        # It operates on `self.wsis` which are determined by `wsi_source` and `custom_list_of_wsis`.
        # `start_idx=0` ensures it focuses on the first chunk of slides if all_valid_slides_from_source
        # were passed during the Processor's initialization indirectly.
        producer_file_copier_processor.populate_cache(start_idx=0)
        queue.put(0)  # Put ID for first batch to trigger consumer

        # Factory function for consumer to create a Processor instance pointing to the batch cache
        def processor_factory_for_consumer(batch_local_dir: str) -> Processor:
            # Create a new argparse.Namespace object for the local processor instance
            # This ensures modifications to `local_args` don't affect the main `args`
            local_args = argparse.Namespace(**vars(args))
            local_args.wsi_dir = (
                batch_local_dir  # Point processor to the local cache batch directory
            )
            local_args.wsi_cache = None  # Disable caching for this local processor
            local_args.custom_list_of_wsis = (
                None  # Custom list already filtered, now process locally
            )
            local_args.search_nested = (
                False  # Already collected, no need to search nested locally
            )
            local_args.clear_cache = (
                args.clear_cache
            )  # Pass through clear_cache decision
            local_args.custom_mpp_keys = args.custom_mpp_keys  # Pass through MPP keys
            local_args.reader_type = args.reader_type  # Pass through reader type
            local_args.skip_errors = args.skip_errors  # Pass through skip_errors
            local_args.max_workers = args.max_workers  # Pass through max_workers

            # The WSI objects in this local processor will now be constructed from `batch_local_dir`
            return initialize_processor(
                local_args, wsi_dir_for_processor=batch_local_dir
            )

        # Function to run the desired task for the consumer
        def run_task_for_consumer_fn(processor_instance: Processor, task_name: str):
            # Temporarily set the task for the current run within this thread context
            original_task_arg = args.task
            args.task = task_name
            try:
                run_feature_extraction_task(processor_instance, args)
            finally:
                args.task = original_task_arg  # Restore original task arg

        producer = Thread(
            target=batch_producer,
            args=(
                queue,
                original_wsi_paths,
                min(len(original_wsi_paths), args.cache_batch_size),
                args.cache_batch_size,
                args.wsi_cache,
            ),
        )

        consumer = Thread(
            target=batch_consumer,
            args=(
                queue,
                args.task,
                args.wsi_cache,
                processor_factory_for_consumer,
                run_task_for_consumer_fn,
            ),
        )

        logger.info(
            "[Feature Extraction Runner] Starting producer and consumer threads for parallel processing."
        )
        producer.start()
        consumer.start()
        producer.join()  # Wait for producer to finish (all slides copied)
        consumer.join()  # Wait for consumer to finish (all slides processed)

    else:
        # === Sequential mode or cache-only task ===
        if args.task == "cache":
            # In cache-only mode, we just populate the cache and exit.
            # No need for consumer/producer threads.
            processor_instance = initialize_processor(args, args.wsi_dir)
            processor_instance.populate_cache()
            logger.info("Cache population task completed.")
        else:
            # Run tasks sequentially
            processor_instance = initialize_processor(args, args.wsi_dir)
            tasks = ["seg", "coords", "feat"] if args.task == "all" else [args.task]
            for task_name in tasks:
                args.task = task_name  # Set current task
                run_feature_extraction_task(processor_instance, args)
            logger.info("Sequential processing task completed.")


if __name__ == "__main__":
    # This block is for direct execution during development/testing outside of package
    # For package usage, `aegis.cli:feature_extract_main` will be called.
    args = parse_feature_extraction_arguments()
    run_feature_extraction_job(args)
