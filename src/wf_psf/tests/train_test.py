"""UNIT TESTS FOR PACKAGE MODULE: Train.

This module contains unit tests for the wf_psf.train module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.training import train
from wf_psf.psf_models import psf_models
import tensorflow as tf
import numpy as np
import os
import scipy
from re import search
import logging

logger = logging.getLogger(__name__)

def test_train(
    training_params,
    training_data,
    test_data,
    checkpoint_dir,
    optimizer_dir,
    psf_model_dir,
    tmp_checkpoint_dir,
    tmp_optimizer_dir,
    tmp_psf_model_dir,
    psf_model,
):
    logger.info("Starting training...")
    train.train(
        training_params,
        training_data,
        test_data,
        tmp_checkpoint_dir,
        tmp_optimizer_dir,
        tmp_psf_model_dir,
    )

    logger.info("Training complete")
    weights_type_dict = {
        checkpoint_dir: "checkpoint_callback_",
        psf_model_dir: "psf_model_",
    }

    logger.info(weights_type_dict)
    # Evaluate the weights for each checkpoint callback and the final psf models wrt baseline
    weights_basename = (
        training_params.model_params.model_name + training_params.id_name + "_cycle"
    )

    logger.info("Retrieving psf model.")
    tmp_psf_model = psf_models.get_psf_model(
        training_params.model_params, training_params.training_hparams
    )

    for weights_dir, tmp_weights_dir in zip(
        [checkpoint_dir, psf_model_dir], [tmp_checkpoint_dir, tmp_psf_model_dir]
    ):
        first_cycle = 1
        logger.info(weights_dir)
        if search("psf_model", weights_dir):
            if not training_params.training_hparams.multi_cycle_params.save_all_cycles:
                first_cycle = (
                    training_params.training_hparams.multi_cycle_params.total_cycles
                )

        for cycle in range(
            first_cycle,
            training_params.training_hparams.multi_cycle_params.total_cycles,
        ):
            basename_cycle = (
                weights_dir
                + "/"
                + weights_type_dict[weights_dir]
                + weights_basename
                + str(cycle)
            )
            logger.info(cycle)
            
            tmp_basename_cycle = (
                tmp_weights_dir
                + "/"
                + weights_type_dict[weights_dir]
                + weights_basename
                + str(cycle)
            )

            psf_model.load_weights(basename_cycle)
            saved_model_weights = psf_model.get_weights()

            tmp_psf_model.load_weights(tmp_basename_cycle)
            tmp_saved_model_weights = tmp_psf_model.get_weights()
            logger.info("Evaluating difference...")
            diff = abs(
                np.array(saved_model_weights) - np.array(tmp_saved_model_weights)
            )
            logger.info(diff)
            for arr in diff:
                logger.info(np.mean(arr))
                assert np.mean(arr) < 1.e-11, logger.info("Test failed.")
