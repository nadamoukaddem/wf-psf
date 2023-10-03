"""UNIT TESTS FOR PACKAGE MODULE: Metrics.

This module contains unit tests for the wf_psf.metrics module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.training import train
from wf_psf.metrics.metrics_interface import MetricsParamsHandler, evaluate_model
from wf_psf.psf_models import psf_models
import tensorflow as tf
import numpy as np
import os

cwd = os.getcwd()


@pytest.fixture(scope="module")
def psf_model_path(metrics_params):
    return os.path.join(
        cwd,
        "src/wf_psf/tests",
        metrics_params.trained_model_path,
        metrics_params.model_save_path,
    )


@pytest.fixture(scope="session")
def weights_path_basename(psf_model_path, metrics_params, training_params):
    weights_path = (
        psf_model_path
        + "/"
        + metrics_params.model_save_path
        + "_"
        + training_params.model_params.model_name
        + training_params.id_name
        + "_cycle"
        + metrics_params.saved_training_cycle
    )

    return weights_path


def test_eval_metrics_polychromatic_lowres(
    metrics_params,
    training_params,
    weights_path_basename,
    training_data,
    psf_model,
    test_dataset,
    main_metrics,
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    # Load the trained model weights
    psf_model.load_weights(weights_path_basename)

    poly_metric = metrics_handler.evaluate_metrics_polychromatic_lowres(
        psf_model, simPSF_np, test_dataset
    )

    tol = 1.0e-7
    ratio_rmse = abs(
        1.0 - main_metrics["test_metrics"]["poly_metric"]["rmse"] / poly_metric["rmse"]
    )
    ratio_rel_rmse = abs(
        1.0
        - main_metrics["test_metrics"]["poly_metric"]["rel_rmse"]
        / poly_metric["rel_rmse"]
    )
    ratio_std_rmse = abs(
        1.0
        - main_metrics["test_metrics"]["poly_metric"]["std_rmse"]
        / poly_metric["std_rmse"]
    )
    ratio_rel_std_rmse = abs(
        1.0
        - main_metrics["test_metrics"]["poly_metric"]["std_rel_rmse"]
        / poly_metric["std_rel_rmse"]
    )

    assert ratio_rmse < tol
    assert ratio_rel_rmse < tol
    assert ratio_std_rmse < tol
    assert ratio_rel_std_rmse < tol


def test_evaluate_metrics_opd(
    metrics_params,
    training_params,
    weights_path_basename,
    training_data,
    psf_model,
    test_dataset,
    main_metrics,
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    ## Load the trained model weights
    psf_model.load_weights(weights_path_basename)

    opd_metric = metrics_handler.evaluate_metrics_opd(
        psf_model, simPSF_np, test_dataset
    )

    tol = 1.0e-9
    ratio_rmse_opd = abs(
        1
        - main_metrics["test_metrics"]["opd_metric"]["rmse_opd"]
        / opd_metric["rmse_opd"]
    )
    ratio_rel_rmse_opd = abs(
        1.0
        - main_metrics["test_metrics"]["opd_metric"]["rel_rmse_opd"]
        / opd_metric["rel_rmse_opd"]
    )
    ratio_rmse_std_opd = abs(
        1.0
        - main_metrics["test_metrics"]["opd_metric"]["rmse_std_opd"]
        / opd_metric["rmse_std_opd"]
    )
    ratio_rel_rmse_std_opd = abs(
        1.0
        - main_metrics["test_metrics"]["opd_metric"]["rel_rmse_std_opd"]
        / opd_metric["rel_rmse_std_opd"]
    )

    assert ratio_rmse_opd < tol
    assert ratio_rel_rmse_opd < tol
    assert ratio_rmse_std_opd < tol
    assert ratio_rel_rmse_std_opd < tol


def test_eval_metrics_mono_rmse(
    metrics_params,
    training_params,
    weights_path_basename,
    training_data,
    psf_model,
    test_dataset,
    main_metrics,
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    ## Load the trained model weights
    psf_model.load_weights(weights_path_basename)

    mono_metric = metrics_handler.evaluate_metrics_mono_rmse(
        psf_model, simPSF_np, test_dataset
    )

    nlambda = len(mono_metric["rmse_lda"])

    tol = 1.0e-9
    ratio_rmse_mono = abs(
        1
        - np.sum(
            np.array(main_metrics["test_metrics"]["mono_metric"]["rmse_lda"])
            / np.array(mono_metric["rmse_lda"])
        )
        / nlambda
    )
    ratio_rel_rmse_mono = abs(
        1.0
        - np.sum(
            np.array(main_metrics["test_metrics"]["mono_metric"]["rel_rmse_lda"])
            / np.array(mono_metric["rel_rmse_lda"])
        )
        / nlambda
    )
    ratio_rmse_std_mono = abs(
        1.0
        - np.sum(
            np.array(main_metrics["test_metrics"]["mono_metric"]["std_rmse_lda"])
            / np.array(mono_metric["std_rmse_lda"])
        )
        / nlambda
    )

    ratio_rel_rmse_std_mono = abs(
        1.0
        - np.sum(
            np.array(main_metrics["test_metrics"]["mono_metric"]["std_rel_rmse_lda"])
            / np.array(mono_metric["std_rel_rmse_lda"])
        )
        / nlambda
    )

    assert ratio_rmse_mono < tol
    assert ratio_rel_rmse_mono < tol
    assert ratio_rmse_std_mono < tol
    assert ratio_rel_rmse_std_mono < tol


def test_evaluate_metrics_shape(
    metrics_params,
    training_params,
    weights_path_basename,
    training_data,
    psf_model,
    test_dataset,
    main_metrics,
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    ## Load the trained model weights
    psf_model.load_weights(weights_path_basename)

    shape_metric = metrics_handler.evaluate_metrics_shape(
        psf_model, simPSF_np, test_dataset
    )

    tol = 1.0e-9
    ratio_rmse_e1 = abs(
        1.0
        - main_metrics["test_metrics"]["shape_results_dict"]["rmse_e1"]
        / shape_metric["rmse_e1"]
    )

    ratio_std_rmse_e1 = abs(
        1.0
        - main_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e1"]
        / shape_metric["std_rmse_e1"]
    )

    ratio_rel_rmse_e1 = abs(
        1.0
        - main_metrics["test_metrics"]["shape_results_dict"]["rel_rmse_e1"]
        / shape_metric["rel_rmse_e1"]
    )

    ratio_std_rel_rmse_e1 = abs(
        1.0
        - main_metrics["test_metrics"]["shape_results_dict"]["std_rel_rmse_e1"]
        / shape_metric["std_rel_rmse_e1"]
    )

    ratio_rmse_e2 = abs(
        1.0
        - main_metrics["test_metrics"]["shape_results_dict"]["rmse_e2"]
        / shape_metric["rmse_e2"]
    )

    ratio_std_rmse_e2 = abs(
        1.0
        - main_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e2"]
        / shape_metric["std_rmse_e2"]
    )

    ratio_rmse_R2_meanR2 = abs(
        1.0
        - main_metrics["test_metrics"]["shape_results_dict"]["rmse_R2_meanR2"]
        / shape_metric["rmse_R2_meanR2"]
    )

    assert ratio_rmse_e1 < tol
    assert ratio_std_rmse_e1 < tol
    assert ratio_rel_rmse_e1 < tol
    assert ratio_std_rel_rmse_e1 < tol
    assert ratio_rmse_e2 < tol
    assert ratio_std_rmse_e2 < tol
    assert ratio_rmse_R2_meanR2 < tol
