"""WF_PSF Run.

This module setups the run of the WF_PSF pipeline.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import argparse
from wf_psf.utils.read_config import read_stream, read_conf
from wf_psf.utils.io import FileIOHandler
import os
import logging.config
import logging
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
from wf_psf.training import train
from wf_psf.psf_models import psf_models
from wf_psf.metrics.metrics_refactor import evaluate_model


def setProgramOptions():
    """Define Program Options.

    Set command-line options for
    this program.

    Returns
    -------
    args: type
        Argument Parser Namespace

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conffile",
        "-c",
        type=str,
        required=True,
        help="a configuration file containing program settings.",
    )

    parser.add_argument(
        "--repodir",
        "-r",
        type=str,
        required=True,
        help="the path of the code repository directory.",
    )

    parser.add_argument(
        "--outputdir",
        "-o",
        type=str,
        required=True,
        help="the path of the output directory.",
    )

    args = parser.parse_args()

    return args


def mainMethod():
    """Main Method.

    The main entry point to wavediff program.


    """
    args = setProgramOptions()

    file_handler = FileIOHandler(args.repodir, args.outputdir)
    file_handler.setup_outputs()

    logger = logging.getLogger("wavediff")

    logger.info("#")
    logger.info("# Entering wavediff mainMethod()")
    logger.info("#")

    configs = read_stream(os.path.join(args.repodir, args.conffile))

    data_params = None
    training_params = None
    metrics_params = None
    for conf in configs:
        try:
            if hasattr(conf, "data_conf"):
                data_params = read_conf(os.path.join(args.repodir, conf.data_conf))
                logger.info(data_params)
            else:
                raise ValueError("Data Config file not provided...")
        except FileNotFoundError as e:
            logger.exception(e)
            exit()
        except ValueError as e:
            logger.exception(e)
            exit()

        if hasattr(conf, "training_conf"):
            training_params = read_conf(os.path.join(args.repodir, conf.training_conf))
            logger.info(training_params.training)

        if hasattr(conf, "metrics_conf"):
            metrics_params = read_conf(os.path.join(args.repodir, conf.metrics_conf))
            logger.info(metrics_params.metrics)

    try:
        logger.info("Performing training...")
        simPSF = psf_models.simPSF(training_params.training.model_params)

        training_data = TrainingDataHandler(
            data_params.data.training,
            simPSF,
            training_params.training.model_params.n_bins_lda,
        )

        test_data = TestDataHandler(
            data_params.data.test,
            simPSF,
            training_params.training.model_params.n_bins_lda,
        )

        psf_model, checkpoint_filepath = train.train(
            training_params.training,
            training_data,
            test_data,
            file_handler.get_checkpoint_dir(),
            file_handler.get_optimizer_dir(),
        )

        if metrics_params is not None:
            logger.info("Performing metrics evaluation of trained PSF model...")
            evaluate_model(
                metrics_params.metrics,
                training_params.training,
                training_data,
                test_data,
                psf_model,
                checkpoint_filepath,
                file_handler.get_metrics_dir(),
            )

    except AttributeError:
        logger.info("Training or Metrics not set in configs.yaml. Skipping...")

    if training_params is None:
        try:
            logger.info("Performing metrics evaluation only...")
            # Get Config File for Trained PSF Model
            trained_params = read_conf(
                os.path.join(args.repodir, metrics_params.metrics.trained_model_config)
            )

            logger.info(trained_params.training)

            simPSF = psf_models.simPSF(trained_params.training.model_params)

            checkpoint_filepath = train.filepath_chkp_callback(
                file_handler.get_checkpoint_dir(),
                trained_params.training.model_params.model_name,
                trained_params.training.id_name,
                metrics_params.metrics.saved_training_cycle,
            )

            training_data = TrainingDataHandler(
                data_params.data.training,
                simPSF,
                trained_params.training.model_params.n_bins_lda,
            )

            test_data = TestDataHandler(
                data_params.data.test,
                simPSF,
                trained_params.training.model_params.n_bins_lda,
            )

            psf_model = psf_models.get_psf_model(
                trained_params.training.model_params,
                trained_params.training.training_hparams,
            )

            evaluate_model(
                metrics_params.metrics,
                trained_params.training,
                training_data,
                test_data,
                psf_model,
                checkpoint_filepath,
                file_handler.get_metrics_dir(),
            )
        except AttributeError:
            logger.exception(
                "Configs are not correctly set in configs.yaml.  Please check your config file."
            )

    logger.info("#")
    logger.info("# Exiting wavediff mainMethod()")
    logger.info("#")


if __name__ == "__main__":
    mainMethod()
