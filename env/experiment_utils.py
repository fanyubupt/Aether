import logging
import os
import threading
import time
from enum import Enum, unique
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

import torch
from torch.utils.tensorboard import SummaryWriter

import datetime
import csv

import numpy as np

import json
import sys

import time

class ExperimentLoggingManager:
    """
    A singleton class that manages logging, TensorBoard writers, and model saving
    for experiments. It ensures centralized and thread-safe handling of logs, metrics, and model checkpoints,
    organized according to specified directory modes.
    """

    _instance = None
    _lock = threading.Lock()

    @unique
    class LOG_DIR_MODE(Enum):
        DATE_FIRST = 0
        NUMBER_FIRST = 1
        CATEGORY_FIRST = 2
    LOG_DIR_MODE_ROOT_DICT = {LOG_DIR_MODE.DATE_FIRST: 'log', LOG_DIR_MODE.NUMBER_FIRST: 'log', LOG_DIR_MODE.CATEGORY_FIRST: ''}

    DIR_MODE = LOG_DIR_MODE.DATE_FIRST

    def __new__(cls, log_root=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExperimentLoggingManager, cls).__new__(cls)
                if log_root is None:
                    log_root = cls.LOG_DIR_MODE_ROOT_DICT[cls.DIR_MODE]
                cls._instance._initialize(log_root, cls.DIR_MODE)
            return cls._instance

    def _initialize(self, log_root, dir_mode):
        self.log_queue = Queue(-1)

        if dir_mode == self.LOG_DIR_MODE.DATE_FIRST:
            log_root = os.path.join(log_root, time.strftime("experiment_%Y%m%d"))
            run_num = 1
            while FileTool.ensure_dir(os.path.join(log_root, f'run{run_num}')):
                run_num += 1
            self.log_dir = os.path.join(log_root, f'run{run_num}')
            self.log_file_path = os.path.join(self.log_dir,
                                              f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}_PID{os.getpid()}_PPID{threading.get_ident()}.log')
            self.tb_dir_path = self.log_dir
            self.model_dir_path = os.path.join(self.log_dir, 'models')
            FileTool.ensure_dir(self.model_dir_path)
        elif dir_mode == self.LOG_DIR_MODE.NUMBER_FIRST:
            run_num = 1
            while FileTool.ensure_dir(os.path.join(log_root, f'run{run_num}')):
                run_num += 1
            self.log_dir = os.path.join(log_root, f'run{run_num}')
            self.log_file_path = os.path.join(self.log_dir, f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}_PID{os.getpid()}_PPID{threading.get_ident()}.log')
            self.tb_dir_path = self.log_dir
            self.model_dir_path = os.path.join(self.log_dir, 'models')
            FileTool.ensure_dir(self.model_dir_path)
        elif dir_mode == self.LOG_DIR_MODE.CATEGORY_FIRST:
            self.log_dir = log_root
            self.log_file_path = os.path.join(self.log_dir, 'logs', f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}_PID{os.getpid()}_PPID{threading.get_ident()}.log')
            self.tb_dir_path = os.path.join(self.log_dir, 'tbs', f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}')
            self.model_dir_path = os.path.join(self.log_dir, 'models', f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}')
            FileTool.ensure_dir(os.path.join(self.log_dir, 'logs'))
            FileTool.ensure_dir(self.tb_dir_path)
            FileTool.ensure_dir(self.model_dir_path)

        file_handler = logging.FileHandler(self.log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.queue_listener = QueueListener(self.log_queue, file_handler, respect_handler_level=True)
        self.queue_listener.start()

        self.loggers = {}
        self.writers = {}
        self.model_savers = {}

    class ExperimentLogger:
        def __init__(self, logger_name, log_file_path, log_queue):
            self.experiment_name = logger_name
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(QueueHandler(log_queue))

            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        def log(self, message, level=logging.INFO):
            if level == logging.DEBUG:
                self.logger.debug(message)
            elif level == logging.INFO:
                self.logger.info(message)
            elif level == logging.WARNING:
                self.logger.warning(message)
            elif level == logging.ERROR:
                self.logger.error(message)
            elif level == logging.CRITICAL:
                self.logger.critical(message)

        def close(self):
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)

    class ExperimentTBWriter:
        def __init__(self, writer_name, tb_dir_path):
            self.writer_name = writer_name
            self.writer = SummaryWriter(tb_dir_path)

        def log_metric(self, tag, value, step):
            self.writer.add_scalar(tag, value, step)

        def log_metrics(self, main_tag, tag_value_dict, step):
            self.writer.add_scalars(main_tag, tag_value_dict, step)

        def close(self):
            self.writer.close()

    class ModelSaver:
        def __init__(self, model_name, model_dir_path):
            self.model_name = model_name
            self.model_dir_path = model_dir_path
            self.model_metric_list = []
            self.highest_metric_index = -1
            self.lowest_metric_index = -1

        def save(self, model, total_step, metric, epoch=None, episode=None, step=None):
            model_name = f'{self.model_name}_total_step{total_step}'
            if epoch is not None:
                model_name += f'_epoch{epoch}'
            if episode is not None:
                model_name += f'_episode{episode}'
            if step is not None:
                model_name += f'_step{step}'
            if self.highest_metric_index == -1 or metric > self.model_metric_list[self.highest_metric_index][1]:
                self.highest_metric_index = len(self.model_metric_list) - 1
            elif self.lowest_metric_index == -1 or metric < self.model_metric_list[self.lowest_metric_index][1]:
                self.lowest_metric_index = len(self.model_metric_list) - 1
            model_path = os.path.join(self.model_dir_path, f'{model_name}.pth')
            self.model_metric_list.append((model_name, metric, (total_step, epoch, step)))
            torch.save(model, model_path)

        def get_highest_metric(self):
            best_model_name, best_metric, _ = self.model_metric_list[self.highest_metric_index]
            best_model_path = os.path.join(self.model_dir_path, f'{best_model_name}.pth')
            return torch.load(best_model_path), best_model_name, best_metric

        def get_lowest_metric(self):
            best_model_name, best_metric, _ = self.model_metric_list[self.lowest_metric_index]
            best_model_path = os.path.join(self.model_dir_path, f'{best_model_name}.pth')
            return torch.load(best_model_path), best_model_name, best_metric

    def get_logger(self, logger_name):
        if logger_name not in self.loggers:
            self.loggers[logger_name] = self.ExperimentLogger(logger_name, self.log_file_path, self.log_queue)
        return self.loggers[logger_name]

    def get_writer(self, writer_name):
        assert threading.get_ident() == threading.main_thread().ident, "SummaryWriter is only allowed in the main thread." # Check if main thread
        if writer_name not in self.writers:
            self.writers[writer_name] = self.ExperimentTBWriter(writer_name, self.tb_dir_path)
        return self.writers[writer_name]

    def get_model_saver(self, model_name):
        import torch
        if model_name not in self.model_savers:
            self.model_savers[model_name] = self.ModelSaver(model_name, self.model_dir_path)
        return self.model_savers[model_name]

    def close_all(self):
        self.queue_listener.stop()
        for logger in self.loggers.values():
            logger.close()
        for writer in self.writers.values():
            writer.close()

class Timer:
    """
    A class to represent a timer.

    This class is used to track multiple time intervals and record the start time.
    It is commonly used in scenarios where time measurement and recording are needed.
    """

    def __init__(self):
        self._times = []
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise RuntimeError('Timer is already running. Use stop() to stop it.')
        self._start_time = time.time()

    def stop(self):
        if self._start_time is None:
            raise RuntimeError('Timer is not running. Use start() to start it.')
        self._times.append(time.time() - self._start_time)
        self._start_time = None
        return self._times[-1]

    def avg(self):
        return np.mean(self._times) if self._times else 0

    def total(self):
        return np.sum(self._times)

    def cumsum(self):
        return np.cumsum(self._times).tolist()

    def reset(self):
        self._times = []

    @property
    def times(self):
        return list(self._times)
