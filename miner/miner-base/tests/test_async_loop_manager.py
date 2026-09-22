import threading
import time
from unittest.mock import Mock, patch

import pytest
import torch
from miner_base.async_loop_manager import AsyncLoopManager, CudaEventQueueItem
from miner_base.gateway_client import MinerRpcConfig, MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.comm.dataclasses import MiningJob, OpenedBlockInfo


@pytest.fixture
def miner_config():
    return MinerRpcConfig(transport="uds", socket_path="/temp/gateway.pipe")


@pytest.fixture
def mock_mining_job():
    mock_job = Mock(spec=MiningJob)
    mock_job.incomplete_header_bytes = b"test_bytes"
    mock_job.target = "test_target"
    return mock_job


class TestAsyncLoopManagerInit:
    def test_init_with_default_settings(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)

        assert isinstance(manager._conf, MinerSettings)
        assert manager._client_config == miner_config
        assert manager._mining_job is None

    def test_init_multiple_instances_independent(self):
        config1 = MinerRpcConfig(transport="uds", socket_path="/temp/gateway1.pipe")
        config2 = MinerRpcConfig(transport="uds", socket_path="/temp/gateway2.pipe")

        manager1 = AsyncLoopManager(config1, MinerSettings())
        manager2 = AsyncLoopManager(config2, MinerSettings())

        assert manager1._stop_event is not manager2._stop_event
        assert manager1._cuda_event_queue is not manager2._cuda_event_queue
        assert manager1._block_results is not manager2._block_results


class TestAsyncLoopManagerStartStop:
    @patch("miner_base.async_loop_manager._make_client")
    def test_start_initializes_components(self, mock_make_client, miner_config, mock_mining_job):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client

        manager = AsyncLoopManager(miner_config, None)
        manager.start()

        assert manager._thread is not None
        assert manager._thread.daemon is True
        assert manager._pool is not None
        assert manager._client is mock_client

        manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_start_raises_if_already_started(self, mock_make_client, miner_config, mock_mining_job):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client

        manager = AsyncLoopManager(miner_config, None)
        manager.start()

        with pytest.raises(RuntimeError, match="Already started"):
            manager.start()

        manager.stop()

    @patch("miner_base.async_loop_manager._make_client")
    def test_stop_cleans_up_components(self, mock_make_client, miner_config, mock_mining_job):
        mock_client = Mock(spec=MiningClient)
        mock_client.get_mining_info.return_value = mock_mining_job
        mock_make_client.return_value = mock_client

        manager = AsyncLoopManager(miner_config, None)
        manager.start()
        time.sleep(0.1)
        manager.stop()

        assert manager._thread is None
        assert manager._pool is None
        mock_client.close.assert_called_once()

    def test_stop_without_start_is_safe(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        manager.stop()


class TestAsyncLoopManagerGetMiningJob:
    def test_get_mining_job_returns_current_job(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)

        with pytest.raises(RuntimeError, match="Mining job is not initialized"):
            manager.get_mining_job()

        mock_job = Mock(spec=MiningJob)
        manager._mining_job = mock_job

        assert manager.get_mining_job() is mock_job

    def test_get_mining_job_concurrent_access_no_crash(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        results = []

        def reader():
            for _ in range(100):
                results.append(manager.get_mining_job())
                time.sleep(0.001)

        def writer():
            for _ in range(50):
                manager._mining_job = Mock(spec=MiningJob)
                time.sleep(0.002)

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()
        reader_thread.join()
        writer_thread.join()

        assert len(results) == 100


class TestAsyncLoopManagerHandleSubmitBlock:
    def test_handle_submit_block_no_loop_raises(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)

        with pytest.raises(AssertionError, match="Async loop is not started"):
            manager.handle_submit_block(Mock(spec=OpenedBlockInfo), Mock(spec=MiningJob))

    def test_handle_submit_block_no_pool_raises(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)
        manager._loop = Mock()

        with pytest.raises(AssertionError, match="Thread Pool Executor is not initialized"):
            manager.handle_submit_block(Mock(spec=OpenedBlockInfo), Mock(spec=MiningJob))


class TestAsyncLoopManagerScheduleStatusCheck:
    def test_schedule_status_check_queues_item(self, miner_config):
        settings = MinerSettings()
        settings.enable_async_cuda_event_processing = True

        manager = AsyncLoopManager(miner_config, settings)
        mock_loop = Mock()
        manager._loop = mock_loop

        mock_cuda_event = Mock(spec=torch.cuda.Event)
        mock_callback = Mock()

        manager.schedule_status_check(mock_cuda_event, mock_callback)

        mock_loop.call_soon_threadsafe.assert_called_once()
        args = mock_loop.call_soon_threadsafe.call_args[0]
        assert args[0] == manager._cuda_event_queue.put_nowait
        assert isinstance(args[1], CudaEventQueueItem)
        assert args[1].cuda_event is mock_cuda_event
        assert args[1].callback is mock_callback

    def test_schedule_status_check_disabled_warns(self, miner_config):
        settings = MinerSettings()
        settings.enable_async_cuda_event_processing = False

        manager = AsyncLoopManager(miner_config, settings)

        with pytest.warns(UserWarning, match="Async CUDA event processing disabled"):
            manager.schedule_status_check(Mock(spec=torch.cuda.Event), Mock())


class TestAsyncLoopManagerMiningJobCallbacks:
    def test_register_multiple_callbacks(self, miner_config):
        manager = AsyncLoopManager(miner_config, None)

        callback1 = Mock()
        callback2 = Mock()

        manager.register_mining_job_changed_callback(callback1)
        manager.register_mining_job_changed_callback(callback2)

        assert len(manager._mining_job_changed_callbacks) == 2
        assert callback1 in manager._mining_job_changed_callbacks
        assert callback2 in manager._mining_job_changed_callbacks
