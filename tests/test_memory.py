import pytest
from unittest.mock import MagicMock, patch
from swalot import MemoryProtector

# Mocking GPUtil's getGPUs method to return a list with a mock that has a memoryFree attribute
@pytest.fixture
def mock_gpu():
    with patch('GPUtil.getGPUs') as mock_getGPUs:
        mock_gpu = MagicMock()
        mock_gpu.memoryFree = 1024  # Example: 1024 MB free memory
        mock_getGPUs.return_value = [mock_gpu]
        yield mock_getGPUs

# Mocking torch and numpy to prevent actual GPU calls
@pytest.fixture
def mock_torch_numpy():
    with patch('torch.from_numpy') as mock_from_numpy:
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        with patch('numpy.empty') as mock_empty:
            mock_empty.return_value = MagicMock()

            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                yield mock_from_numpy, mock_empty, mock_empty_cache

def test_initial_protect_called(mock_gpu, mock_torch_numpy):
    # Test that during initialization, protect is called and setup correctly
    mock_from_numpy, mock_empty, mock_empty_cache = mock_torch_numpy

    protector = MemoryProtector(remain=256, device=0)

    # Check if protect was indeed called and if the memory was reserved correctly
    assert protector.protecting_mb == 768  # 1024 - 256

def test_free_memory(mock_gpu, mock_torch_numpy):
    mock_from_numpy, mock_empty, mock_empty_cache = mock_torch_numpy

    protector = MemoryProtector(remain=256, device=0)
    initial_tensor = protector.reserve_tensor
    protector.free_memory()

    # After freeing, reserve_tensor should be None
    assert protector.reserve_tensor is None

def test_restore_functionality(mock_gpu, mock_torch_numpy):
    mock_from_numpy, mock_empty, mock_empty_cache = mock_torch_numpy

    protector = MemoryProtector(remain=256, device=0)
    protector.restore()

    # Check if protect was called again and if the memory was reserved correctly
    assert protector.protecting_mb == 768  # 1024 - 256

if __name__ == "__main__":
    pytest.main()