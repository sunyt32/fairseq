# Yaoyao, 2025

from typing import Optional
import os
import tempfile
import shutil
from multiprocessing import Pool
import collections.abc
import functools
import abc
import os
import pathlib
import shutil
import sys
import urllib.parse
from tempfile import mkdtemp
from typing import Any, Callable, Optional, Sequence, TypeVar, Union, cast, overload
import tqdm
from time import sleep
import random

from azure.storage.blob import BlobServiceClient
import os

DEFAULT_TIMEOUT = 60

UPLOADERS = {
    's3': 'S3Uploader',
    'gs': 'GCSUploader',
    'oci': 'OCIUploader',
    'hf': 'HFUploader',
    'azure': 'AzureUploader',
    'azure-dl': 'AzureDataLakeUploader',
    'dbfs:/Volumes': 'DatabricksUnityCatalogUploader',
    'dbfs': 'DBFSUploader',
    'alipan': 'AlipanUploader',
    '': 'LocalUploader',
}


TCallable = TypeVar('TCallable', bound=Callable)

# error: Type "(TCallable@retry) -> TCallable@retry" cannot be assigned to type
# "(func: Never) -> Never"


def retry(  # type: ignore
    exc_class: Union[TCallable, type[Exception],
                     Sequence[type[Exception]]] = Exception,
    clean_up_fn: Optional[Callable[[], None]] = None,
    num_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_jitter: float = 0.5,
):
    """Decorator to retry a function with backoff and jitter.

    Attempts are spaced out with
    ``initial_backoff * 2**num_attempts + random.random() * max_jitter`` seconds.

    Example:
        .. testcode::

            from streaming.base.util import retry

            num_tries = 0

            def clean_up():
                # Do clean up stuff here
                print("cleaning up")

            @retry(RuntimeError, clean_up_fn=clean_up, num_attempts=3, initial_backoff=0.1)
            def flaky_function():
                global num_tries
                if num_tries < 2:
                    num_tries += 1
                    raise RuntimeError("Called too soon!")
                return "Third time's a charm."

            print(flaky_function())

    .. testoutput::

        cleaning up
        cleaning up
        Third time's a charm.

    Args:
        exc_class (Type[Exception] | Sequence[Type[Exception]]], optional): The exception class or
            classes to retry. Defaults to Exception.
        clean_up_fn (Callable[[], None], optional): A function to call after each failed attempt
            before retrying. Defaults to None.
        num_attempts (int, optional): The total number of attempts to make. Defaults to 3.
        initial_backoff (float, optional): The initial backoff, in seconds. Defaults to 1.0.
        max_jitter (float, optional): The maximum amount of random jitter to add. Defaults to 0.5.

            Increasing the ``max_jitter`` can help prevent overloading a resource when multiple
            processes in parallel are calling the same underlying function.
    """
    if num_attempts < 1:
        raise ValueError('num_attempts must be at-least 1')

    def wrapped_func(func: TCallable) -> TCallable:

        @functools.wraps(func)
        def new_func(*args: Any, **kwargs: Any):
            i = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exc_class as e:
                    if clean_up_fn is not None:
                        clean_up_fn()

                    if i + 1 == num_attempts:
                        print(
                            f'Attempt {i + 1}/{num_attempts} failed with: {e}')
                        raise e
                    else:
                        sleep(initial_backoff * 2**i +
                              random.random() * max_jitter)
                        print(
                            f'Attempt {i + 1}/{num_attempts} failed with: {e}')
                        i += 1

        return cast(TCallable, new_func)

    if not isinstance(exc_class, collections.abc.Sequence) and not (isinstance(
            exc_class, type) and issubclass(exc_class, Exception)):
        # Using the decorator without (), like @retry_with_backoff
        func = cast(TCallable, exc_class)
        exc_class = Exception

        return wrapped_func(func)

    return wrapped_func


class CloudUploader:
    """Upload local files to a cloud storage."""

    @classmethod
    def get(cls,
            out: Union[str, tuple[str, str]],
            keep_local: bool = False,
            progress_bar: bool = False,
            retry: int = 2,
            exist_ok: bool = False) -> Any:
        """Instantiate a cloud provider uploader or a local uploader based on remote path.

        Args:
            out (str | Tuple[str, str]): Output dataset directory to save shard files.

                1. If ``out`` is a local directory, shard files are saved locally.
                2. If ``out`` is a remote directory, a local temporary directory is created to
                   cache the shard files and then the shard files are uploaded to a remote
                   location. At the end, the temp directory is deleted once shards are uploaded.
                3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
                   the `local_dir` and also uploaded to a remote location.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            retry (int): Number of times to retry uploading a file. Defaults to ``2``.
            exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
                exists and has contents. Defaults to ``False``.

        Returns:
            CloudUploader: An instance of sub-class.
        """
        cls._validate(cls, out)
        obj = urllib.parse.urlparse(out) if isinstance(
            out, str) else urllib.parse.urlparse(out[1])
        provider_prefix = obj.scheme
        if obj.scheme == 'dbfs':
            path = pathlib.Path(out) if isinstance(
                out, str) else pathlib.Path(out[1])
            prefix = os.path.join(path.parts[0], path.parts[1])
            if prefix == 'dbfs:/Volumes':
                provider_prefix = prefix
        return getattr(sys.modules[__name__],
                       UPLOADERS[provider_prefix])(out, keep_local, progress_bar, retry, exist_ok)

    def _validate(self, out: Union[str, tuple[str, str]]) -> None:
        """Validate the `out` argument.

        Args:
            out (str | Tuple[str, str]): Output dataset directory to save shard files.

                1. If ``out`` is a local directory, shard files are saved locally.
                2. If ``out`` is a remote directory, a local temporary directory is created to
                   cache the shard files and then the shard files are uploaded to a remote
                   location. At the end, the temp directory is deleted once shards are uploaded.
                3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
                   the `local_dir` and also uploaded to a remote location.

        Raises:
            ValueError: Invalid number of `out` argument.
            ValueError: Invalid Cloud provider prefix.
        """
        if isinstance(out, str):
            obj = urllib.parse.urlparse(out)
        else:
            if len(out) != 2:
                raise ValueError(f'Invalid `out` argument. It is either a string of ' +
                                 f'local/remote directory or a list of two strings with ' +
                                 f'[local, remote].')
            obj = urllib.parse.urlparse(out[1])
        if obj.scheme not in UPLOADERS:
            raise ValueError(f'Invalid Cloud provider prefix: {obj.scheme}.')

    def __init__(self,
                 out: Union[str, tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        """Initialize and validate local and remote path.

        Args:
            out (str | Tuple[str, str]): Output dataset directory to save shard files.

                1. If ``out`` is a local directory, shard files are saved locally.
                2. If ``out`` is a remote directory, a local temporary directory is created to
                   cache the shard files and then the shard files are uploaded to a remote
                   location. At the end, the temp directory is deleted once shards are uploaded.
                3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
                   the `local_dir` and also uploaded to a remote location.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            retry (int): Number of times to retry uploading a file. Defaults to ``2``.
            exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
                exists and has contents. Defaults to ``False``.

        Raises:
            FileExistsError: Local directory must be empty.
        """
        self._validate(out)
        self.keep_local = keep_local
        self.progress_bar = progress_bar
        self.retry = retry

        if isinstance(out, str):
            # It is a remote directory
            if urllib.parse.urlparse(out).scheme != '':
                self.local = mkdtemp()
                self.remote = out
            # It is a local directory
            else:
                self.local = out
                self.remote = None
        else:
            self.local = out[0]
            self.remote = out[1]

        if os.path.exists(self.local) and len(os.listdir(self.local)) != 0:
            if not exist_ok:
                raise FileExistsError(f'Directory is not empty: {self.local}')
            else:
                print(
                    f'Directory {self.local} exists and not empty. But continue to mkdir since exist_ok is set to be True.'
                )

        os.makedirs(self.local, exist_ok=True)

    def upload_file(self, filename: str):
        """Upload file from local instance to remote instance.

        Args:
            filename (str): File to upload.

        Raises:
            NotImplementedError: Override this method in your sub-class.
        """
        raise NotImplementedError(
            f'{type(self).__name__}.upload_file is not implemented')

    def list_objects(self, prefix: Optional[str] = None) -> Optional[list[str]]:
        """List all objects in the object store with the given prefix.

        Args:
            prefix (Optional[str], optional): The prefix to search for. Defaults to ``None``.

        Returns:
            List[str]: A list of object names that match the prefix.
        """
        raise NotImplementedError(
            f'{type(self).__name__}.list_objects is not implemented')

    def clear_local(self, local: str):
        """Remove the local file if it is enabled.

        Args:
            local (str): A local file path.
        """
        if not self.keep_local and os.path.isfile(local):
            os.remove(local)


class AzureUploader(CloudUploader):
    """Upload file from local machine to Microsoft Azure bucket.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
               the `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
            a remote location. Default to ``False``.
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)

        # print("debug: init azure uploader", os.environ['AZURE_ACCOUNT_NAME'])

        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        # self.azure_service = BlobServiceClient(
        #     account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.blob.core.windows.net",
        #     credential=cred,
        # )
        self.blobname_to_containername_to_client = {}
        # self.check_bucket_exists(self.remote)  # pyright: ignore

    def _get_container_client(self, blob_name, container_name):
        if blob_name not in self.blobname_to_containername_to_client or container_name not in self.blobname_to_containername_to_client[blob_name]:
            if blob_name not in self.blobname_to_containername_to_client:
                self.blobname_to_containername_to_client[blob_name] = {}
            var_name = f"{blob_name.upper()}_{container_name.upper()}_SAS_TOKEN"
            if var_name not in os.environ:
                raise ValueError(f"Environment variable {var_name} not found")
            sas_token = os.environ.get(var_name)
            account_url = f"https://{blob_name}.blob.core.windows.net" + sas_token
            service_client = BlobServiceClient(account_url)
            container_client = service_client.get_container_client(
                container=container_name)
            self.blobname_to_containername_to_client[blob_name][container_name] = container_client
            print(
                f"AzureUploader read env variable {var_name} successfully to get client for {blob_name} {container_name}")
        return self.blobname_to_containername_to_client[blob_name][container_name]

    def upload_file(self, filename: str):
        """Upload file from local instance to Microsoft Azure bucket.

        Args:
            filename (str): File to upload.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
            local_filename = os.path.join(self.local, filename)
            local_filename = local_filename.replace('\\', '/')
            remote_filename = os.path.join(
                self.remote, filename)  # pyright: ignore
            remote_filename = remote_filename.replace('\\', '/')
            # obj = urllib.parse.urlparse(remote_filename)
            # logger.debug(f'Uploading to {remote_filename}')
            file_size = os.stat(local_filename).st_size
            # print("debug: file_name", filename)
            # print("debug: local_filename", local_filename)
            # print("debug: remote_filename", remote_filename)
            # print("debug: obj.netloc", obj.netloc)
            # print("debug: obj.path.lstrip('/'), ", obj.path.lstrip('/'))
            stripped_str = remote_filename[len("azure://"):]
            parts = stripped_str.split('/', 2)
            blob_name = parts[0]
            container_name = parts[1]
            relative_path = parts[2]
            # print("debug: blob_name", blob_name)
            # print("debug: container_name", container_name)
            # print("debug: relative_path", relative_path)
            # if blob_name not in self.blobname_to_azure_service:
            #     self.blobname_to_azure_service[blob_name] = self.azure_service.get_blob_service_client(container=blob_name)

            # container_client = self.azure_service.get_container_client(container=obj.netloc)
            container_client = self._get_container_client(
                blob_name, container_name)

            with tqdm.tqdm(total=file_size,
                           unit='B',
                           unit_scale=True,
                           desc=f'Uploading to {remote_filename}',
                           disable=(not self.progress_bar)) as pbar:
                with open(local_filename, 'rb') as data:
                    container_client.upload_blob(
                        name=relative_path,
                        data=data,
                        progress_hook=lambda bytes_transferred, _: pbar.update(
                            bytes_transferred),
                        overwrite=True)
            self.clear_local(local=local_filename)

        _upload_file()

    def upload_file(self, local_path: str, remote_path: str, keep_local: bool = True):
        """Similar to the upload_file function above, but allows specifying local_path and remote_path, 
        and choosing whether to keep the local file (default is to keep, btw the function above does not keep the local file).

        Args:
            filename (str): File to upload.
        """
        @retry(num_attempts=self.retry)
        def _upload_file():
            local_filename = local_path.replace('\\', '/')
            remote_filename = remote_path.replace('\\', '/')
            # logger.debug(f'Uploading to {remote_filename}')
            file_size = os.stat(local_filename).st_size
            # print("debug: file_name", filename)
            # print("debug: local_filename", local_filename)
            # print("debug: remote_filename", remote_filename)
            # print("debug: obj.netloc", obj.netloc)
            # print("debug: obj.path.lstrip('/'), ", obj.path.lstrip('/'))
            stripped_str = remote_filename[len("azure://"):]
            parts = stripped_str.split('/', 2)
            blob_name = parts[0]
            container_name = parts[1]
            relative_path = parts[2]
            # print("debug: blob_name", blob_name)
            # print("debug: container_name", container_name)
            # print("debug: relative_path", relative_path)
            # if blob_name not in self.blobname_to_azure_service:
            #     self.blobname_to_azure_service[blob_name] = self.azure_service.get_blob_service_client(container=blob_name)

            # container_client = self.azure_service.get_container_client(container=obj.netloc)
            container_client = self._get_container_client(
                blob_name, container_name)

            with tqdm.tqdm(total=file_size,
                           unit='B',
                           unit_scale=True,
                           desc=f'Uploading to {remote_filename}',
                           disable=(not self.progress_bar)) as pbar:
                with open(local_filename, 'rb') as data:
                    container_client.upload_blob(
                        name=relative_path,
                        data=data,
                        progress_hook=lambda bytes_transferred, _: pbar.update(
                            bytes_transferred),
                        overwrite=True)
            if not keep_local:
                self.clear_local(local=local_filename)

        _upload_file()

    def check_bucket_exists(self, remote: str):
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): azure bucket path.

        Raises:
            error: Bucket does not exist.
        """
        bucket_name = urllib.parse.urlparse(remote).netloc
        if self.azure_service.get_container_client(container=bucket_name).exists() is False:
            raise FileNotFoundError(
                f'Either bucket `{bucket_name}` does not exist! ' +
                f'or check the bucket permission.',)


class CloudDownloader(abc.ABC):
    """Download files from remote storage to a local filesystem."""

    @classmethod
    def direct_download(cls,
                        remote: Optional[str],
                        local: str,
                        timeout: float = DEFAULT_TIMEOUT) -> None:
        """Directly download a file from remote storage to local filesystem.

        Args:
            remote (str | None): Remote path.
            local (str): Local path.
            timeout (float): How long to wait for file to download before raising an exception.
                Defaults to `60 seconds.

        Raises:
            ValueError: If the remote path is not provided while local does not exist or remote
                path is not supported.
        """
        downloader = cls.get(remote)
        downloader.download(remote, local, timeout)
        downloader.clean_up()

    def download(self,
                 remote: Optional[str],
                 local: str,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        """Download a file from remote storage to local filesystem.

        Args:
            remote (str | None): Remote path.
            local (str): Local path.
            timeout (float): How long to wait for file to download before raising an exception.
                Defaults to `60 seconds.

        Raises:
            ValueError: If the remote path does not contain the expected prefix or remote is
                not provided while local does not exist.
        """
        if os.path.exists(local):
            return

        if not remote:
            raise ValueError(
                'In the absence of local dataset, path to remote dataset must be provided')

        if sys.platform == 'win32':
            remote = pathlib.PureWindowsPath(remote).as_posix()
            local = pathlib.PureWindowsPath(local).as_posix()

        local_dir = os.path.dirname(local)
        os.makedirs(local_dir, exist_ok=True)

        self._validate_remote_path(remote)
        self._download_file_impl(remote, local, timeout)

    @staticmethod
    @abc.abstractmethod
    def _client_identifier() -> str:
        """Return the client identifier for the downloader.

        Returns:
            str: Identifier of the client downloader. Can be a schema or prefix of the remote path.
        """

    @abc.abstractmethod
    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        raise NotImplementedError

    @abc.abstractmethod
    def _download_file_impl(self, remote: str, local: str, timeout: float) -> None:
        """Implementation of the download function for a file.

        Args:
            remote (str): Remote path.
            local (str): Local path.
            timeout (float): How long to wait for file to download before raising an exception.
        """
        raise NotImplementedError

    def _validate_remote_path(self, remote: str) -> None:
        """Validate the remote path.

        Args:
            remote (str): Remote path.

        Raises:
            ValueError: If the remote path does not contain the expected prefix.
        """
        url_scheme = urllib.parse.urlparse(remote).scheme

        if url_scheme != self._client_identifier():
            raise ValueError(
                f'Expected remote path to start with url scheme of {url_scheme}, got {remote}.')


class AzureDownloader(CloudDownloader):
    """Download files from Azure to local filesystem."""

    def __init__(self):
        """Initialize the Azure downloader."""
        super().__init__()

        self.blobname_to_containername_to_client = {}

    def _get_container_client(self, blob_name, container_name):
        if blob_name not in self.blobname_to_containername_to_client or container_name not in self.blobname_to_containername_to_client[blob_name]:
            if blob_name not in self.blobname_to_containername_to_client:
                self.blobname_to_containername_to_client[blob_name] = {}
            var_name = f"{blob_name.upper()}_{container_name.upper()}_SAS_TOKEN"
            if var_name not in os.environ:
                raise ValueError(f"Environment variable {var_name} not found")
            sas_token = os.environ.get(var_name)
            account_url = f"https://{blob_name}.blob.core.windows.net" + sas_token
            service_client = BlobServiceClient(account_url)
            container_client = service_client.get_container_client(
                container=container_name)
            self.blobname_to_containername_to_client[blob_name][container_name] = container_client
            print(
                f"AzureDownloader read env variable {var_name} successfully to get client for {blob_name} {container_name}")
        return self.blobname_to_containername_to_client[blob_name][container_name]

    @staticmethod
    def _client_identifier() -> str:
        """Return the client identifier for the downloader.

        Returns:
            str: returns azure.
        """
        return 'azure'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        return

    def _download_file_impl(self, remote: str, local: str, timeout: float) -> None:
        """Implementation of the download function for a file."""

        stripped_str = remote[len("azure://"):]
        parts = stripped_str.split('/', 2)
        blob_name = parts[0]
        container_name = parts[1]
        relative_path = parts[2]
        # print("debug: blob_name", blob_name)
        # print("debug: container_name", container_name)
        # print("debug: relative_path", relative_path)
        container_client = self._get_container_client(
            blob_name, container_name)
        blob_client = container_client.get_blob_client(blob=relative_path)
        local_tmp = local + '.tmp'
        with open(local_tmp, 'wb') as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
        os.rename(local_tmp, local)


# our wrapper starts here
class AzureStorage:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.uploader = CloudUploader.get(
            out="azure://it/does/not/matter/what/this/path/is")
        self.downloader = AzureDownloader()
        print(f"our Awesome AzureStorage initialized.")

    def upload(self, local_path: str, remote_path: str):
        if self.debug:
            print(f"Uploading {local_path} to {remote_path}")
        self.uploader.upload_file(local_path, remote_path)

    def download(self, remote_path: str, local_path: str):
        if self.debug:
            print(f"Downloading {remote_path} to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.downloader._download_file_impl(
            remote_path, local_path, timeout=DEFAULT_TIMEOUT)

    def get_index_of_azure(self, remote_data_dir):
        stripped_str = remote_data_dir[len("azure://"):]
        parts = stripped_str.split('/', 2)
        blob_name = parts[0]
        container_name = parts[1]
        relative_path = parts[2]
        container_client = self.downloader._get_container_client(
            blob_name, container_name)

        # Avoid listing contents of other folders. For example, if you want to list the contents of the tmp folder, and there is a tmp2 folder in the same directory, tmp2 will also be listed because the name_starts_with parameter of list_blobs is designed this way. Adding a slash ensures that only files in the tmp folder are listed.
        blob_start_name = relative_path
        if not relative_path.endswith("/"):
            blob_start_name = relative_path + "/"

        lines = []
        for blob in container_client.list_blobs(name_starts_with=blob_start_name):
            if blob['size'] == 0:
                continue
            remote_file_name = blob['name'].replace(
                relative_path, remote_data_dir)
            lines.append(remote_file_name)
        return lines


_storage_instance: Optional[AzureStorage] = None


def _get_storage() -> AzureStorage:
    """
    Get the global singleton instance of AzureStorage.
    If it has not been created yet, create and cache it here.
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = AzureStorage(debug=False)
    return _storage_instance


def is_azure_path(path: str) -> bool:
    """
    Check if the given path is an Azure path.
    """
    return path.startswith("azure://")


def copyfile(src: str, dst: str):
    """
    Copy src to dst. Supports:
    1. Local -> Local (directly shutil.copyfile)
    2. Local -> Azure (using AzureUploader)
    3. Azure -> Local (using AzureDownloader)
    4. Azure -> Azure (download to a temporary file, then upload)

    Args:
        src (str): Source path
        dst (str): Destination path
    """
    # print(src, dst)
    src_is_azure = is_azure_path(src)
    dst_is_azure = is_azure_path(dst)
    if not src_is_azure and not dst_is_azure:
        # Case 1: 本地 -> 本地
        shutil.copyfile(src, dst)
    elif not src_is_azure and dst_is_azure:
        # Case 2: 本地 -> Azure
        _get_storage().upload(src, dst)
    elif src_is_azure and not dst_is_azure:
        # Case 3: Azure -> 本地
        _get_storage().download(src, dst)
    else:
        # Case 4: Azure -> Azure
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
        _get_storage().download(src, tmp_file_path)
        _get_storage().upload(tmp_file_path, dst)
        os.remove(tmp_file_path)


def copyfile_wrapper(line):
    src, dst = line
    copyfile(src, dst)


def copydir(src_dir: str, dst_dir: str, process_count: int = 1):
    """
    Copy src to dst. Supports:
    1. Local -> Local (directly shutil.copytree)
    2. Local -> Azure (using AzureUploader)
    3. Azure -> Local (using AzureDownloader)
    4. Azure -> Azure (download to a temporary file, then upload)

    Args:
        src_dir (str): Source directory
        dst_dir (str): Destination directory
    """
    src_is_azure = is_azure_path(src_dir)
    dst_is_azure = is_azure_path(dst_dir)
    if not src_is_azure and not dst_is_azure:
        # Case 1: 本地 -> 本地
        shutil.copytree(src_dir, dst_dir)
        return
    elif not src_is_azure and dst_is_azure:
        # Case 2: 本地 -> Azure
        lines = []
        for root, dirs, files in os.walk(src_dir):
            for name in files:
                filepath = os.path.join(root, name)
                lines.append((filepath, filepath.replace(src_dir, dst_dir)))
    else:
        # Case 3 和 Case 4: Azure -> 本地 和 Azure -> Azure
        if not dst_is_azure:
            os.makedirs(dst_dir, exist_ok=True)
        lines = _get_storage().get_index_of_azure(src_dir)
        lines = [(filepath, filepath.replace(src_dir, dst_dir))
                 for filepath in lines]

    # copy all files for case 2-4
    if process_count <= 1:
        for src, dst in lines:
            copyfile(src, dst)
    else:
        with Pool(processes=process_count) as p:
            with tqdm.tqdm(total=len(lines)) as pbar:
                for i, _ in enumerate(p.imap_unordered(copyfile_wrapper, lines)):
                    pbar.update()
