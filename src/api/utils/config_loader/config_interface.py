import abc

class ConfigReaderInterface(abc.ABC):

    def __init__(self):
        super().__init__()
        
    def read_config_from_file(self, file_path: str):
        """
        Reads configuration from a file.
        
        :param file_path: Path to the configuration file.
        :return: Parsed configuration data.
        """
        raise NotImplementedError("This method should be implement by subclasses.")