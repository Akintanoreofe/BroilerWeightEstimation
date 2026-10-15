"""Camera configuration data and construction helpers."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraConfig:
    """Paths and identifier associated with one camera.

    Attributes
    ----------
    camera_id : str
        Camera identifier, normalized to lowercase by the loader.
    video_path : pathlib.Path
        Input video path.
    video_save_path : pathlib.Path
        Path where the processed video is saved.
    """

    camera_id: str
    video_path: Path
    video_save_path: Path


class CameraConfigLoader:
    """Build camera configuration objects from path strings."""

    def get_camera_info(self, camera_id: str, video_path: str, video_save_path: str) -> CameraConfig:
        """Create a camera configuration and normalize its identifier.

        Parameters
        ----------
        camera_id : str
            Identifier for the camera.
        video_path : str
            Path to the source video.
        video_save_path : str
            Destination path for the processed video.

        Returns
        -------
        CameraConfig
            Configuration containing the lowercase identifier and paths.
        """
        return CameraConfig(
            camera_id=camera_id.lower(),
            video_path=Path(video_path),
            video_save_path=Path(video_save_path),
        )
