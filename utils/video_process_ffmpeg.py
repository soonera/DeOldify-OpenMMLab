import ffmpeg
import os
from pathlib import Path
import re
from fastprogress.fastprogress import progress_bar
import shutil
from apis.colorization_inference import colorization_inference


class VideoColorizer:
    def __init__(self, model, workfolder):
        self.model = model
        # workfolder = Path('./video')
        workfolder = Path(workfolder)
        self.source_folder = workfolder / "source"
        self.bwframes_root = workfolder / "bwframes"
        self.audio_root = workfolder / "audio"
        self.colorframes_root = workfolder / "colorframes"
        self.result_folder = workfolder / "result"

    def _purge_images(self, dir):
        for f in os.listdir(dir):
            if re.search('.*?\.jpg', f):
                os.remove(os.path.join(dir, f))

    def _get_fps(self, source_path: Path) -> str:
        probe = ffmpeg.probe(str(source_path))
        stream_data = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None,
        )
        return stream_data['avg_frame_rate']

    def _extract_raw_frames(self, source_path: Path):
        bwframes_folder = self.bwframes_root / (source_path.stem)
        bwframe_path_template = str(bwframes_folder / '%5d.jpg')
        bwframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(bwframes_folder)
        ffmpeg.input(str(source_path)).output(
            str(bwframe_path_template), format='image2', vcodec='mjpeg', qscale=0
        ).run(capture_stdout=True)

    def _colorize_raw_frames(
        self, source_path: Path, render_factor: int = None, post_process: bool = True,
        watermarked: bool = True,
    ):
        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(colorframes_folder)
        bwframes_folder = self.bwframes_root / (source_path.stem)

        for img in progress_bar(os.listdir(str(bwframes_folder))):
            img_path = bwframes_folder / img

            if os.path.isfile(str(img_path)):
                # color_image = self.vis.get_transformed_image(
                #     str(img_path), render_factor=render_factor, post_process=post_process,watermarked=watermarked
                # )
                color_image = colorization_inference(self.model, img_path)
                color_image.save(str(colorframes_folder / img))

    def _build_video(self, source_path: Path) -> Path:
        colorized_path = self.result_folder / (
            source_path.name.replace('.mp4', '_no_audio.mp4')
        )
        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_path_template = str(colorframes_folder / '%5d.jpg')
        colorized_path.parent.mkdir(parents=True, exist_ok=True)
        if colorized_path.exists():
            colorized_path.unlink()
        fps = self._get_fps(source_path)

        ffmpeg.input(
            str(colorframes_path_template),
            format='image2',
            vcodec='mjpeg',
            framerate=fps,
        ).output(str(colorized_path), crf=17, vcodec='libx264').run(capture_stdout=True)

        result_path = self.result_folder / source_path.name
        if result_path.exists():
            result_path.unlink()
        # making copy of non-audio version in case adding back audio doesn't apply or fails.
        shutil.copyfile(str(colorized_path), str(result_path))

        # adding back sound here
        audio_file = Path(str(source_path).replace('.mp4', '.aac'))
        if audio_file.exists():
            audio_file.unlink()

        os.system(
            'ffmpeg -y -i "'
            + str(source_path)
            + '" -vn -acodec copy "'
            + str(audio_file)
            + '"'
        )

        if audio_file.exists:
            os.system(
                'ffmpeg -y -i "'
                + str(colorized_path)
                + '" -i "'
                + str(audio_file)
                + '" -shortest -c:v copy -c:a aac -b:a 256k "'
                + str(result_path)
                + '"'
            )
        print('Video created here: ' + str(result_path))
        return result_path

    def colorize_from_file_name(
        self, file_name: str, render_factor: int = None,  watermarked: bool = True, post_process: bool = True,
    ) -> Path:
        source_path = self.source_folder / file_name
        return self._colorize_from_path(
            source_path, render_factor=render_factor,  post_process=post_process,watermarked=watermarked
        )

    def _colorize_from_path(
        self, source_path: Path, render_factor: int = None,  watermarked: bool = True, post_process: bool = True
    ) -> Path:
        if not source_path.exists():
            raise Exception(
                'Video at path specfied, ' + str(source_path) + ' could not be found.'
            )
        self._extract_raw_frames(source_path)
        self._colorize_raw_frames(
            source_path, render_factor=render_factor,post_process=post_process,watermarked=watermarked
        )
        return self._build_video(source_path)