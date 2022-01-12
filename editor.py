import cv2
import numpy as np
from sys import argv, stdin
from os import remove
from shutil import which
from platform import system
from subprocess import call
from tabulate import tabulate
from utilities import Picture, Video

DEFAULT_SETTINGS = {
    'filename':             'out.mp4',
    'cover_filename':       'cover.png',
    'logo_filename':        'logo_black.png',
    'music_filename':       'audio.mp3',
    'seconds':              120,
    'format_size':          (1024, 640),
    'fps':                  60,
    'rpm':                  15,
    'logo_tolerance':       20,
    'min_padding':          50,
    'thickness':            2,
    'small_radius':         0.08,
    'background_BGR':       (78, 155, 0),
    'disco_BGR':            (66, 245, 173),
    'logo':                 False,
    'logo_black_bg':        True,
    'clockwise_rotation':   True,
    'music':                False,
    'custom_cover':         False
}


class Editor:

    def __init__(self, configuration_file='configuration.txt', ffmpeg='ffmpeg'):

        self.settings = DEFAULT_SETTINGS
        self.set_settings(configuration_file)
        self.ffmpeg = ffmpeg
        self.picture = None
        self.video = None

        self.framesRequired = 0
        self.side = 0
        self.fpr = 0
        self.step = 0
        self.loops = 0
        self.width = 0
        self.big_radius = 0
        self.center_coordinates = (0, 0)
        self.calculate_params()

    def __str__(self):
        string = "Editor\nOutput file:\t{}".format(self.settings['filename'])
        return string

    def __call__(self, *args, **kwargs):
        self.picture = Picture(picture=self.create_cover())
        self.picture.crop(center_coordinates=self.center_coordinates, width=self.width)
        if self.settings['custom_cover']:
            self.suspend()
        cover = self.picture.get_picture()
        frames = self.picture.rotate(step=self.step, fix=False, bgr_fix=self.settings['background_BGR'])
        self.video = Video(
            frames,
            loops=self.loops,
            fps=self.settings['fps'],
            background_bgr=self.settings['background_BGR'],
            format_size=self.settings['format_size'],
            clockwise_rotation=self.settings['clockwise_rotation'],
            music_filename=self.settings['music_filename'],
            filename=self.settings['filename'],
            music=self.settings['music'],
            ffmpeg=self.ffmpeg
        )
        print("{}\n\nClick on picture and press c to confirm".format(self.video))
        self.video.insert_frame(cover)
        if self.video.get_seconds() == 0:
            print("Video length == 0. Exit.")
            exit(0)
        cover = self.video.add_padding(img=cover)
        key = Picture.show(img=cover, label="Preview")
        if key == ord('k'):
            # To avoid a bug (tested on WSL)
            print("OpenCV key pressed bug found\nType 'c' in the terminal and press enter (instead that on the picture)\n$", end="")
            key = ord(stdin.read(1))
        if key != ord('c'):
            print("Not ready ({} != 'c')".format(chr(key)))
            exit(0)
        Picture.save_img(cover, filename=self.settings['cover_filename'].strip('"').strip("'"))
        self.video()

    def set_settings(self, file):
        configuration = Editor.get_configuration(file)
        if len(configuration) == 0:
            return
        valid_keys = self.settings.keys()
        for key, value in configuration:
            if key in valid_keys:
                my_type = type(self.settings[key])
                if my_type == str:
                    self.settings[key] = value
                elif my_type == bool:
                    if value.lower() == 'true':
                        self.settings[key] = True
                    elif value.lower() == 'false':
                        self.settings[key] = False
                elif my_type == int:
                    self.settings[key] = int(value)
                elif my_type == float:
                    self.settings[key] = float(value)
                elif my_type == tuple:
                    value = value.lstrip('(').rstrip(')')
                    values = [int(val) for val in value.split(',')]
                    values = tuple(values)
                    self.settings[key] = values

    def calculate_params(self):
        self.side = min(self.settings['format_size']) - self.settings['min_padding'] * 2
        self.framesRequired = self.settings['seconds'] * self.settings['fps']
        self.fpr = self.settings['fps'] * 60 / self.settings['rpm']     # Frame per round [f/r]
        self.step = 360 / self.fpr                                      # The angle at each step
        self.loops = self.framesRequired/self.fpr
        self.center_coordinates = (int(self.side/2), int(self.side/2))
        self.big_radius = int(self.side / 2 - self.settings['thickness'])
        # Calculate the number of pixels of the small radius
        if 0 < self.settings['small_radius'] <= 1:
            self.settings['small_radius'] = int(self.big_radius * self.settings['small_radius'])
        else:
            self.settings['small_radius'] = int(self.big_radius / self.settings['small_radius'])
        self.width = self.big_radius + self.settings['thickness']
        # Loop must be integer
        self.loops = int(self.loops)

    def create_cover(self):
        background = np.full((self.side, self.side, 3), self.settings['background_BGR'], dtype=np.uint8)
        result = cv2.circle(background, self.center_coordinates, self.big_radius, (0, 0, 0), self.settings['thickness'])
        result = cv2.circle(result, self.center_coordinates, self.big_radius - 1, self.settings['disco_BGR'], -1)
        if self.settings['logo']:
            logo = Picture(self.settings['logo_filename'])
            result = Picture(picture=result)
            result.add_logo(logo,
                            black_background=self.settings['logo_black_bg'],
                            disco_bgr=self.settings['disco_BGR'],
                            shape=(self.side, self.side),
                            tolerance=self.settings['logo_tolerance']
                            )
            result = result.get_picture()
        result = cv2.circle(result, self.center_coordinates, self.settings['small_radius'], (0, 0, 0), self.settings['thickness'])
        result = cv2.circle(result, self.center_coordinates, self.settings['small_radius'] - 1, self.settings['background_BGR'], -1)
        return result

    def suspend(self):
        # This function allow to user to make some manually modifications and then import the new image
        print("Saving the image of cover > tmp_cover.png\nWhen you're ready press Enter (don't change the filename)")
        Picture.save_img(self.picture.get_picture(), filename='tmp_cover.png')
        input("Ready?\n")
        self.picture = Picture(filename='tmp_cover.png')
        try:
            remove('tmp_cover.png')
        except Exception as e:
            print("Can't delete tmp file\nException:\n{}".format(e))
            pass

    @staticmethod
    def get_configuration(file):
        try:
            with open(file, 'r') as fp:
                lines = fp.readlines()
        except IOError:
            print("Error opening the configuration file ({}). Abort.".format(file))
            exit(0)
        result = []
        for line in lines:
            if line.startswith('#') or line.startswith('\n'):
                continue
            words = line.split('=')
            key = Editor.clean_settings_string(words[0])
            value = Editor.clean_settings_string(words[-1])
            if key == '' or value == '':
                continue
            result.append((key, value))
        return result

    @staticmethod
    def clean_settings_string(string):
        if not string.startswith((' ', '"', "'")) and not string.endswith((' ', '"', "'")):
            return string
        return Editor.clean_settings_string(string.strip().strip("'").strip('"'))


def get_ffmpeg():
    path = None
    if len(argv) > 2:
        path = str(argv[2]).replace("\\", "/")
    if check_ffmpeg(path=path):
        print("ffmpeg ready\n")
    else:
        print("Please before install ffmpeg\nOn Linux:\tapt-get install ffmpeg -y")
        exit(0)
    if path is not None:
        return path
    return 'ffmpeg'


def check_ffmpeg(path=None):
    name = 'ffmpeg'
    if system() == 'Windows' and path is None:
        return not call(['where', name])
    if path is not None:
        name = str(path)
        print("Trying the given binary of ffmpeg: {}".format(name))
    return which(name) is not None


def check_argv():
    if len(argv) > 1:
        return
    print("[*] Usage:\tpython3 {} conf_filename.txt OPTIONAL_FFMPEG_PATH".format(argv[0]))
    print("\nAll possible settings with an example:", end="\n\n")
    all_settings = [[key, DEFAULT_SETTINGS[key]] for key in DEFAULT_SETTINGS]
    print(tabulate(all_settings, headers=['Key', 'Example'], tablefmt='orgtbl'))
    exit(1)


def main():
    check_argv()
    ffmpeg = get_ffmpeg()
    my_editor = Editor(argv[1], ffmpeg=ffmpeg)
    my_editor()


if __name__ == '__main__':
    main()
    exit(0)
