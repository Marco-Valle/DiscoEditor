import imutils
from tqdm import tqdm
from os import remove, rename
from os.path import isfile
from subprocess import call
from tabulate import tabulate
from time import sleep
import numpy as np
import cv2


class Picture:

    def __init__(self, filename='example.png', picture=None):
        self.filename = filename
        self.height = 0
        self.width = 0
        self.gray = 0
        self.hsv = 0
        if picture is not None:
            self.img = picture
            self.update()
        else:
            self.img = None
            self.import_img()

    def __repr__(self):
        return 'Picture()'

    def __str__(self):
        if self.img is None:
            return "Picture()"
        else:
            return "Picture(filename={}, shape={})".format(self.filename, (self.height, self.width))

    def get_shape(self):
        shape = (self.width, self.height)
        return shape

    def get_picture(self):
        return self.img

    def import_img(self, force_import=False):
        if self.img is None or force_import:
            self.img = cv2.imread(self.filename)
        if self.img is None:
            print("Picture not found ({}). Abort".format(self.filename))
            exit(1)
        self.update()

    def add_logo(self, logo, black_background=False, disco_bgr=(255, 255, 255), shape=(500, 500), tolerance=20):
        # Add logo in the center
        # Logo with black or white background is required
        logo_shape = logo.get_shape()
        if logo_shape == (0, 0):
            return
        ratio = logo_shape[0] / logo_shape[1]
        print("Current logo shape (x,y): {} (ratio: {:.2f})".format(logo_shape, ratio))
        if logo_shape[0] >= logo_shape[1]:
            logo_shape_x = int(shape[0] - shape[0]/20)
            logo_shape_y = int(logo_shape_x / ratio)
        else:
            logo_shape_y = int(shape[1] - shape[1]/20)
            logo_shape_x = int(logo_shape_y * ratio)
        new_logo_shape = (logo_shape_x, logo_shape_y)
        ratio = logo_shape_x / logo_shape_y
        print("Resizing logo, new shape: {} (ratio: {:.2f})".format(new_logo_shape, ratio), end="\n\n")
        logo.resize(new_logo_shape)
        logo_picture = logo.get_picture()
        if black_background:
            lower = np.array(0, dtype="uint8")
            upper = np.array(tolerance, dtype="uint8")
        else:
            lower = np.array(255 - tolerance, dtype="uint8")
            upper = np.array(255, dtype="uint8")
        # Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(logo_picture, lower, upper)
        # Replace the logo background with disco's color
        logo_picture[mask > 0] = disco_bgr
        y0 = int(self.img.shape[1]/2 - logo_shape_y/2)
        y1 = int(self.img.shape[1]/2 + logo_shape_y/2)
        x0 = int(self.img.shape[0]/2 - logo_shape_x/2)
        x1 = int(self.img.shape[0]/2 + logo_shape_x/2)
        self.img[y0:y1, x0:x1] = logo_picture

    def update(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        (self.height, self.width) = self.gray.shape

    def resize(self, new_shape):
        self.img = cv2.resize(self.img, new_shape)
        self.update()

    def crop(self, center_coordinates=None, width=None):
        img = self.img
        if center_coordinates and width:
            x = center_coordinates[0]-width
            y = center_coordinates[1]-width
            w = 2*width
            h = w
        else:
            (ret, thresh) = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY)
            # Create mask
            mask = np.zeros((self.height, self.width), np.uint8)
            edges = cv2.Canny(thresh, 100, 200)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1=50, param2=30, minRadius=0, maxRadius=0)
            for i in circles[0, :]:
                i[2] = i[2] + 4
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)
            # Apply Threshold
            (_, thresh) = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            # Find Contour
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (x, y, w, h) = cv2.boundingRect(contours[0][0])
        # Crop masked_data
        self.img = img[y:y + h, x:x + w]
        self.update()

    def rotate(self, minimum=0, maximum=360, step=15, fix=True, bgr_fix=None):
        results = []
        for angle in np.arange(minimum, maximum, step):
            # Fix the shape of picture (it's useful when you haven't square picture)
            if bgr_fix is not None:
                if fix:
                    result = Picture.rotate_bound(self.img, angle, bgr=bgr_fix)
                else:
                    result = Picture.rotate_imutils(self.img, angle, bgr=bgr_fix)
            else:
                if fix:
                    result = imutils.rotate_bound(self.img, angle)
                else:
                    result = imutils.rotate(self.img, angle)
            results.append(result)
        return results

    @staticmethod
    def save_img(picture, filename='img.png'):
        try:
            cv2.imwrite(filename, picture)
        except Exception as e:
            print("Can't save the file {}\nException:\n{}".format(filename, e))

    @staticmethod
    def show(img=None, img_array=None, label='Default'):
        key = '\0'
        if img_array is None:
            img_array = []
        if len(img_array) == 0 and img is not None:
            img_array = [img]
        for i in img_array:
            cv2.imshow(label, i)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
        return key

    @staticmethod
    def rotate_imutils(picture, angle, bgr=None, center=None, scale=1.0):
        # IMUTILS: redefine this function
        # Grab the dimensions of the image
        (h, w) = picture.shape[:2]
        # If the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)
        # Perform the rotation
        m = cv2.getRotationMatrix2D(center, angle, scale)
        if bgr:
            result = cv2.warpAffine(picture, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bgr)
        else:
            result = cv2.warpAffine(picture, m, (w, h))
        return result

    @staticmethod
    def rotate_bound(picture, angle, bgr=None):
        # IMUTILS: redefine this function
        # Grab the dimensions of the image and then determine the
        # center
        (h, w) = picture.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # Grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        m = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        # Compute the new bounding dimensions of the image
        n_w = int((h * sin) + (w * cos))
        n_h = int((h * cos) + (w * sin))
        # Adjust the rotation matrix to take into account translation
        m[0, 2] += (n_w / 2) - cX
        m[1, 2] += (n_h / 2) - cY
        # Perform the actual rotation and return the image
        # Custom: I append the possibility to change the background color fix
        if bgr:
            result = cv2.warpAffine(picture, m, (n_w, n_h), borderMode=cv2.BORDER_CONSTANT, borderValue=bgr)
        else:
            result = cv2.warpAffine(picture, m, (n_w, n_h))
        return result


class Video:

    def __init__(self,
                 frames,
                 filename='out.mp4',
                 format_size=(1024, 640),
                 loops=1,
                 fps=30,
                 background_bgr=(255, 255, 255),
                 clockwise_rotation=False,
                 music=False,
                 music_filename=None,
                 ffmpeg='ffmpeg'
                 ):
        self.frames = frames                                        # Frames of video
        self.format_size = format_size                              # Video Resolution
        self.ffmpeg = ffmpeg                                        # Which ffmpeg use
        self.background_BGR = background_bgr
        self.fps = int(fps)
        self.loops = int(loops)                                     # Number of loops of disco rotation
        self.filename = "out.avi"
        self.music = music
        self.music_filename = music_filename
        self.filename = filename
        self.size = (0, 0)
        self.update_size()
        self.seconds = len(self.frames) * self.loops / self.fps     # Video's duration
        if clockwise_rotation:
            self.frames.reverse()

    def __str__(self):
        n_frames = len(self.frames)
        info = [
            ['Format', self.format_size],
            ['Audio', self.music_filename],
            ['Frames', n_frames],
            ['Total frames generated:', n_frames * self.loops],
            ['Estimated duration (seconds)', '{:.2f}'.format(self.seconds)],
            ['Estimated duration (minutes)', '{:.2f}'.format(self.seconds / 60)]
        ]
        string = "Video Object\n"
        string += tabulate(info)
        return string

    def __call__(self, *args, **kwargs):
        # Check if frames shape and format size are the same
        if np.array_equal(self.size, self.format_size) is False:
            self.add_padding()
        # Select format
        extension = self.filename.split('.')[-1]
        if 'avi' == extension:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif 'mp4' == extension:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            print("Format {} not supported. Exit.".format(extension))
            return
        out = cv2.VideoWriter(self.filename, fourcc, fps=self.fps, frameSize=self.size)
        print("\nWriting video:")
        for _ in tqdm(range(self.loops)):
            for frame in self.frames:
                out.write(frame)
        out.release()
        if self.music:
            self.add_audio()
        # If it's mp4, change codec
        if 'mp4' == self.filename.split('.')[-1]:
            self.convert_to_h264()
        print("\nVideo terminated > {}".format(self.filename))

    def insert_frame(self, frame):
        self.frames.insert(0, frame)

    def get_seconds(self):
        return self.seconds

    def update_size(self):
        height, width, _ = self.frames[0].shape
        self.size = (width, height)

    def add_padding(self, img=None):
        frames = []
        base_size = (self.format_size[0], self.format_size[1], 3)
        base = np.zeros(base_size, dtype=np.uint8)
        cv2.rectangle(base, (0, 0), (self.format_size[1], self.format_size[0]), self.background_BGR, -1)
        x0 = int(base_size[1]/2 - self.size[0]/2)
        x1 = int(base_size[1]/2 + self.size[0]/2)
        y0 = int(base_size[0]/2 - self.size[1]/2)
        y1 = int(base_size[0]/2 + self.size[1]/2)
        if img is not None:
            base[y0:y1, x0:x1] = img
            return base
        print("\nAdd padding to frames:")
        # Create loading bar
        for frame in tqdm(self.frames):
            tmp = base.copy()
            tmp[y0:y1, x0:x1] = frame
            frames.append(tmp)
        self.frames = frames
        self.update_size()
        sleep(0.5)
        print("Done")

    def add_audio(self):
        if not Video.check_filename(self.music_filename):
            print("Audio not added to the file")
            return
        args = ['-i', self.music_filename, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest']
        self.execute_ffmpeg(args)

    def convert_to_h264(self):
        # Convert to H264 codec
        args = ['-vcodec', 'libx264']
        self.execute_ffmpeg(args)

    def execute_ffmpeg(self, args, tmp_args_idx=-1):
        tmp_filename = 'tmp_output.{}'.format(self.filename.split('.')[-1])
        if not Video.check_filename(tmp_filename, must_exists=False) or not Video.check_filename(self.filename, must_exists=False):
            print("Skip ffmpeg")
            return
        if isfile(tmp_filename):
            remove(tmp_filename)
        if tmp_args_idx < 0:
            args.append(tmp_filename)
        else:
            args.insert(tmp_args_idx, tmp_filename)
        args.insert(0, self.ffmpeg)
        args.insert(1, '-i')
        args.insert(2, self.filename)
        error_str = "ffmpeg error ({})".format(' '.join(args))
        try:
            if call(args):
                print(error_str)
        except Exception as e:
            print("{}\nException:\n{}".format(error_str, e))
        try:
            remove(self.filename)
            rename(tmp_filename, self.filename)
        except OSError as e:
            print("Something is gone wrong at the end\nException\n:{}".format(e))

    @staticmethod
    def check_filename(filename, must_exists=True):
        if filename.find('&') >= 0 or filename.find('|') >= 0 or filename.find(';') >= 0:
            print("Filenames passed to ffmpeg can't contain the following characters: & | ;")
            return False
        return not must_exists or isfile(filename)
